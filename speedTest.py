#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           HessGPT — DIAGNOSTIC VITESSE COMPLET                             ║
║                                                                              ║
║  Lance ce script AVANT pretrain.py pour identifier exactement               ║
║  d'où vient le bottleneck.                                                  ║
║                                                                              ║
║  Tests effectués :                                                           ║
║    1. GPU info + capacité théorique                                          ║
║    2. Flash Attention disponibilité réelle                                   ║
║    3. soft_cap : chemin manuel vs flex_attention                             ║
║    4. Throughput forward seul (sans backward)                                ║
║    5. Throughput forward + backward                                          ║
║    6. Throughput forward + backward + optimizer step                         ║
║    7. Muon Newton-Schulz : device, dtype, temps par step                    ║
║    8. DataLoader : temps de chargement / prefetch                            ║
║    9. torch.compile : gain réel                                              ║
║   10. Résumé + recommandations automatiques                                  ║
║                                                                              ║
║  USAGE :                                                                     ║
║    python benchmark_speed.py                                                 ║
║    python benchmark_speed.py --seq-len 512                                  ║
║    python benchmark_speed.py --batch-size 28 --seq-len 512 --no-compile     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import sys
import os
import argparse
import gc
import traceback
from typing import Optional, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='HessGPT Speed Benchmark')
parser.add_argument('--batch-size',  type=int,   default=28)
parser.add_argument('--seq-len',     type=int,   default=512)
parser.add_argument('--embed-dim',   type=int,   default=1280)
parser.add_argument('--num-heads',   type=int,   default=20)
parser.add_argument('--num-layers',  type=int,   default=24)
parser.add_argument('--n-kv-heads',  type=int,   default=5)
parser.add_argument('--vocab-size',  type=int,   default=128256)
parser.add_argument('--soft-cap',    type=float, default=30.0)
parser.add_argument('--no-compile',  action='store_true')
parser.add_argument('--warmup-steps',type=int,   default=5,
                    help='Steps de chauffe avant mesure')
parser.add_argument('--bench-steps', type=int,   default=20,
                    help='Steps mesurés pour chaque benchmark')
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
RESET  = '\033[0m'
BOLD   = '\033[1m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
CYAN   = '\033[96m'

def ok(msg):    print(f"  {GREEN}✅ {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠️  {msg}{RESET}")
def bad(msg):   print(f"  {RED}❌ {msg}{RESET}")
def info(msg):  print(f"  {CYAN}ℹ️  {msg}{RESET}")
def header(msg):print(f"\n{BOLD}{'═'*70}\n  {msg}\n{'═'*70}{RESET}")

results = {}   # stocke tous les résultats pour le résumé final

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timeit(fn, warmup=args.warmup_steps, steps=args.bench_steps):
    """Lance fn() warmup fois, puis mesure steps fois. Retourne ms/step."""
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    sync()
    return (time.perf_counter() - t0) / steps * 1000   # ms

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype  = torch.bfloat16

print(f"\n{BOLD}HessGPT Speed Benchmark{RESET}")
print(f"  batch={args.batch_size}  seq={args.seq_len}  embed={args.embed_dim}  "
      f"layers={args.num_layers}  heads={args.num_heads}  kv={args.n_kv_heads}")
print(f"  soft_cap={args.soft_cap}  compile={'off' if args.no_compile else 'on'}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1 — GPU INFO
# ═════════════════════════════════════════════════════════════════════════════
header("TEST 1 — GPU Info & Capacité Théorique")

if device == 'cpu':
    bad("Pas de GPU détecté — tous les benchmarks seront faussés")
    results['gpu'] = 'cpu'
else:
    props = torch.cuda.get_device_properties(0)
    vram  = props.total_memory / 1e9
    name  = props.name

    print(f"  GPU       : {name}")
    print(f"  VRAM      : {vram:.0f} GB")
    print(f"  SM count  : {props.multi_processor_count}")
    print(f"  CUDA caps : {props.major}.{props.minor}")
    print(f"  PyTorch   : {torch.__version__}")

    # Tflops théoriques approximatifs selon le GPU
    tflops_map = {
        'B200': 2250, 'H200': 989, 'H100': 989,
        'A100': 312,  'A6000': 154, '4090': 165,
    }
    tflops = next((v for k, v in tflops_map.items() if k in name), None)

    params = (args.vocab_size * args.embed_dim +          # embeddings
              args.num_layers * (
                  args.embed_dim * args.num_heads * (args.embed_dim // args.num_heads) * 3 +  # QKV
                  args.embed_dim * args.embed_dim +        # out_proj
                  args.embed_dim * int(8 * args.embed_dim / 3 / 64) * 64 * 3  # FFN SwiGLU
              ))
    params_B = params / 1e9

    print(f"\n  Params estimés : {params_B:.2f}B")

    if tflops:
        # Forward+backward ~= 6 FLOPs/param/token
        flops_per_token = 6 * params
        tokens_per_sec_theory = (tflops * 1e12) / flops_per_token
        print(f"  TFLOPs théo   : {tflops} TF/s (bfloat16)")
        print(f"  Tokens/sec    : {tokens_per_sec_theory/1e6:.0f}M théo (100% MFU)")
        print(f"  Tokens/sec    : {tokens_per_sec_theory*0.4/1e6:.0f}M réaliste (40% MFU)")
        print(f"  2B tokens ETA : {2e9/(tokens_per_sec_theory*0.4)/3600:.1f}h à 40% MFU")
        results['tflops_theory']   = tflops
        results['tokens_theory']   = tokens_per_sec_theory
        results['params_B']        = params_B
    else:
        warn(f"GPU '{name}' inconnu dans la table TFLOPs — calcul MFU impossible")

    results['gpu_name'] = name
    results['vram_gb']  = vram


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2 — FLASH ATTENTION
# ═════════════════════════════════════════════════════════════════════════════
header("TEST 2 — Flash Attention Disponibilité")

# 2a — SDPA (PyTorch built-in Flash)
try:
    F.scaled_dot_product_attention
    ok("F.scaled_dot_product_attention disponible (PyTorch Flash)")
    results['sdpa'] = True
except AttributeError:
    bad("F.scaled_dot_product_attention ABSENT — PyTorch < 2.0 ?")
    results['sdpa'] = False

# 2b — flex_attention (PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import flex_attention
    ok("flex_attention disponible (PyTorch >= 2.5) — soft_cap sans overhead !")
    results['flex_attn'] = True
except ImportError:
    warn("flex_attention ABSENT (PyTorch < 2.5) — soft_cap force chemin manuel")
    results['flex_attn'] = False

# 2c — Vérifier si soft_cap désactive Flash dans le code actuel
print(f"\n  Avec soft_cap={args.soft_cap} dans attention.py actuel :")
if args.soft_cap is not None:
    bad(f"soft_cap={args.soft_cap} → Flash DÉSACTIVÉ → chemin manuel lent !")
    info("Fix : utiliser flex_attention avec score_mod pour garder Flash + soft_cap")
else:
    ok("soft_cap=None → Flash Attention activée")

results['soft_cap_kills_flash'] = (args.soft_cap is not None) and (not results.get('flex_attn', False))


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3 — ATTENTION : FLASH vs MANUEL vs FLEX
# ═════════════════════════════════════════════════════════════════════════════
header("TEST 3 — Attention : Flash vs Manuel vs Flex")

B  = args.batch_size
S  = args.seq_len
H  = args.num_heads
Hk = args.n_kv_heads
D  = args.embed_dim // args.num_heads

q  = torch.randn(B, H,  S, D, device=device, dtype=dtype)
k  = torch.randn(B, Hk, S, D, device=device, dtype=dtype)
v  = torch.randn(B, Hk, S, D, device=device, dtype=dtype)

# Expand KV pour GQA
num_queries_per_kv = H // Hk
k_expanded = k.repeat_interleave(num_queries_per_kv, dim=1)
v_expanded = v.repeat_interleave(num_queries_per_kv, dim=1)

scale = 1.0 / math.sqrt(D)

# 3a — Flash Attention (SDPA)
if results.get('sdpa'):
    try:
        def bench_flash():
            return F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                is_causal=True, scale=scale
            )
        ms_flash = timeit(bench_flash)
        ok(f"Flash SDPA      : {ms_flash:.2f} ms/step")
        results['ms_flash'] = ms_flash
    except Exception as e:
        bad(f"Flash SDPA échoue : {e}")
        results['ms_flash'] = None

# 3b — Chemin manuel (ce qui tourne avec soft_cap actif)
try:
    causal_mask = torch.triu(
        torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1
    )
    def bench_manual():
        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
        if args.soft_cap:
            scores = args.soft_cap * torch.tanh(scores / args.soft_cap)
        scores = scores.masked_fill(causal_mask[None, None], float('-inf'))
        w = F.softmax(scores, dim=-1)
        return torch.matmul(w, v_expanded)

    ms_manual = timeit(bench_manual)
    if results.get('ms_flash'):
        ratio = ms_manual / results['ms_flash']
        color = GREEN if ratio < 1.5 else (YELLOW if ratio < 3 else RED)
        print(f"  {color}{'✅' if ratio < 1.5 else ('⚠️ ' if ratio < 3 else '❌')} "
              f"Chemin manuel   : {ms_manual:.2f} ms/step  "
              f"({ratio:.1f}x plus lent que Flash){RESET}")
    else:
        info(f"Chemin manuel   : {ms_manual:.2f} ms/step")
    results['ms_manual'] = ms_manual
except Exception as e:
    bad(f"Chemin manuel échoue : {e}")

# 3c — flex_attention avec soft_cap
if results.get('flex_attn'):
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask

        soft_cap_val = args.soft_cap or 30.0

        def soft_cap_score_mod(score, b, h, q_idx, kv_idx):
            return soft_cap_val * torch.tanh(score / soft_cap_val)

        def causal_mask_fn(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(causal_mask_fn, B=None, H=None, Q_LEN=S, KV_LEN=S)

        def bench_flex():
            return flex_attention(
                q, k_expanded, v_expanded,
                score_mod=soft_cap_score_mod,
                block_mask=block_mask,
                scale=scale,
            )

        ms_flex = timeit(bench_flex)
        ratio_vs_flash  = ms_flex / results['ms_flash'] if results.get('ms_flash') else None
        ratio_vs_manual = ms_flex / results['ms_manual'] if results.get('ms_manual') else None
        ok(f"flex_attention  : {ms_flex:.2f} ms/step"
           + (f"  ({ratio_vs_flash:.2f}x vs Flash, {results['ms_manual']/ms_flex:.1f}x vs manuel)"
              if ratio_vs_manual else ""))
        results['ms_flex'] = ms_flex
    except Exception as e:
        warn(f"flex_attention échoue : {e}")
        results['ms_flex'] = None
else:
    info("flex_attention non disponible — skip")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4 — MODÈLE COMPLET : FORWARD SEUL
# ═════════════════════════════════════════════════════════════════════════════
header("TEST 4 — Forward seul (sans backward)")

# Import du modèle
sys.path.extend(['./Core/Model', './Core/Attention', './Core/FeedForward', './Core/TransformerBlock', '.'])
try:
    from HessGpt import HessGPT
    MODEL_AVAILABLE = True
except ImportError as e:
    warn(f"HessGPT non importable ({e}) — utilisation d'un modèle stub pour les tests attention")
    MODEL_AVAILABLE = False

if MODEL_AVAILABLE and device == 'cuda':
    # Modèle SANS soft_cap (Flash activée)
    model_flash = HessGPT(
        vocab_size    = args.vocab_size,
        embed_dim     = args.embed_dim,
        num_heads     = args.num_heads,
        num_layers    = args.num_layers,
        max_seq_len   = args.seq_len,
        dropout       = 0.0,
        use_rope      = True,
        use_swiglu    = True,
        n_kv_heads    = args.n_kv_heads,
        use_qk_norm   = True,
        soft_cap      = None,          # ← Flash activée
        use_flash_attn= True,
    ).to(device).to(dtype)
    model_flash.eval()

    # Modèle AVEC soft_cap (chemin manuel)
    model_softcap = HessGPT(
        vocab_size    = args.vocab_size,
        embed_dim     = args.embed_dim,
        num_heads     = args.num_heads,
        num_layers    = args.num_layers,
        max_seq_len   = args.seq_len,
        dropout       = 0.0,
        use_rope      = True,
        use_swiglu    = True,
        n_kv_heads    = args.n_kv_heads,
        use_qk_norm   = True,
        soft_cap      = args.soft_cap,  # ← Manuel
        use_flash_attn= True,
    ).to(device).to(dtype)
    model_softcap.eval()

    x_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    print(f"\n  {model_flash.count_parameters()['total']/1e6:.1f}M paramètres")

    with torch.no_grad():
        # Flash
        def fwd_flash():
            return model_flash(x_ids)
        ms_fwd_flash = timeit(fwd_flash)
        toks_flash = args.batch_size * args.seq_len / (ms_fwd_flash / 1000)
        ok(f"Forward Flash    : {ms_fwd_flash:.1f} ms  →  {toks_flash/1e6:.0f}M tok/s")
        results['ms_fwd_flash'] = ms_fwd_flash

        # Soft cap manuel
        def fwd_manual():
            return model_softcap(x_ids)
        ms_fwd_manual = timeit(fwd_manual)
        toks_manual = args.batch_size * args.seq_len / (ms_fwd_manual / 1000)
        ratio = ms_fwd_manual / ms_fwd_flash
        color = GREEN if ratio < 1.3 else (YELLOW if ratio < 2 else RED)
        print(f"  {color}{'✅' if ratio < 1.3 else ('⚠️ ' if ratio < 2 else '❌')} "
              f"Forward Manuel   : {ms_fwd_manual:.1f} ms  →  {toks_manual/1e6:.0f}M tok/s  "
              f"({ratio:.2f}x plus lent){RESET}")
        results['ms_fwd_manual'] = ms_fwd_manual
        results['fwd_slowdown']  = ratio

    del model_flash, model_softcap
    gc.collect()
    torch.cuda.empty_cache()
else:
    if not MODEL_AVAILABLE:
        warn("Modèle non disponible — test forward skippé")
    else:
        warn("CPU détecté — test forward skippé")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5 — FORWARD + BACKWARD
# ═════════════════════════════════════════════════════════════════════════════
header("TEST 5 — Forward + Backward (sans optimizer)")

if MODEL_AVAILABLE and device == 'cuda':

    model_bwd = HessGPT(
        vocab_size    = args.vocab_size,
        embed_dim     = args.embed_dim,
        num_heads     = args.num_heads,
        num_layers    = args.num_layers,
        max_seq_len   = args.seq_len,
        dropout       = 0.0,
        use_rope      = True,
        use_swiglu    = True,
        n_kv_heads    = args.n_kv_heads,
        use_qk_norm   = True,
        soft_cap      = args.soft_cap,
        use_flash_attn= True,
    ).to(device).to(dtype)
    model_bwd.train()

    y_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    def fwd_bwd():
        with torch.amp.autocast(device, dtype=dtype):
            _, loss, _ = model_bwd(x_ids, targets=y_ids)
        loss.backward()
        model_bwd.zero_grad(set_to_none=True)

    ms_fwd_bwd = timeit(fwd_bwd)
    toks_fwd_bwd = args.batch_size * args.seq_len / (ms_fwd_bwd / 1000)
    ok(f"Forward+Backward : {ms_fwd_bwd:.1f} ms  →  {toks_fwd_bwd/1e6:.0f}M tok/s")

    if results.get('ms_fwd_manual'):
        bwd_overhead = (ms_fwd_bwd - results['ms_fwd_manual']) / results['ms_fwd_manual'] * 100
        info(f"Overhead backward : +{bwd_overhead:.0f}% vs forward seul (attendu ~150-200%)")

    results['ms_fwd_bwd']    = ms_fwd_bwd
    results['toks_fwd_bwd']  = toks_fwd_bwd

    del model_bwd
    gc.collect()
    torch.cuda.empty_cache()
else:
    warn("Skip — modèle non disponible ou CPU")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6 — FORWARD + BACKWARD + OPTIMIZER
# ═════════════════════════════════════════════════════════════════════════════
header("TEST 6 — Forward + Backward + Optimizer Step (Muon + AdamW)")

# Implémentation inline Muon pour ne pas dépendre de pretrain.py
def zeropower_via_newtonschulz5(G, steps=5):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + 1e-7)
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1): X = X.T
    return X.to(G.dtype)

class MuonBench(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.grad.ndim < 2: continue
                g = p.grad
                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(g)
                buf = state['buf']
                buf.mul_(group['momentum']).add_(g)
                g = zeropower_via_newtonschulz5(buf, steps=group['ns_steps'])
                g = g * (max(g.size(0), g.size(1)) ** 0.5)
                p.add_(g, alpha=-group['lr'])

if MODEL_AVAILABLE and device == 'cuda':
    model_opt = HessGPT(
        vocab_size    = args.vocab_size,
        embed_dim     = args.embed_dim,
        num_heads     = args.num_heads,
        num_layers    = args.num_layers,
        max_seq_len   = args.seq_len,
        dropout       = 0.0,
        use_rope      = True,
        use_swiglu    = True,
        n_kv_heads    = args.n_kv_heads,
        use_qk_norm   = True,
        soft_cap      = args.soft_cap,
        use_flash_attn= True,
    ).to(device).to(dtype)
    model_opt.train()

    muon_params  = [p for n, p in model_opt.named_parameters()
                    if p.requires_grad and p.dim() >= 2 and n.startswith('blocks.')]
    other_params = [p for n, p in model_opt.named_parameters()
                    if p.requires_grad and not (p.dim() >= 2 and n.startswith('blocks.'))]

    print(f"\n  Muon  : {len(muon_params)} tenseurs 2D dans blocks")
    print(f"  AdamW : {len(other_params)} autres paramètres")

    # Test avec différents ns_steps
    for ns in [3, 5]:
        muon_opt  = MuonBench(muon_params,  lr=0.002, ns_steps=ns)
        adamw_opt = torch.optim.AdamW(other_params, lr=4e-4, fused=True)

        def full_step():
            with torch.amp.autocast(device, dtype=dtype):
                _, loss, _ = model_opt(x_ids, targets=y_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_opt.parameters(), 1.0)
            muon_opt.step()
            adamw_opt.step()
            muon_opt.zero_grad(set_to_none=True)
            adamw_opt.zero_grad(set_to_none=True)

        ms_full = timeit(full_step, warmup=3, steps=10)
        toks_full = args.batch_size * args.seq_len / (ms_full / 1000)
        ok(f"ns_steps={ns} : {ms_full:.1f} ms/step  →  {toks_full/1e6:.0f}M tok/s")

        if ns == 5: results['ms_full_ns5'] = ms_full
        if ns == 3: results['ms_full_ns3'] = ms_full

        del muon_opt, adamw_opt

    # Mesurer le temps Muon seul (sans forward/backward)
    # D'abord générer des gradients
    with torch.amp.autocast(device, dtype=dtype):
        _, loss, _ = model_opt(x_ids, targets=y_ids)
    loss.backward()

    muon_only = MuonBench(muon_params, lr=0.002, ns_steps=5)

    def muon_step_only():
        muon_only.step()

    ms_muon_only = timeit(muon_step_only, warmup=2, steps=10)
    if results.get('ms_full_ns5'):
        pct_muon = ms_muon_only / results['ms_full_ns5'] * 100
        info(f"Muon step seul  : {ms_muon_only:.1f} ms  ({pct_muon:.0f}% du step total)")
        results['ms_muon_only'] = ms_muon_only
        results['pct_muon']     = pct_muon

    del model_opt, muon_only
    gc.collect()
    torch.cuda.empty_cache()
else:
    warn("Skip — modèle non disponible ou CPU")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 7 — MUON INTERNALS : device, dtype, syncs
# ═════════════════════════════════════════════════════════════════════════════
header("TEST 7 — Muon Newton-Schulz Internals")

# Simule un gradient réaliste (taille d'une couche attention)
shapes = [
    (args.embed_dim, args.embed_dim),                # q_proj
    (args.embed_dim, args.n_kv_heads * (args.embed_dim // args.num_heads)),  # k_proj
    (int(8 * args.embed_dim / 3 / 64) * 64, args.embed_dim),  # gate_proj
]

print(f"\n  Test Newton-Schulz sur shapes réalistes :")
for shape in shapes:
    G = torch.randn(*shape, device=device, dtype=dtype)
    print(f"\n  Shape {shape[0]}×{shape[1]} :")
    print(f"    device={G.device}  dtype={G.dtype}")

    for ns in [3, 5]:
        def bench_ns():
            return zeropower_via_newtonschulz5(G, steps=ns)
        ms_ns = timeit(bench_ns, warmup=2, steps=20)
        color = GREEN if ms_ns < 1.0 else (YELLOW if ms_ns < 5.0 else RED)
        print(f"  {color}  ns_steps={ns} : {ms_ns:.3f} ms{RESET}")

# Vérifier si norm() force une sync CPU
print(f"\n  Test sync CPU via .norm() :")
G_test = torch.randn(args.embed_dim, args.embed_dim, device=device, dtype=dtype)

t0 = time.perf_counter()
for _ in range(100):
    n = G_test.norm()    # tensor sur GPU
t1 = time.perf_counter()
ms_norm_gpu = (t1 - t0) / 100 * 1000

t0 = time.perf_counter()
for _ in range(100):
    n = G_test.norm().item()  # force sync CPU
t1 = time.perf_counter()
ms_norm_cpu = (t1 - t0) / 100 * 1000

info(f".norm() (GPU tensor) : {ms_norm_gpu*1000:.1f} µs")
if ms_norm_cpu > ms_norm_gpu * 5:
    bad(f".norm().item() (sync CPU)  : {ms_norm_cpu*1000:.1f} µs  "
        f"({ms_norm_cpu/ms_norm_gpu:.0f}x plus lent — évite .item() dans la boucle Muon !)")
else:
    ok(f".norm().item() (sync CPU)  : {ms_norm_cpu*1000:.1f} µs  (overhead acceptable)")

results['norm_gpu_us'] = ms_norm_gpu * 1000
results['norm_cpu_us'] = ms_norm_cpu * 1000


# ═════════════════════════════════════════════════════════════════════════════
# TEST 8 — DATALOADER
# ═════════════════════════════════════════════════════════════════════════════
header("TEST 8 — DataLoader Throughput")

import numpy as np
from torch.utils.data import Dataset, DataLoader

class FakeChunkDataset(Dataset):
    """Dataset synthétique qui simule un chunk de tokens en RAM."""
    def __init__(self, n_samples, seq_len):
        self.seq_len = seq_len
        total = n_samples * (seq_len + 1)
        tokens = torch.arange(total, dtype=torch.long) % 1000
        self.tokens = tokens.share_memory_()
        self.n = n_samples
    def __len__(self): return self.n
    def __getitem__(self, idx):
        s = idx * (self.seq_len + 1)
        c = self.tokens[s:s + self.seq_len + 1]
        return c[:-1].clone(), c[1:].clone()

n_fake_samples = 4_000_000  # ~2B tokens
fake_ds = FakeChunkDataset(n_fake_samples, args.seq_len)

print(f"\n  Dataset synthétique : {n_fake_samples:,} samples × {args.seq_len} → "
      f"{n_fake_samples * args.seq_len / 1e6:.0f}M tokens")

best_config = None
best_throughput = 0

for num_workers, persistent, prefetch in [
    (0,  False, None),
    (4,  False, 2),
    (8,  True,  2),
    (8,  True,  4),
    (16, True,  4),
]:
    loader_kwargs = dict(
        batch_size         = args.batch_size,
        shuffle            = True,
        drop_last          = True,
        pin_memory         = (device == 'cuda'),
        num_workers        = num_workers,
        persistent_workers = persistent and num_workers > 0,
    )
    if prefetch and num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch

    try:
        loader = DataLoader(fake_ds, **loader_kwargs)

        t0 = time.perf_counter()
        batches_measured = 0
        for x, y in loader:
            if device == 'cuda':
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                torch.cuda.synchronize()
            batches_measured += 1
            if batches_measured >= 50:
                break
        elapsed = time.perf_counter() - t0

        toks = batches_measured * args.batch_size * args.seq_len
        throughput = toks / elapsed / 1e6

        label = f"workers={num_workers:2d}  persistent={'Y' if persistent else 'N'}  prefetch={prefetch}"
        if throughput > best_throughput:
            best_throughput = throughput
            best_config = label
            ok(f"{label}  →  {throughput:.0f}M tok/s  ← BEST")
        else:
            ratio = throughput / best_throughput
            color = GREEN if ratio > 0.9 else (YELLOW if ratio > 0.6 else RED)
            print(f"  {color}  {label}  →  {throughput:.0f}M tok/s{RESET}")

        if hasattr(loader, '_iterator'):
            loader._iterator = None
        del loader

    except Exception as e:
        warn(f"workers={num_workers} échoue : {e}")

results['best_dataloader_config'] = best_config
results['best_dataloader_toks']   = best_throughput

if device == 'cuda' and results.get('toks_fwd_bwd'):
    ratio_dl_vs_gpu = best_throughput / (results['toks_fwd_bwd'] / 1e6)
    if ratio_dl_vs_gpu < 2:
        bad(f"DataLoader ({best_throughput:.0f}M tok/s) trop proche du GPU "
            f"({results['toks_fwd_bwd']/1e6:.0f}M tok/s) — risque de starve GPU !")
    else:
        ok(f"DataLoader {ratio_dl_vs_gpu:.1f}x plus rapide que le GPU — pas de bottleneck I/O")

del fake_ds
gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# TEST 9 — torch.compile
# ═════════════════════════════════════════════════════════════════════════════
header("TEST 9 — torch.compile Gain Réel")

if MODEL_AVAILABLE and device == 'cuda' and not args.no_compile:
    model_compile = HessGPT(
        vocab_size    = args.vocab_size,
        embed_dim     = args.embed_dim,
        num_heads     = args.num_heads,
        num_layers    = args.num_layers,
        max_seq_len   = args.seq_len,
        dropout       = 0.0,
        use_rope      = True,
        use_swiglu    = True,
        n_kv_heads    = args.n_kv_heads,
        use_qk_norm   = True,
        soft_cap      = args.soft_cap,
        use_flash_attn= True,
    ).to(device).to(dtype)
    model_compile.train()

    adamw_ref = torch.optim.AdamW(model_compile.parameters(), lr=4e-4, fused=True)

    # Baseline sans compile
    def step_no_compile():
        with torch.amp.autocast(device, dtype=dtype):
            _, loss, _ = model_compile(x_ids, targets=y_ids)
        loss.backward()
        adamw_ref.step()
        adamw_ref.zero_grad(set_to_none=True)

    ms_no_compile = timeit(step_no_compile, warmup=3, steps=10)
    info(f"Sans compile    : {ms_no_compile:.1f} ms/step")

    # Avec compile
    print(f"  Compilation en cours (peut prendre 1-2 min)...")
    try:
        model_compiled = torch.compile(model_compile, mode='default')
        adamw_comp = torch.optim.AdamW(model_compiled.parameters(), lr=4e-4, fused=True)

        def step_compile():
            with torch.amp.autocast(device, dtype=dtype):
                _, loss, _ = model_compiled(x_ids, targets=y_ids)
            loss.backward()
            adamw_comp.step()
            adamw_comp.zero_grad(set_to_none=True)

        ms_compile = timeit(step_compile, warmup=5, steps=10)
        speedup = ms_no_compile / ms_compile
        if speedup > 1.2:
            ok(f"Avec compile    : {ms_compile:.1f} ms/step  ({speedup:.2f}x speedup ✅)")
        elif speedup > 1.0:
            warn(f"Avec compile    : {ms_compile:.1f} ms/step  ({speedup:.2f}x speedup — marginal)")
        else:
            bad(f"Avec compile    : {ms_compile:.1f} ms/step  ({speedup:.2f}x — pas de gain !)")
            info("Cause probable : recompilations dues au masque causal dynamique (soft_cap)")

        results['compile_speedup'] = speedup
        results['ms_compiled']     = ms_compile
        del adamw_comp
    except Exception as e:
        bad(f"torch.compile échoue : {e}")
        results['compile_speedup'] = None

    del model_compile, model_compiled, adamw_ref
    gc.collect()
    torch.cuda.empty_cache()
else:
    if args.no_compile:
        info("--no-compile passé, test skippé")
    else:
        warn("Skip — modèle non disponible ou CPU")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 10 — MÉMOIRE GPU
# ═════════════════════════════════════════════════════════════════════════════
header("TEST 10 — Utilisation Mémoire GPU")

if device == 'cuda':
    torch.cuda.empty_cache()
    gc.collect()

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    reserved   = torch.cuda.memory_reserved(0)  / 1e9
    allocated  = torch.cuda.memory_allocated(0) / 1e9

    info(f"VRAM total    : {total_vram:.0f} GB")
    info(f"VRAM reserved : {reserved:.1f} GB")
    info(f"VRAM libre    : {total_vram - reserved:.1f} GB")

    if MODEL_AVAILABLE:
        model_mem = HessGPT(
            vocab_size    = args.vocab_size,
            embed_dim     = args.embed_dim,
            num_heads     = args.num_heads,
            num_layers    = args.num_layers,
            max_seq_len   = args.seq_len,
            dropout       = 0.0,
            use_rope      = True,
            use_swiglu    = True,
            n_kv_heads    = args.n_kv_heads,
            use_qk_norm   = True,
            soft_cap      = args.soft_cap,
            use_flash_attn= True,
        ).to(device).to(dtype)

        torch.cuda.empty_cache()
        mem_model = torch.cuda.memory_allocated(0) / 1e9
        ok(f"Poids modèle  : {mem_model:.2f} GB")

        # Un batch forward+backward
        with torch.amp.autocast(device, dtype=dtype):
            _, loss, _ = model_mem(x_ids, targets=y_ids)
        loss.backward()
        mem_with_grads = torch.cuda.memory_allocated(0) / 1e9
        ok(f"Avec gradients: {mem_with_grads:.2f} GB  "
           f"(activations+grads = +{mem_with_grads - mem_model:.2f} GB)")

        batch_mem = mem_with_grads
        max_batch_possible = int((total_vram * 0.85 - mem_model * 2) /
                                  ((mem_with_grads - mem_model) / args.batch_size))
        info(f"Batch max estimé : ~{max_batch_possible} (à 85% VRAM)")
        results['batch_max_estimate'] = max_batch_possible

        del model_mem
        gc.collect()
        torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL + RECOMMANDATIONS
# ═════════════════════════════════════════════════════════════════════════════
header("RÉSUMÉ & RECOMMANDATIONS")

print(f"\n{'─'*70}")
print(f"  {'RÉSULTATS CLÉS':}")
print(f"{'─'*70}")

# Throughput réel vs théorique
if results.get('toks_fwd_bwd') and results.get('tokens_theory'):
    actual_mfu = results['toks_fwd_bwd'] / results['tokens_theory'] * 100
    color = GREEN if actual_mfu > 35 else (YELLOW if actual_mfu > 15 else RED)
    print(f"\n  {color}MFU réel : {actual_mfu:.1f}%  "
          f"(forward+backward, {results['toks_fwd_bwd']/1e6:.0f}M tok/s){RESET}")
    results['mfu'] = actual_mfu

# ETA avec throughput réel
if results.get('ms_full_ns5'):
    toks_per_step = args.batch_size * args.seq_len
    steps_per_sec = 1000 / results['ms_full_ns5']
    toks_per_sec  = toks_per_step * steps_per_sec
    eta_2b = 2e9 / toks_per_sec / 3600
    print(f"  Throughput réel (avec optimizer) : {toks_per_sec/1e6:.1f}M tok/s")
    print(f"  ETA 2B tokens : {eta_2b:.2f}h")
    results['eta_2b_hours'] = eta_2b

print(f"\n{'─'*70}")
print(f"  PROBLÈMES DÉTECTÉS :")
print(f"{'─'*70}")

issues = []

# soft_cap
if results.get('soft_cap_kills_flash'):
    issues.append(('CRITIQUE', 'soft_cap désactive Flash Attention',
        f"Ralentissement : {results.get('fwd_slowdown', '?'):.2f}x sur forward\n"
        f"    Fix : installer PyTorch >= 2.5 et utiliser flex_attention avec score_mod\n"
        f"    OU : retirer soft_cap en pretrain (moins critique qu'en SFT)"))

# MFU faible
if results.get('mfu') and results['mfu'] < 15:
    issues.append(('CRITIQUE', f"MFU très faible ({results['mfu']:.1f}%)",
        "Le GPU est sous-utilisé — probable bottleneck CPU ou sync"))
elif results.get('mfu') and results['mfu'] < 30:
    issues.append(('WARN', f"MFU moyen ({results['mfu']:.1f}%)",
        "Marge d'amélioration avec torch.compile + batch_size plus grand"))

# compile speedup
if results.get('compile_speedup') and results['compile_speedup'] < 1.1:
    issues.append(('WARN', "torch.compile sans effet",
        "Probable recompilation à cause du masque causal dynamique (soft_cap path)\n"
        "    Fix : pré-allouer le masque dans MultiHeadAttention"))

# Muon overhead
if results.get('pct_muon') and results['pct_muon'] > 30:
    issues.append(('WARN', f"Muon prend {results['pct_muon']:.0f}% du step total",
        "Essaie ns_steps=3 au lieu de 5 (gain ~15%, qualité quasi-identique en pretrain)"))

# DataLoader
if results.get('best_dataloader_toks') and results.get('toks_fwd_bwd'):
    ratio = results['best_dataloader_toks'] / (results['toks_fwd_bwd'] / 1e6)
    if ratio < 2:
        issues.append(('WARN', "DataLoader proche du GPU throughput",
            f"Ratio DataLoader/GPU = {ratio:.1f}x (idéal > 5x)\n"
            f"    Config optimale : {results.get('best_dataloader_config', '?')}"))

# batch size
if results.get('batch_max_estimate') and results.get('batch_max_estimate', 0) > args.batch_size * 2:
    issues.append(('INFO', f"Batch size peut être augmenté",
        f"Batch actuel={args.batch_size}, estimé possible={results['batch_max_estimate']}\n"
        f"    Plus grand batch → moins de steps → moins d'overhead optimizer"))

if not issues:
    ok("Aucun problème majeur détecté !")
else:
    for severity, title, detail in issues:
        if severity == 'CRITIQUE':
            bad(f"[CRITIQUE] {title}")
        elif severity == 'WARN':
            warn(f"[WARN]     {title}")
        else:
            info(f"[INFO]     {title}")
        for line in detail.split('\n'):
            print(f"             {line}")

print(f"\n{'─'*70}")
print(f"  OPTIMISATIONS PRIORITAIRES :")
print(f"{'─'*70}")

prios = []

if results.get('soft_cap_kills_flash') and results.get('flex_attn'):
    prios.append("1. Migrer vers flex_attention (PyTorch >= 2.5) → +2-3x sur l'attention")
elif results.get('soft_cap_kills_flash'):
    prios.append("1. pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124")
    prios.append("   puis migrer vers flex_attention avec score_mod pour soft_cap")

if results.get('ms_full_ns5') and results.get('ms_full_ns3'):
    gain_ns = (results['ms_full_ns5'] - results['ms_full_ns3']) / results['ms_full_ns5'] * 100
    if gain_ns > 5:
        prios.append(f"2. ns_steps=3 dans Muon → +{gain_ns:.0f}% throughput")

if results.get('batch_max_estimate') and results['batch_max_estimate'] > args.batch_size * 1.5:
    prios.append(f"3. batch_size={min(results['batch_max_estimate'], 64)} "
                 f"(actuel={args.batch_size}) → meilleur remplissage GPU")

if not prios:
    ok("Config déjà bien optimisée pour le hardware !")
else:
    for p in prios:
        print(f"  {GREEN}{p}{RESET}")

# ETA comparatif
if results.get('eta_2b_hours'):
    print(f"\n{'─'*70}")
    print(f"  ETA 2B TOKENS :")
    print(f"{'─'*70}")
    print(f"  Actuel (mesuré)   : {results['eta_2b_hours']:.2f}h")
    if results.get('soft_cap_kills_flash') and results.get('fwd_slowdown'):
        eta_fix_flash = results['eta_2b_hours'] / results['fwd_slowdown'] * 1.3
        print(f"  Après fix Flash   : ~{eta_fix_flash:.2f}h  (estimé)")
    if results.get('ms_full_ns5') and results.get('ms_full_ns3'):
        ratio_ns = results['ms_full_ns5'] / results['ms_full_ns3']
        eta_fix_ns = results['eta_2b_hours'] / ratio_ns
        print(f"  Après ns_steps=3  : ~{eta_fix_ns:.2f}h  (estimé)")

print(f"\n{'═'*70}")
print(f"  Benchmark terminé.")
print(f"{'═'*70}\n")

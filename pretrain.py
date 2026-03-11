#!/usr/bin/env python3
"""
HessGPT Pre-Training v6 — LLaMA-3 Tokenizer
FIXES v6 (par rapport à v5) :
  ✅ Dataset entièrement en VRAM — zéro transfert PCIe pendant le training
  ✅ num_workers=0 — inutile quand les données sont déjà en VRAM
  ✅ pin_memory=False — inutile quand les données sont déjà en VRAM
  ✅ Prefetch du chunk suivant en background (threading) pendant le training
  ✅ compile_cache persisté sur disk — évite les 25min de recompile au restart
  ✅ ns_steps=3 hardcodé dans configure_optimizers (était 5 malgré défaut=3)
  ✅ compile_mode='reduce-overhead' — optimisé boucles répétitives B200

FIXES v5 (conservés) :
  ✅ MARS-M intégré dans Muon
  ✅ RMSNorm weights dans blocks → AdamW nodecay
  ✅ WSD scheduler : tag is_muon
  ✅ Shuffle seed variable par chunk
  ✅ Sauvegarde atomique + JSON info
  ✅ Reprise propre par chunk
"""

import torch
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
import math
import json
import gc
import threading
from tqdm import tqdm
from transformers import AutoTokenizer
from datetime import datetime
import traceback
import numpy as np

sys.path.append('./Core/Model')
sys.path.append('./Core/Attention')
sys.path.append('./Core/FeedForward')
sys.path.append('./Core/TransformerBlock')

SPECIAL_TOKENS = ['<code>', '<think>', '</think>']

print("=" * 80)
print("HessGPT v6 — LLaMA-3 | VRAM Dataset | Prefetch | Muon+MARS-M")
print("=" * 80)

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    'vocab_size':            None,
    'embed_dim':             1280,
    'num_heads':             20,
    'num_layers':            24,
    'max_seq_len':           1024,
    'dropout':               0.0,
    'use_rope':              True,
    'use_yarn':              False,
    'yarn_scale':            4.0,
    'yarn_original_max_len': 1024,
    'use_swiglu':            True,
    'n_kv_heads':            5,
    'use_qk_norm':           True,
    'soft_cap':              30.0,
    'use_flash_attn':        True,
    'batch_size':            24,
    'gradient_accumulation': 8,
    'max_grad_norm':         1.0,
    'learning_rate':         4e-4,
    'weight_decay':          0.1,
    'adam_beta1':            0.9,
    'adam_beta2':            0.95,
    'adam_eps':              1e-8,
    'num_epochs':            5,
    'chunks_per_epoch':      3,
    'data_dir':              './data/ultra_filtered',
    'val_tokens':            15_000_000,
    'warmup_ratio':          0.03,
    'decay_ratio':           0.15,
    'min_lr_ratio':          0.1,
    'validate_every_steps':  500,
    'val_batches':           50,
    'shuffle_seed':          42,
    'save_every_steps':      2000,
    'checkpoint_file':       './Model/HessGpt_pretrain.pt',
    'use_compile':           True,
    'compile_mode':          'reduce-overhead',  # ✅ v6
    'compile_cache':         './compile_cache',  # ✅ v6
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ✅ v6 : tf32 — TensorCores B200/H100 sur toutes les matmuls fp32 résiduelles
if device == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high')  # 'high'=tf32  'highest'=fp32 pur

# ✅ v6 : cache compile persisté sur disk
if CONFIG['compile_cache']:
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = CONFIG['compile_cache']
    os.makedirs(CONFIG['compile_cache'], exist_ok=True)

print(f"\nCONFIG :")
print(f"  embed={CONFIG['embed_dim']}  layers={CONFIG['num_layers']}  "
      f"heads={CONFIG['num_heads']}  kv={CONFIG['n_kv_heads']}")
print(f"  batch={CONFIG['batch_size']}  accum={CONFIG['gradient_accumulation']}  "
      f"device={device}")
print(f"  compile_mode={CONFIG['compile_mode']}  cache={CONFIG['compile_cache']}")
print(f"  ✅ Dataset VRAM — zéro transfert PCIe pendant training")
if device == 'cuda':
    print(f"  GPU={torch.cuda.get_device_name(0)}  "
          f"VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")

# ============================================================
# SCAN CHUNKS
# ============================================================
def scan_available_chunks(data_dir):
    available = []
    if not os.path.exists(data_dir):
        return available
    for entry in sorted(os.listdir(data_dir)):
        if not entry.startswith('chunk'):
            continue
        chunk_dir  = os.path.join(data_dir, entry)
        stats_file = os.path.join(chunk_dir, 'stats.json')
        if not os.path.isdir(chunk_dir) or not os.path.exists(stats_file):
            continue
        try:
            with open(stats_file) as f:
                stats = json.load(f)
            npy_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.npy')])
            if not npy_files:
                continue
            cid = int(entry.split('_')[1]) if '_' in entry else int(entry.replace('chunk', ''))
            available.append({'id': cid, 'dir': chunk_dir, 'files': npy_files, 'stats': stats})
        except Exception as e:
            print(f"  skip {entry}: {e}")
    available.sort(key=lambda x: x['id'])
    return available

print(f"\nScan chunks...")
ALL_CHUNKS = scan_available_chunks(CONFIG['data_dir'])
n_chunks   = len(ALL_CHUNKS)
print(f"  {n_chunks} chunks trouvés")

if n_chunks == 0:
    print(f"ERREUR: aucun chunk dans {CONFIG['data_dir']}")
    sys.exit(1)

epochs_possible = n_chunks // CONFIG['chunks_per_epoch']
if epochs_possible == 0:
    print(f"  ERREUR: {n_chunks} chunks < chunks_per_epoch={CONFIG['chunks_per_epoch']}")
    sys.exit(1)
if epochs_possible < CONFIG['num_epochs']:
    print(f"  WARN: seulement {epochs_possible} epochs possibles")
    CONFIG['num_epochs'] = epochs_possible

TOTAL_CHUNKS_USED = CONFIG['num_epochs'] * CONFIG['chunks_per_epoch']
ALL_TRAIN_CHUNKS  = ALL_CHUNKS[:TOTAL_CHUNKS_USED]

print(f"\nPLAN CHUNKS :")
for ep in range(CONFIG['num_epochs']):
    s   = ep * CONFIG['chunks_per_epoch']
    ids = [f"chunk_{c['id']:03d}" for c in ALL_TRAIN_CHUNKS[s:s + CONFIG['chunks_per_epoch']]]
    print(f"  Epoch {ep+1:2d} : {' '.join(ids)}")

def steps_for_chunk(stats):
    samples = stats['total_tokens'] // (CONFIG['max_seq_len'] + 1)
    batches = math.ceil(samples / CONFIG['batch_size'])
    return max(math.ceil(batches / CONFIG['gradient_accumulation']), 1)

TOTAL_STEPS = sum(steps_for_chunk(c['stats']) for c in ALL_TRAIN_CHUNKS)
print(f"\nPLAN STEPS : total={TOTAL_STEPS:,}  save every {CONFIG['save_every_steps']} steps")

# ============================================================
# TOKENIZER
# ============================================================
print(f"\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
CONFIG['vocab_size'] = len(tokenizer)
print(f"  vocab={len(tokenizer)}")

# ============================================================
# WSD SCHEDULER
# ============================================================
class WSDScheduler:
    def __init__(self, optimizers, max_lr, total_steps,
                 warmup_ratio=0.03, decay_ratio=0.15, min_lr_ratio=0.1):
        self.optimizers   = optimizers if isinstance(optimizers, list) else [optimizers]
        self.max_lr       = max_lr
        self.min_lr       = max_lr * min_lr_ratio
        self.total_steps  = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.decay_steps  = int(total_steps * decay_ratio)
        self.stable_steps = total_steps - self.warmup_steps - self.decay_steps
        self.current_step = 0
        print(f"\nWSD LR : warmup={self.warmup_steps:,}  "
              f"stable={self.stable_steps:,}  decay={self.decay_steps:,}")
        print(f"  AdamW {self.min_lr:.2e} → {self.max_lr:.2e}  "
              f"| Muon {self.min_lr*5:.2e} → {self.max_lr*5:.2e}")

    def get_lr(self):
        s = self.current_step
        if s < self.warmup_steps:
            return self.max_lr * (s / max(self.warmup_steps, 1))
        elif s < self.warmup_steps + self.stable_steps:
            return self.max_lr
        else:
            d = s - self.warmup_steps - self.stable_steps
            p = min(d / max(self.decay_steps, 1), 1.0)
            return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * p))

    def step(self):
        lr = self.get_lr()
        self.current_step += 1
        for opt in self.optimizers:
            for pg in opt.param_groups:
                pg['lr'] = lr * 5.0 if pg.get('is_muon', False) else lr
        return lr

    def get_last_lr(self):
        return [self.get_lr()]

    def state_dict(self):
        return {'current_step': self.current_step}

    def load_state_dict(self, sd):
        self.current_step = sd['current_step']

# ============================================================
# DATASET EN VRAM
# ============================================================
class VRAMDataset(Dataset):
    """
    ✅ v6 : Tokens directement en VRAM.
    __getitem__ retourne des slices déjà sur GPU — zéro copie, zéro PCIe.
    Requiert num_workers=0 (pas de fork avec tenseurs CUDA).
    """
    def __init__(self, tokens_cuda: torch.Tensor, seq_len: int):
        self.seq_len     = seq_len
        self.num_samples = len(tokens_cuda) // (seq_len + 1)
        self.tokens      = tokens_cuda[:self.num_samples * (seq_len + 1)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = self.tokens[start:start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


class SeededSampler(torch.utils.data.Sampler):
    def __init__(self, n: int, seed: int, skip_samples: int = 0):
        self.n        = n
        rng           = np.random.default_rng(seed)
        indices       = rng.permutation(n)
        skip          = min(skip_samples, n)
        self._indices = indices[skip:]
        print(f"  SeededSampler : n={n:,}  seed={seed}  "
              f"skip={skip:,}  restant={len(self._indices):,}")

    def __iter__(self):
        return iter(self._indices.tolist())

    def __len__(self):
        return len(self._indices)


class VRAMChunkLoader:
    """
    ✅ v6 : Charge chunk disk → RAM → VRAM en une passe.
    Shuffle fait en RAM avant transfert pour ne pas fragmenter la VRAM.
    """
    def __init__(self, chunk_info, seq_len, pad_token_id,
                 val_tokens=15_000_000, val_seed=0):
        self.seq_len = seq_len
        self._load(chunk_info, val_tokens, val_seed)

    def _load(self, chunk_info, val_tokens, val_seed):
        print(f"  Loading chunk_{chunk_info['id']:03d} → VRAM...")
        t0 = time.time()

        arrays = []
        for fname in chunk_info['files']:
            fpath = os.path.join(chunk_info['dir'], fname)
            try:
                arr = np.load(fpath, mmap_mode='r')
                arrays.append(torch.from_numpy(np.array(arr, dtype=np.int32)).long())
            except Exception as e:
                print(f"    skip {fname}: {e}")

        if not arrays:
            raise ValueError(f"chunk_{chunk_info['id']:03d} : aucun fichier chargé")

        all_tokens = torch.cat(arrays)
        total      = len(all_tokens)

        # Shuffle en RAM
        seq_len_1  = self.seq_len + 1
        n_seqs     = total // seq_len_1
        rng        = np.random.default_rng(val_seed)
        idx        = rng.permutation(n_seqs)
        all_tokens = all_tokens[:n_seqs * seq_len_1].reshape(n_seqs, seq_len_1)[idx].reshape(-1)
        total      = len(all_tokens)

        val_size   = min(val_tokens, int(total * 0.05))
        train_size = total - val_size

        vram_gb = all_tokens.element_size() * total / 1e9
        print(f"  chunk_{chunk_info['id']:03d} : {total/1e6:.0f}M tokens  "
              f"train={train_size/1e6:.0f}M  val={val_size/1e6:.0f}M  "
              f"VRAM={vram_gb:.1f}GB  ({time.time()-t0:.1f}s)")

        # Transfert unique RAM → VRAM
        self._train_toks = all_tokens[:train_size].to(device, non_blocking=True)
        self._val_toks   = all_tokens[train_size:].to(device, non_blocking=True)
        torch.cuda.synchronize()

        del all_tokens
        gc.collect()

    def get_train_dataset(self):
        return VRAMDataset(self._train_toks, self.seq_len)

    def get_val_dataset(self):
        return VRAMDataset(self._val_toks, self.seq_len)

    def unload(self):
        del self._train_toks, self._val_toks
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  VRAM chunk libérée")


# ============================================================
# PREFETCHER — charge le chunk suivant en background
# ============================================================
class ChunkPrefetcher:
    """
    ✅ v6 : Thread daemon qui charge le chunk N+1 pendant que N s'entraîne.
    Le transfert PCIe se fait en overlap avec le compute GPU.
    """
    def __init__(self):
        self._next   = None
        self._thread = None
        self._error  = None

    def prefetch(self, chunk_info, seq_len, pad_token_id, val_tokens, val_seed):
        def _load():
            try:
                self._next = VRAMChunkLoader(chunk_info, seq_len, pad_token_id,
                                             val_tokens, val_seed)
            except Exception as e:
                self._error = e
        self._next   = None
        self._error  = None
        self._thread = threading.Thread(target=_load, daemon=True)
        self._thread.start()

    def get(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            raise self._error
        return self._next

    def cancel(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._next is not None:
            self._next.unload()
            self._next = None


# ============================================================
# CHECKPOINT MANAGER
# ============================================================
class CheckpointManager:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def save(self, model, optimizers, scheduler, metadata):
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        if isinstance(optimizers, (list, tuple)):
            muon_opt, adamw_opt = optimizers
            opt_state = {
                'muon_state_dict':  muon_opt.state_dict(),
                'adamw_state_dict': adamw_opt.state_dict(),
            }
        else:
            opt_state = {'optimizer_state_dict': optimizers.state_dict()}

        cp = {**opt_state, 'model_state_dict': m.state_dict(),
              'scheduler_state_dict': scheduler.state_dict()}
        tmp = self.path + '.tmp'
        torch.save(cp, tmp)
        os.replace(tmp, self.path)

        json_path = self.path.replace('.pt', '_info.json')
        info = {
            'global_step':         metadata['global_step'],
            'chunk_start_step':    metadata.get('chunk_start_step', 0),
            'current_epoch':       metadata['current_epoch'],
            'chunk_within_epoch':  metadata['chunk_within_epoch'],
            'total_training_time': metadata.get('total_training_time', 0.0),
            'last_save':           datetime.now().isoformat(),
            'config':              CONFIG,
            'training_history':    metadata['training_history'],
        }
        with open(json_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)

        print(f"  💾 SAVE → epoch={metadata['current_epoch']}  "
              f"cwi={metadata['chunk_within_epoch']}/{CONFIG['chunks_per_epoch']}  "
              f"step={metadata['global_step']:,}  [{self.path}]")

    def load(self):
        if not os.path.exists(self.path):
            return None
        print(f"\nCheckpoint trouvé : {self.path}")
        cp = torch.load(self.path, map_location='cpu', weights_only=False)
        json_path = self.path.replace('.pt', '_info.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                info = json.load(f)
            cp['global_step']         = info.get('global_step', 0)
            cp['chunk_start_step']    = info.get('chunk_start_step', 0)
            cp['current_epoch']       = info.get('current_epoch', 1)
            cp['chunk_within_epoch']  = info.get('chunk_within_epoch', 0)
            cp['total_training_time'] = info.get('total_training_time', 0.0)
            cp['training_history']    = info.get('training_history',
                                         {'chunks': [], 'validations': [], 'epochs': []})
            cp['last_save']           = info.get('last_save', '?')
            print(f"  [JSON] epoch={cp['current_epoch']}  cwi={cp['chunk_within_epoch']}  "
                  f"step={cp['global_step']:,}  saved={cp['last_save']}")
        return cp


# ============================================================
# VALIDATION
# ============================================================
@torch.no_grad()
def validate(model, val_loader, max_batches=50):
    model.eval()
    total_loss, n = 0.0, 0
    ae  = (device == 'cuda')
    adt = torch.bfloat16 if ae else torch.float32
    try:
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            # x, y déjà en VRAM
            with torch.amp.autocast(device, dtype=adt, enabled=ae):
                _, loss, _ = model(x, targets=y, pad_token_id=tokenizer.pad_token_id)
            total_loss += loss.item()
            n += 1
    finally:
        model.train()
    avg = total_loss / max(n, 1)
    return math.exp(min(avg, 10)), avg


# ============================================================
# MUON + MARS-M
# ============================================================
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 3) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon + MARS-M — identique pretrain v5, ns_steps=3 par défaut."""
    def __init__(self, params, lr=0.02, momentum=0.95,
                 nesterov=True, ns_steps=3, weight_decay=0.0,
                 use_mars=True, mars_gamma=0.025):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        use_mars=use_mars, mars_gamma=mars_gamma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum  = group['lr'], group['momentum']
            nesterov      = group['nesterov']
            ns_steps, wd  = group['ns_steps'], group['weight_decay']
            use_mars      = group.get('use_mars', True)
            mars_gamma    = group.get('mars_gamma', 0.025)

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    continue
                state = self.state[p]

                if use_mars:
                    if 'prev_grad' not in state:
                        state['prev_grad'] = torch.zeros_like(g)
                    prev_g    = state['prev_grad']
                    c_t       = (mars_gamma / (1.0 - mars_gamma)) * (
                                 (g.norm() + 1e-8) / (prev_g.norm() + 1e-8))
                    c_t       = torch.clamp(c_t, max=1.0)
                    g         = g + c_t * (g - prev_g)
                    state['prev_grad'].copy_(p.grad)

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g + momentum * buf if nesterov else buf

                g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                g = g * (max(g.size(0), g.size(1)) ** 0.5)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)


def configure_optimizers(model, lr, weight_decay, betas, eps):
    MUON_EXCLUDE = {'token_embeddings.weight', 'output_head.weight',
                    'position_embeddings.weight'}
    muon_params, adamw_decay, adamw_nodecay = [], [], []

    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if pn in MUON_EXCLUDE:
            (adamw_decay if p.dim() >= 2 else adamw_nodecay).append(p)
        elif p.dim() >= 2 and pn.startswith('blocks.'):
            muon_params.append(p)
        elif p.dim() < 2 and pn.startswith('blocks.'):
            adamw_nodecay.append(p)
        elif p.dim() >= 2:
            adamw_decay.append(p)
        else:
            adamw_nodecay.append(p)

    lr_muon  = lr * 5.0
    muon_opt = Muon(
        [{'params': muon_params, 'is_muon': True}],
        lr=lr_muon, momentum=0.95, nesterov=True,
        ns_steps=3,               # ✅ v6 : 3 hardcodé
        weight_decay=0.0, use_mars=True, mars_gamma=0.025,
    )
    muon_opt.param_groups[0]['is_muon'] = True

    adamw_opt = torch.optim.AdamW(
        [{'params': adamw_decay,   'weight_decay': weight_decay, 'is_muon': False},
         {'params': adamw_nodecay, 'weight_decay': 0.0,          'is_muon': False}],
        lr=lr, betas=betas, eps=eps, fused=(device == 'cuda'),
    )

    print(f"\nOptimizer Muon+MARS-M + AdamW :")
    print(f"  Muon+MARS-M : {len(muon_params):3d} tenseurs  lr={lr_muon:.2e}  "
          f"ns_steps=3  mars_gamma=0.025")
    print(f"  AdamW       : {len(adamw_decay):3d} decay + "
          f"{len(adamw_nodecay):3d} no-decay  lr={lr:.2e}")
    return muon_opt, adamw_opt


# ============================================================
# TRAIN ONE CHUNK
# ============================================================
def train_one_chunk(
    model, cds, chunk_id,
    optimizers, scheduler,
    checkpoint_manager, training_history,
    global_step, total_training_time,
    current_epoch, chunk_within_epoch, chunk_start_step,
    prefetcher, next_chunk_info,
):
    muon_opt, adamw_opt = optimizers
    label        = (f"Epoch {current_epoch}/{CONFIG['num_epochs']} | "
                    f"cwi {chunk_within_epoch+1}/{CONFIG['chunks_per_epoch']} "
                    f"(chunk_{chunk_id:03d})")
    steps_done   = global_step - chunk_start_step
    batches_done = steps_done * CONFIG['gradient_accumulation']
    shuffle_seed = CONFIG['shuffle_seed'] + (current_epoch - 1) * 1000 + chunk_within_epoch

    print(f"\n{'='*80}")
    print(f"  {label}  LR_adamw={scheduler.get_last_lr()[0]:.2e}"
          f"  LR_muon={scheduler.get_last_lr()[0]*5:.2e}")
    if batches_done > 0:
        print(f"  ⏩ Reprise : steps_done={steps_done:,}  batches_done={batches_done:,}")
    print(f"{'='*80}")

    train_ds   = cds.get_train_dataset()
    total_seqs = len(train_ds)

    if batches_done >= math.ceil(total_seqs / CONFIG['batch_size']):
        print(f"  ✅ Chunk déjà traité, skip.")
        return global_step, total_training_time, chunk_start_step

    sampler = SeededSampler(total_seqs, shuffle_seed,
                            skip_samples=batches_done * CONFIG['batch_size'])

    # ✅ v6 : num_workers=0, pin_memory=False — données déjà en VRAM
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              sampler=sampler, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(cds.get_val_dataset(), batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=0, pin_memory=False)

    total_batches = total_seqs // CONFIG['batch_size']
    num_batches   = len(train_loader)
    already_done  = max(total_batches - num_batches, 0)  # ✅ fix tqdm clamping
    print(f"  train={total_batches:,} batches | restant={num_batches:,} | val={len(val_loader):,}")
    print(f"  ✅ Zéro transfert PCIe — tout en VRAM")

    # ✅ Lancer prefetch du chunk suivant dès que le training démarre
    if next_chunk_info is not None and prefetcher is not None:
        next_val_seed = (CONFIG['shuffle_seed'] + (current_epoch - 1) * 1000
                         + chunk_within_epoch + 1)
        print(f"  🔄 Prefetch chunk_{next_chunk_info['id']:03d} en background...")
        prefetcher.prefetch(next_chunk_info, CONFIG['max_seq_len'],
                            tokenizer.pad_token_id, CONFIG['val_tokens'], next_val_seed)

    model.train()
    chunk_loss, valid_batches     = 0.0, 0
    accumulated_steps             = 0
    running_loss, running_batches = 0.0, 0
    t_start = time.time()
    ae  = (device == 'cuda')
    adt = torch.bfloat16 if ae else torch.float32

    pbar = tqdm(train_loader, desc=label, leave=True,
                initial=already_done, total=total_batches)

    for batch_idx, (x, y) in enumerate(pbar):
        try:
            # ✅ x, y déjà en VRAM — pas de .to(device)
            with torch.amp.autocast(device, dtype=adt, enabled=ae):
                _, loss, _ = model(x, targets=y, pad_token_id=tokenizer.pad_token_id)
                loss = loss / CONFIG['gradient_accumulation']

            if torch.isnan(loss) or torch.isinf(loss):
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                accumulated_steps = 0
                continue

            loss.backward()
            accumulated_steps += 1

            is_last = (batch_idx + 1 == num_batches)
            if (accumulated_steps % CONFIG['gradient_accumulation'] == 0) or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                muon_opt.step()
                adamw_opt.step()
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                scheduler.step()
                accumulated_steps = 0
                global_step += 1

                if global_step % CONFIG['validate_every_steps'] == 0:
                    val_ppl, val_loss = validate(model, val_loader, CONFIG['val_batches'])
                    avg = running_loss / max(running_batches, 1)
                    print(f"\n  step={global_step:,} | "
                          f"train={avg:.4f} ppl={math.exp(min(avg,10)):.1f} | "
                          f"val={val_loss:.4f} ppl={val_ppl:.1f} | "
                          f"lr={scheduler.get_last_lr()[0]:.2e}\n")
                    training_history['validations'].append({
                        'step': global_step, 'current_epoch': current_epoch,
                        'chunk_within_epoch': chunk_within_epoch, 'chunk_id': chunk_id,
                        'val_loss': val_loss, 'val_ppl': val_ppl,
                        'train_loss': avg, 'lr': scheduler.get_last_lr()[0],
                    })
                    running_loss, running_batches = 0.0, 0

                if global_step % CONFIG['save_every_steps'] == 0:
                    checkpoint_manager.save(model, optimizers, scheduler, metadata={
                        'current_epoch': current_epoch,
                        'chunk_within_epoch': chunk_within_epoch,
                        'global_step': global_step,
                        'chunk_start_step': chunk_start_step,
                        'total_training_time': total_training_time + (time.time() - t_start),
                        'training_history': training_history,
                    })

            raw = loss.item() * CONFIG['gradient_accumulation']
            chunk_loss      += raw
            running_loss    += raw
            valid_batches   += 1
            running_batches += 1

            if batch_idx % 20 == 0:
                avg = running_loss / max(running_batches, 1)
                pbar.set_postfix(loss=f'{raw:.4f}', avg=f'{avg:.4f}',
                                 ppl=f'{math.exp(min(avg,10)):.1f}',
                                 lr=f'{scheduler.get_last_lr()[0]:.2e}',
                                 step=f'{global_step:,}')

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\n  OOM batch {batch_idx} — skip")
                torch.cuda.empty_cache()
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                accumulated_steps = 0
                gc.collect()
                model.train()
                continue
            raise

    pbar.close()
    elapsed = time.time() - t_start
    total_training_time += elapsed
    avg_loss = chunk_loss / max(valid_batches, 1)
    print(f"\n  chunk_{chunk_id:03d} terminé | loss={avg_loss:.4f} | {elapsed/60:.1f}min")

    training_history['chunks'].append({
        'current_epoch': current_epoch, 'chunk_within_epoch': chunk_within_epoch,
        'chunk_id': chunk_id, 'train_loss': avg_loss,
        'time_sec': elapsed, 'batches': valid_batches, 'global_step': global_step,
    })

    return global_step, total_training_time, chunk_start_step


# ============================================================
# MAIN
# ============================================================
def main():
    from HessGpt import HessGPT

    print('\n' + '='*80 + '\nCREATION MODELE\n' + '='*80)

    ckpt_mgr   = CheckpointManager(CONFIG['checkpoint_file'])
    prefetcher = ChunkPrefetcher()

    model = HessGPT(
        vocab_size=CONFIG['vocab_size'], embed_dim=CONFIG['embed_dim'],
        num_heads=CONFIG['num_heads'], num_layers=CONFIG['num_layers'],
        max_seq_len=CONFIG['max_seq_len'], dropout=CONFIG['dropout'],
        use_rope=CONFIG['use_rope'], use_yarn=CONFIG['use_yarn'],
        yarn_scale=CONFIG['yarn_scale'],
        yarn_original_max_len=CONFIG['yarn_original_max_len'],
        use_swiglu=CONFIG['use_swiglu'], n_kv_heads=CONFIG['n_kv_heads'],
        use_qk_norm=CONFIG['use_qk_norm'], soft_cap=CONFIG['soft_cap'],
        use_flash_attn=CONFIG['use_flash_attn'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params : {total_params/1e6:.1f}M")

    if CONFIG['use_compile'] and device == 'cuda':
        print(f"torch.compile (mode={CONFIG['compile_mode']})...")
        try:
            model = torch.compile(model, mode=CONFIG['compile_mode'])
            print('  OK')
        except Exception as e:
            print(f'  FAIL : {e}')

    raw_model  = model._orig_mod if hasattr(model, '_orig_mod') else model
    optimizers = configure_optimizers(
        raw_model, CONFIG['learning_rate'], CONFIG['weight_decay'],
        (CONFIG['adam_beta1'], CONFIG['adam_beta2']), CONFIG['adam_eps'],
    )
    muon_opt, adamw_opt = optimizers

    scheduler = WSDScheduler(
        list(optimizers), max_lr=CONFIG['learning_rate'],
        total_steps=TOTAL_STEPS, warmup_ratio=CONFIG['warmup_ratio'],
        decay_ratio=CONFIG['decay_ratio'], min_lr_ratio=CONFIG['min_lr_ratio'],
    )

    training_history = {
        'config': CONFIG, 'total_params': total_params, 'total_steps': TOTAL_STEPS,
        'chunks': [], 'validations': [], 'epochs': [],
        'start_time': datetime.now().isoformat(),
    }

    global_step, current_epoch    = 0, 1
    chunk_within_epoch             = 0
    total_training_time            = 0.0
    chunk_start_step               = 0

    cp = ckpt_mgr.load()
    if cp:
        print('\nREPRISE')
        unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
        unwrapped.load_state_dict(cp['model_state_dict'])

        if 'muon_state_dict' in cp and 'adamw_state_dict' in cp:
            muon_opt.load_state_dict(cp['muon_state_dict'])
            adamw_opt.load_state_dict(cp['adamw_state_dict'])
        elif 'optimizer_state_dict' in cp:
            print('  WARN : ancien format — Muon repart de zéro')
            adamw_opt.load_state_dict(cp['optimizer_state_dict'])

        scheduler.load_state_dict(cp['scheduler_state_dict'])
        for pg in muon_opt.param_groups:
            pg['lr'] = scheduler.get_lr() * 5.0
        for pg in adamw_opt.param_groups:
            pg['lr'] = scheduler.get_lr()

        current_epoch       = cp.get('current_epoch', 1)
        chunk_within_epoch  = cp.get('chunk_within_epoch', 0)
        global_step         = cp.get('global_step', 0)
        chunk_start_step    = cp.get('chunk_start_step', 0)
        total_training_time = cp.get('total_training_time', 0.0)
        training_history    = cp.get('training_history', training_history)

        steps_in_chunk = global_step - chunk_start_step
        print(f'  -> epoch={current_epoch}  cwi={chunk_within_epoch}  '
              f'step={global_step:,}  '
              f'steps_done_in_chunk={steps_in_chunk:,}')

        if current_epoch > CONFIG['num_epochs']:
            print(f'\n✅ Training terminé.')
            return

    print('\n' + '='*80)
    print(f'TRAINING START — epochs {current_epoch}→{CONFIG["num_epochs"]}  '
          f'chunks/epoch={CONFIG["chunks_per_epoch"]}  sliding window')
    print(f'✅ Dataset VRAM + prefetch background')
    print('='*80)

    for epoch in range(current_epoch, CONFIG['num_epochs'] + 1):
        ep_start  = (epoch - 1) * CONFIG['chunks_per_epoch']
        ep_chunks = ALL_TRAIN_CHUNKS[ep_start:ep_start + CONFIG['chunks_per_epoch']]
        start_cwi = chunk_within_epoch if epoch == current_epoch else 0

        chunk_ids_str = ' '.join(f'chunk_{c["id"]:03d}' for c in ep_chunks)
        print(f'\nEPOCH {epoch}/{CONFIG["num_epochs"]} — {chunk_ids_str}')

        # Charger le premier chunk directement
        first_chunk    = ep_chunks[start_cwi]
        first_val_seed = CONFIG['shuffle_seed'] + (epoch - 1) * 1000 + start_cwi
        cds = VRAMChunkLoader(first_chunk, CONFIG['max_seq_len'],
                              tokenizer.pad_token_id, CONFIG['val_tokens'],
                              val_seed=first_val_seed)

        for cwi in range(start_cwi, CONFIG['chunks_per_epoch']):
            chunk_info = ep_chunks[cwi]

            is_resume = (epoch == current_epoch and cwi == chunk_within_epoch and cp is not None)
            if not is_resume:
                chunk_start_step = global_step

            # Déterminer le chunk suivant pour prefetch
            if cwi + 1 < CONFIG['chunks_per_epoch']:
                next_chunk_info = ep_chunks[cwi + 1]
            elif epoch + 1 <= CONFIG['num_epochs']:
                next_ep_start   = epoch * CONFIG['chunks_per_epoch']
                next_chunk_info = (ALL_TRAIN_CHUNKS[next_ep_start]
                                   if next_ep_start < len(ALL_TRAIN_CHUNKS) else None)
            else:
                next_chunk_info = None

            try:
                global_step, total_training_time, chunk_start_step = train_one_chunk(
                    model=model, cds=cds, chunk_id=chunk_info['id'],
                    optimizers=optimizers, scheduler=scheduler,
                    checkpoint_manager=ckpt_mgr, training_history=training_history,
                    global_step=global_step, total_training_time=total_training_time,
                    current_epoch=epoch, chunk_within_epoch=cwi,
                    chunk_start_step=chunk_start_step,
                    prefetcher=prefetcher, next_chunk_info=next_chunk_info,
                )
                cp = None

            except KeyboardInterrupt:
                print('\nCTRL+C — sauvegarde...')
                ckpt_mgr.save(model, optimizers, scheduler, metadata={
                    'current_epoch': epoch, 'chunk_within_epoch': cwi,
                    'global_step': global_step, 'chunk_start_step': chunk_start_step,
                    'total_training_time': total_training_time,
                    'training_history': training_history,
                })
                prefetcher.cancel()
                return

            except Exception:
                print(f'\nERREUR :\n{traceback.format_exc()}')
                ckpt_mgr.save(model, optimizers, scheduler, metadata={
                    'current_epoch': epoch, 'chunk_within_epoch': cwi,
                    'global_step': global_step, 'chunk_start_step': chunk_start_step,
                    'total_training_time': total_training_time,
                    'training_history': training_history,
                })
                prefetcher.cancel()
                raise

            # Save 25%
            next_cwi = cwi + 1
            save_ep  = epoch + 1 if next_cwi >= CONFIG['chunks_per_epoch'] else epoch
            save_cwi = 0         if next_cwi >= CONFIG['chunks_per_epoch'] else next_cwi
            pct      = int(next_cwi / CONFIG['chunks_per_epoch'] * 100)
            print(f'\n  [{pct}% epoch {epoch}] Save...')
            ckpt_mgr.save(model, optimizers, scheduler, metadata={
                'current_epoch': save_ep, 'chunk_within_epoch': save_cwi,
                'global_step': global_step, 'chunk_start_step': global_step,
                'total_training_time': total_training_time,
                'training_history': training_history,
            })

            # Swap chunk : unload courant, récupérer prefetché
            cds.unload()
            del cds

            if next_chunk_info is not None:
                print(f"  ⏳ Récupération prefetch chunk_{next_chunk_info['id']:03d}...")
                cds = prefetcher.get()
                if cds is None:
                    # Fallback si prefetch a échoué
                    next_val_seed = CONFIG['shuffle_seed'] + (epoch - 1) * 1000 + (cwi + 1)
                    cds = VRAMChunkLoader(next_chunk_info, CONFIG['max_seq_len'],
                                         tokenizer.pad_token_id, CONFIG['val_tokens'],
                                         val_seed=next_val_seed)

        ep_hist  = [c for c in training_history['chunks'] if c['current_epoch'] == epoch]
        avg_loss = sum(c['train_loss'] for c in ep_hist) / max(len(ep_hist), 1)
        print(f'\n{"="*80}')
        print(f'EPOCH {epoch} TERMINÉE  loss={avg_loss:.4f}  '
              f'step={global_step:,}  time={total_training_time/3600:.2f}h')
        print(f'{"="*80}')
        training_history['epochs'].append({
            'epoch': epoch, 'train_loss': avg_loss,
            'global_step': global_step, 'time_sec': total_training_time,
        })
        chunk_within_epoch = 0

    print(f'\n{"="*80}\nTRAINING TERMINÉ\n{"="*80}')
    print(f'  Steps : {global_step:,}  Temps : {total_training_time/3600:.2f}h')
    if training_history.get('validations'):
        last = training_history['validations'][-1]
        print(f'  Val PPL : {last["val_ppl"]:.2f}  Val Loss : {last["val_loss"]:.4f}')

    ckpt_mgr.save(model, optimizers, scheduler, metadata={
        'current_epoch': CONFIG['num_epochs'] + 1, 'chunk_within_epoch': 0,
        'global_step': global_step, 'chunk_start_step': global_step,
        'total_training_time': total_training_time, 'training_history': training_history,
    })
    history_path = CONFIG['checkpoint_file'].replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2, default=str)
    print(f'  History : {history_path}')
    print('DONE')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrompu')
    except Exception:
        print(traceback.format_exc())
    finally:
        print('\nBye')

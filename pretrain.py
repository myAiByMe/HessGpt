#!/usr/bin/env python3
"""
HessGPT Pre-Training v5 — LLaMA-3 Tokenizer
"""

import os

os.environ["TORCHINDUCTOR_CACHE_DIR"]    = "./CompileCache"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.makedirs("./CompileCache", exist_ok=True)

import torch
from torch.utils.data import DataLoader, Dataset
import sys
import time
import math
import json
import gc
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
print("HessGPT v5 — LLaMA-3 | RMSNorm | Flash | QK-Norm | WSD | Muon+MARS-M")
print("=" * 80)
print(f"  CompileCache → ./CompileCache")

CONFIG = {
    'vocab_size':            None,
    'embed_dim':             1280,
    'num_heads':             20,
    'num_layers':            24,
    'max_seq_len':           512,
    'dropout':               0.0,
    'use_rope':              True,
    'use_yarn':              False,
    'yarn_scale':            4.0,
    'yarn_original_max_len': 512,
    'use_swiglu':            True,
    'n_kv_heads':            5,
    'use_qk_norm':           True,
    'soft_cap':              30.0,
    'use_flash_attn':        True,
    'batch_size':            80,
    'gradient_accumulation': 8,
    'max_grad_norm':         1.0,
    'learning_rate':         4e-4,
    'weight_decay':          0.1,
    'adam_beta1':            0.9,
    'adam_beta2':            0.95,
    'adam_eps':              1e-8,
    'num_epochs':       5,
    'chunks_per_epoch': 3,
    'data_dir':   './data/ultra_filtered',
    'val_tokens': 15_000_000,
    'warmup_ratio': 0.03,
    'decay_ratio':  0.15,
    'min_lr_ratio': 0.1,
    'validate_every_steps': 500,
    'val_batches':          50,
    'shuffle_seed': 42,
    'save_every_steps': 2000,
    'checkpoint_file': './Model/HessGpt_pretrain.pt',
    'use_compile':  True,
    'compile_mode': 'default',
    'num_workers':  8,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\nCONFIG :")
print(f"  embed={CONFIG['embed_dim']}  layers={CONFIG['num_layers']}  "
      f"heads={CONFIG['num_heads']}  kv={CONFIG['n_kv_heads']}")
print(f"  epochs={CONFIG['num_epochs']}  chunks/epoch={CONFIG['chunks_per_epoch']}")
print(f"  batch={CONFIG['batch_size']}  accum={CONFIG['gradient_accumulation']}  "
      f"device={device}")
if CONFIG['soft_cap'] is not None:
    print(f"  soft_cap={CONFIG['soft_cap']} (attention soft cap actif — attention.py v8)")
if device == 'cuda':
    print(f"  GPU={torch.cuda.get_device_name(0)}  "
          f"VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")

def scan_available_chunks(data_dir):
    available = []
    if not os.path.exists(data_dir):
        return available
    for entry in sorted(os.listdir(data_dir)):
        if not entry.startswith('chunk'):
            continue
        chunk_dir = os.path.join(data_dir, entry)
        if not os.path.isdir(chunk_dir):
            continue
        stats_file = os.path.join(chunk_dir, 'stats.json')
        if not os.path.exists(stats_file):
            continue
        try:
            with open(stats_file, 'r') as f:
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
    print(f"  WARN: seulement {epochs_possible} epochs possibles avec {n_chunks} chunks")
    CONFIG['num_epochs'] = epochs_possible

TOTAL_CHUNKS_USED = CONFIG['num_epochs'] * CONFIG['chunks_per_epoch']
ALL_TRAIN_CHUNKS  = ALL_CHUNKS[:TOTAL_CHUNKS_USED]

print(f"\nPLAN CHUNKS (sliding window, jamais recyclés) :")
for ep in range(CONFIG['num_epochs']):
    s   = ep * CONFIG['chunks_per_epoch']
    ids = [f"chunk_{c['id']:03d}" for c in ALL_TRAIN_CHUNKS[s:s + CONFIG['chunks_per_epoch']]]
    print(f"  Epoch {ep+1:2d} : {' '.join(ids)}")

def steps_for_chunk(stats):
    samples = stats['total_tokens'] // (CONFIG['max_seq_len'] + 1)
    batches = math.ceil(samples / CONFIG['batch_size'])
    return max(math.ceil(batches / CONFIG['gradient_accumulation']), 1)

TOTAL_STEPS     = sum(steps_for_chunk(c['stats']) for c in ALL_TRAIN_CHUNKS)
STEPS_PER_EPOCH = TOTAL_STEPS // CONFIG['num_epochs']

print(f"\nPLAN STEPS :")
print(f"  steps/epoch~={STEPS_PER_EPOCH:,}  total={TOTAL_STEPS:,}")
print(f"  safety save : tous les {CONFIG['save_every_steps']} steps")

print(f"\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
CONFIG['vocab_size'] = len(tokenizer)
print(f"  vocab={len(tokenizer)}")
for tok in SPECIAL_TOKENS:
    print(f"  {tok} → {tokenizer.convert_tokens_to_ids(tok)}")

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
                if pg.get('is_muon', False):
                    pg['lr'] = lr * 5.0
                else:
                    pg['lr'] = lr
        return lr

    def get_last_lr(self):
        return [self.get_lr()]

    def state_dict(self):
        return {'current_step': self.current_step}

    def load_state_dict(self, sd):
        self.current_step = sd['current_step']


class ChunkSubset(Dataset):
    def __init__(self, tokens, seq_len, pad_token_id):
        self.seq_len      = seq_len
        self.pad_token_id = pad_token_id
        self.num_samples  = len(tokens) // (seq_len + 1)
        self.tokens       = tokens[:self.num_samples * (seq_len + 1)].share_memory_()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = self.tokens[start:start + self.seq_len + 1]
        return chunk[:-1].clone(), chunk[1:].clone()


class SeededSampler(torch.utils.data.Sampler):
    def __init__(self, n: int, seed: int, skip_samples: int = 0):
        self.n            = n
        self.seed         = seed
        self.skip_samples = min(skip_samples, n)
        rng               = np.random.default_rng(seed)
        indices           = rng.permutation(n)
        self._indices     = indices[self.skip_samples:]
        print(f"  SeededSampler : n={n:,}  seed={seed}  "
              f"skip={self.skip_samples:,}  restant={len(self._indices):,}")

    def __iter__(self):
        return iter(self._indices.tolist())

    def __len__(self):
        return len(self._indices)


class LazyChunkDataset:
    def __init__(self, chunk_info, seq_len, pad_token_id,
                 val_tokens=15_000_000, val_seed=0):
        self.seq_len      = seq_len
        self.pad_token_id = pad_token_id
        self.val_tokens   = val_tokens
        self.val_seed     = val_seed
        self._load(chunk_info)

    def _load(self, chunk_info):
        print(f"  Loading chunk_{chunk_info['id']:03d}...")
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
        seq_len_1  = self.seq_len + 1
        n_seqs     = total // seq_len_1
        rng        = np.random.default_rng(self.val_seed)
        idx        = rng.permutation(n_seqs)
        all_tokens = all_tokens[:n_seqs * seq_len_1].reshape(n_seqs, seq_len_1)[idx].reshape(-1)
        total      = len(all_tokens)
        val_size   = min(self.val_tokens, int(total * 0.05))
        train_size = total - val_size
        self._train_toks = all_tokens[:train_size]
        self._val_toks   = all_tokens[train_size:]
        ram_gb = all_tokens.element_size() * total / 1e9
        print(f"  chunk_{chunk_info['id']:03d} : {total/1e6:.0f}M tokens  "
              f"train={train_size/1e6:.0f}M  val={val_size/1e6:.0f}M  "
              f"RAM={ram_gb:.1f}GB  ({time.time()-t0:.1f}s)  val_seed={self.val_seed}")

    def get_train_dataset(self):
        return ChunkSubset(self._train_toks, self.seq_len, self.pad_token_id)

    def get_val_dataset(self):
        return ChunkSubset(self._val_toks, self.seq_len, self.pad_token_id)

    def unload(self):
        del self._train_toks, self._val_toks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  RAM chunk libérée")


# ============================================================
# CHECKPOINT MANAGER — write-ahead log pattern
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

        # .pt — poids + états optimizer uniquement, zéro métadonnées
        cp = {
            **opt_state,
            'model_state_dict':     m.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }

        json_path = self.path.replace('.pt', '_info.json')
        new_path  = json_path + '.new'  # write-ahead : écrit EN PREMIER

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

        # Ordre atomique :
        # 1. _info.json.new  (métadonnées, rapide)
        # 2. .pt             (poids, lent)
        # 3. supp _info.json + rename .new → _info.json (atomique)
        # Si crash entre 1 et 3 → .new existe au restart → recovery auto
        with open(new_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)

        tmp = self.path + '.tmp'
        torch.save(cp, tmp)
        os.replace(tmp, self.path)

        if os.path.exists(json_path):
            os.remove(json_path)
        os.replace(new_path, json_path)  # atomique

        print(f"  💾 SAVE → epoch={metadata['current_epoch']}  "
              f"cwi={metadata['chunk_within_epoch']}/{CONFIG['chunks_per_epoch']}  "
              f"step={metadata['global_step']:,}  "
              f"chunk_start={metadata.get('chunk_start_step', 0):,}  "
              f"[{self.path}]")

    def load(self):
        if not os.path.exists(self.path):
            return None
        print(f"\nCheckpoint trouvé : {self.path}")
        cp = torch.load(self.path, map_location='cpu', weights_only=False)

        json_path = self.path.replace('.pt', '_info.json')
        new_path  = json_path + '.new'

        # ✅ write-ahead recovery :
        # Si .new existe → crash entre le save .pt et le rename final
        # → .new est plus à jour → on le substitue à .json
        if os.path.exists(new_path):
            print(f"  ⚠️  _info.json.new détecté — crash lors du save précédent")
            print(f"  ✅ Recovery : suppression .json, rename .new → .json")
            if os.path.exists(json_path):
                os.remove(json_path)
            os.replace(new_path, json_path)

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                info = json.load(f)
            cp['global_step']         = info.get('global_step', 0)
            cp['chunk_start_step']    = info.get('chunk_start_step', 0)
            cp['current_epoch']       = info.get('current_epoch', 1)
            cp['chunk_within_epoch']  = info.get('chunk_within_epoch', 0)
            cp['total_training_time'] = info.get('total_training_time', 0.0)
            cp['training_history']    = info.get('training_history',
                                         {'chunks': [], 'validations': [], 'epochs': []})
            cp['last_save']           = info.get('last_save', '?')
            print(f"  [JSON] epoch={cp['current_epoch']}  "
                  f"cwi={cp['chunk_within_epoch']}  "
                  f"step={cp['global_step']:,}  "
                  f"chunk_start={cp['chunk_start_step']:,}  "
                  f"saved={cp['last_save']}")
        else:
            print(f"  ⚠️  Aucun JSON — métadonnées manquantes, repart step=0")
            cp['global_step']         = 0
            cp['chunk_start_step']    = 0
            cp['current_epoch']       = 1
            cp['chunk_within_epoch']  = 0
            cp['total_training_time'] = 0.0
            cp['training_history']    = {'chunks': [], 'validations': [], 'epochs': []}
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
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device, dtype=adt, enabled=ae):
                _, loss, _ = model(x, targets=y, pad_token_id=tokenizer.pad_token_id)
            total_loss += loss.item()
            n += 1
    finally:
        model.train()
    avg = total_loss / max(n, 1)
    return math.exp(min(avg, 10)), avg


# ============================================================
# MARS-M + MUON
# ============================================================
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
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
            lr, momentum, nesterov = group['lr'], group['momentum'], group['nesterov']
            ns_steps, wd           = group['ns_steps'], group['weight_decay']
            use_mars               = group.get('use_mars', True)
            mars_gamma             = group.get('mars_gamma', 0.025)

            for p in group['params']:
                if p.grad is None:
                    continue
                g     = p.grad
                if g.ndim < 2:
                    continue
                state = self.state[p]

                if use_mars:
                    if 'prev_grad' not in state:
                        state['prev_grad'] = torch.zeros_like(g)
                    prev_g    = state['prev_grad']
                    norm_g    = g.norm() + 1e-8
                    norm_prev = prev_g.norm() + 1e-8
                    c_t = (mars_gamma / (1.0 - mars_gamma)) * (norm_g / norm_prev)
                    c_t = torch.clamp(c_t, max=1.0)
                    g   = g + c_t * (g - prev_g)
                    state['prev_grad'].copy_(p.grad)

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g + momentum * buf
                else:
                    g = buf

                g     = zeropower_via_newtonschulz5(g, steps=ns_steps)
                scale = max(g.size(0), g.size(1)) ** 0.5
                g     = g * scale

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)


def configure_optimizers(model, lr, weight_decay, betas, eps):
    MUON_EXCLUDE = {
        'token_embeddings.weight',
        'output_head.weight',
        'position_embeddings.weight',
    }
    muon_params, adamw_decay, adamw_nodecay = [], [], []

    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if pn in MUON_EXCLUDE:
            (adamw_decay if p.dim() >= 2 else adamw_nodecay).append(p)
            continue
        if p.dim() >= 2 and pn.startswith('blocks.'):
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
        ns_steps=3, weight_decay=0.0, use_mars=True, mars_gamma=0.025,
    )
    muon_opt.param_groups[0]['is_muon'] = True

    adamw_opt = torch.optim.AdamW(
        [
            {'params': adamw_decay,   'weight_decay': weight_decay, 'is_muon': False},
            {'params': adamw_nodecay, 'weight_decay': 0.0,          'is_muon': False},
        ],
        lr=lr, betas=betas, eps=eps, fused=(device == 'cuda'),
    )

    print(f"\nOptimizer Muon+MARS-M + AdamW :")
    print(f"  Muon+MARS-M : {len(muon_params):3d} tenseurs  lr={lr_muon:.2e}  mars_gamma=0.025")
    print(f"  AdamW       : {len(adamw_decay):3d} tenseurs decay + "
          f"{len(adamw_nodecay):3d} no-decay  lr={lr:.2e}")
    return muon_opt, adamw_opt


# ============================================================
# TRAIN ONE CHUNK
# ============================================================
def train_one_chunk(
    model, chunk_info, optimizers, scheduler,
    checkpoint_manager, training_history,
    global_step, total_training_time,
    current_epoch, chunk_within_epoch, chunk_start_step,
):
    muon_opt, adamw_opt = optimizers
    label        = (f"Epoch {current_epoch}/{CONFIG['num_epochs']} | "
                    f"cwi {chunk_within_epoch+1}/{CONFIG['chunks_per_epoch']} "
                    f"(chunk_{chunk_info['id']:03d})")
    steps_done   = global_step - chunk_start_step
    batches_done = steps_done * CONFIG['gradient_accumulation']
    shuffle_seed = CONFIG['shuffle_seed'] + (current_epoch - 1) * 1000 + chunk_within_epoch
    val_seed     = shuffle_seed

    print(f"\n{'='*80}")
    print(f"  {label}  LR_adamw={scheduler.get_last_lr()[0]:.2e}"
          f"  LR_muon={scheduler.get_last_lr()[0]*5:.2e}  shuffle_seed={shuffle_seed}")
    if batches_done > 0:
        print(f"  ⏩ Reprise : global_step={global_step:,}  "
              f"chunk_start_step={chunk_start_step:,}  "
              f"steps_done={steps_done:,}  batches_done={batches_done:,}")
    print(f"{'='*80}")

    try:
        cds = LazyChunkDataset(
            chunk_info, CONFIG['max_seq_len'],
            tokenizer.pad_token_id, CONFIG['val_tokens'], val_seed=val_seed,
        )
    except Exception as e:
        print(f"  ERREUR chargement chunk : {e}")
        return global_step, total_training_time, chunk_start_step

    train_ds   = cds.get_train_dataset()
    total_seqs = len(train_ds)

    if batches_done >= math.ceil(total_seqs / CONFIG['batch_size']):
        print(f"  ✅ Chunk déjà entièrement traité, skip.")
        cds.unload()
        gc.collect()
        return global_step, total_training_time, chunk_start_step

    sampler = SeededSampler(
        n=total_seqs, seed=shuffle_seed,
        skip_samples=batches_done * CONFIG['batch_size'],
    )
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'], sampler=sampler,
        num_workers=CONFIG['num_workers'], pin_memory=True,
        persistent_workers=True, prefetch_factor=2, drop_last=True,
    )
    val_loader = DataLoader(
        cds.get_val_dataset(), batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True,
    )

    num_batches   = len(train_loader)
    total_batches = total_seqs // CONFIG['batch_size']
    print(f"  train={total_batches:,} batches total | restant={num_batches:,} | val={len(val_loader):,}")

    model.train()
    chunk_loss, valid_batches     = 0.0, 0
    accumulated_steps             = 0
    running_loss, running_batches = 0.0, 0
    t_start = time.time()
    ae  = (device == 'cuda')
    adt = torch.bfloat16 if ae else torch.float32

    pbar = tqdm(train_loader, desc=label, leave=True,
                initial=total_batches - num_batches, total=total_batches)

    for batch_idx, (x, y) in enumerate(pbar):
        try:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
                        'chunk_within_epoch': chunk_within_epoch,
                        'chunk_id': chunk_info['id'],
                        'val_loss': val_loss, 'val_ppl': val_ppl,
                        'train_loss': avg, 'lr': scheduler.get_last_lr()[0],
                    })

                if global_step % CONFIG['save_every_steps'] == 0:
                    checkpoint_manager.save(model, optimizers, scheduler, metadata={
                        'current_epoch':       current_epoch,
                        'chunk_within_epoch':  chunk_within_epoch,
                        'global_step':         global_step,
                        'chunk_start_step':    chunk_start_step,
                        'total_training_time': total_training_time + (time.time() - t_start),
                        'training_history':    training_history,
                    })

            raw = loss.item() * CONFIG['gradient_accumulation']
            chunk_loss      += raw
            running_loss    += raw
            valid_batches   += 1
            running_batches += 1

            if batch_idx % 20 == 0:
                avg = running_loss / max(running_batches, 1)
                pbar.set_postfix(
                    loss=f'{raw:.4f}', avg=f'{avg:.4f}',
                    ppl=f'{math.exp(min(avg,10)):.1f}',
                    lr=f'{scheduler.get_last_lr()[0]:.2e}',
                    step=f'{global_step:,}',
                )

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
    train_loader._iterator = None
    del train_loader

    elapsed = time.time() - t_start
    total_training_time += elapsed
    avg_loss = chunk_loss / max(valid_batches, 1)
    print(f"\n  chunk_{chunk_info['id']:03d} terminé | loss={avg_loss:.4f} | {elapsed/60:.1f}min")

    training_history['chunks'].append({
        'current_epoch': current_epoch, 'chunk_within_epoch': chunk_within_epoch,
        'chunk_id': chunk_info['id'], 'train_loss': avg_loss,
        'time_sec': elapsed, 'batches': valid_batches, 'global_step': global_step,
    })

    val_loader._iterator = None
    del val_loader
    cds.unload()
    del cds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return global_step, total_training_time, chunk_start_step


# ============================================================
# MAIN
# ============================================================
def main():
    from HessGpt import HessGPT

    print('\n' + '='*80 + '\nCREATION MODELE\n' + '='*80)

    ckpt_mgr = CheckpointManager(CONFIG['checkpoint_file'])

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
        print('torch.compile...')
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.suppress_errors  = True
        cache_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk('./CompileCache') for f in files
        ) if os.path.exists('./CompileCache') else 0
        cache_empty = cache_size < 1024
        print(f"  Cache : ./CompileCache  "
              f"({'vide — première compile' if cache_empty else f'{cache_size/1e6:.0f}MB — cache hit probable'})")
        try:
            model = torch.compile(model, mode=CONFIG['compile_mode'])
            print('  OK')
        except Exception as e:
            print(f'  FAIL (on continue sans) : {e}')

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
        'config': CONFIG, 'total_params': total_params,
        'total_steps': TOTAL_STEPS, 'chunks': [], 'validations': [],
        'epochs': [], 'start_time': datetime.now().isoformat(),
    }

    global_step, current_epoch     = 0, 1
    chunk_within_epoch              = 0
    total_training_time             = 0.0
    chunk_start_step                = 0

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

        print(f'  -> epoch={current_epoch}  cwi={chunk_within_epoch}  '
              f'step={global_step:,}  chunk_start_step={chunk_start_step:,}')

        steps_in_chunk   = global_step - chunk_start_step
        batches_in_chunk = steps_in_chunk * CONFIG['gradient_accumulation']
        print(f'  -> steps déjà faits dans ce chunk = {steps_in_chunk:,}  '
              f'(≈ {batches_in_chunk:,} batches à skipper)')

        if current_epoch > CONFIG['num_epochs']:
            print(f'\n✅ Training déjà terminé ({CONFIG["num_epochs"]} epochs).')
            return

    print('\n' + '='*80)
    print(f'TRAINING START')
    print(f'  epochs {current_epoch} → {CONFIG["num_epochs"]}  |  '
          f'chunks/epoch={CONFIG["chunks_per_epoch"]}  |  '
          f'1 chunk en RAM à la fois  |  sliding window')
    print('='*80)

    for epoch in range(current_epoch, CONFIG['num_epochs'] + 1):
        ep_start  = (epoch - 1) * CONFIG['chunks_per_epoch']
        ep_chunks = ALL_TRAIN_CHUNKS[ep_start:ep_start + CONFIG['chunks_per_epoch']]
        start_cwi = chunk_within_epoch if epoch == current_epoch else 0

        chunk_ids_str = ' '.join(f'chunk_{c["id"]:03d}' for c in ep_chunks)
        print(f'\nEPOCH {epoch}/{CONFIG["num_epochs"]} — chunks : {chunk_ids_str}')
        if start_cwi > 0:
            print(f'  (reprise à cwi={start_cwi}  chunk_start_step={chunk_start_step:,})')

        for cwi in range(start_cwi, CONFIG['chunks_per_epoch']):
            chunk_info = ep_chunks[cwi]

            is_resume_chunk = (
                epoch == current_epoch and cwi == chunk_within_epoch and cp is not None
            )
            if not is_resume_chunk:
                chunk_start_step = global_step

            try:
                global_step, total_training_time, chunk_start_step = train_one_chunk(
                    model=model, chunk_info=chunk_info, optimizers=optimizers,
                    scheduler=scheduler, checkpoint_manager=ckpt_mgr,
                    training_history=training_history, global_step=global_step,
                    total_training_time=total_training_time, current_epoch=epoch,
                    chunk_within_epoch=cwi, chunk_start_step=chunk_start_step,
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
                return

            except Exception:
                print(f'\nERREUR chunk_{chunk_info["id"]:03d} :\n{traceback.format_exc()}')
                ckpt_mgr.save(model, optimizers, scheduler, metadata={
                    'current_epoch': epoch, 'chunk_within_epoch': cwi,
                    'global_step': global_step, 'chunk_start_step': chunk_start_step,
                    'total_training_time': total_training_time,
                    'training_history': training_history,
                })
                raise

            # ✅ save 25% supprimé — redondant avec save_every_steps

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
        'total_training_time': total_training_time,
        'training_history': training_history,
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

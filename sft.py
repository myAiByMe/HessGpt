#!/usr/bin/env python3
"""
HessGPT — SFT avec LoRA  (2-Stage Training)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATÉGIE 2 STAGES (plan Gemini v2) :

  STAGE 1 — Cold Start  (~17k samples, 1 epoch, LR=2e-4)
    Bespoke-Stratos-17k   100%   "branche" le circuit <think> avant
                                  d'injecter la connaissance générale

  STAGE 2 — Full Mix  (~275k samples, 3 epochs, LR=5e-5)
    PiKa-SFT               20%  ~55k   GPT-4o expert, haut niveau
    Smol-Magpie-Ultra       35%  ~96k   Llama-405B, 0 blocs de code
    OpenThoughts3-Logic     25%  ~69k   Science/Puzzles/Logic (<think>)
    AM-DeepSeek-Math         5%  ~14k   math éducatif distillé R1
    Bespoke-Stratos-17k      5%  ~14k   renforcement traces <think>
    Smol-Constraints         5%  ~14k   suivi strict instructions
    Aegis-AI-Safety          5%  ~14k   sécurité / refus propres

FORMAT LLaMA-3 natif :
  <|begin_of_text|>
  <|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>
  <|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>\n{response}<|eot_id|>

MASKING — assistant-only loss :
  Machine à états : PROMPT → IN_HEADER → IN_ASST_BODY → PROMPT → ...
  Loss uniquement sur les tokens assistant (incl. blocs <think>...</think>)
  <|eot_id|> inclus dans la loss (modèle apprend à s'arrêter).

LORA :
  Rank 64, Alpha 128, tous modules Q/K/V/O + MLP
  Seuls les poids LoRA sauvés dans le checkpoint

SAVES :
  Tous les save_every_steps, fin de chaque epoch, CTRL+C / exception

REPRISE :
  Relancer la même commande — reprend là où ça s'est arrêté.
  Stage 1 complet → passe automatiquement au Stage 2.

USAGE :
  python sft_hessgpt_lora.py
  python sft_hessgpt_lora.py --pretrain-checkpoint ./checkpoints/HessGpt_pretrain.pt
  python sft_hessgpt_lora.py --dry-run
  python sft_hessgpt_lora.py --stage 2   # saute Stage 1 si déjà fait
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys, os, time, math, json, gc, re, argparse, traceback, warnings, random
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from datetime import datetime
from typing import Dict, List, Optional

warnings.filterwarnings('ignore')
sys.path.append('./Core/Model')

# ============================================================
# TOKENS CUSTOM  (les tokens de chat LLaMA-3 sont natifs 128k)
# ============================================================
CUSTOM_TOKEN_STRINGS = ['<think>', '</think>', '<code>']
SPECIAL_TOKENS: Dict[str, int] = {}   # rempli après tokenizer load

# ============================================================
# ARGS
# ============================================================
parser = argparse.ArgumentParser(description='HessGPT SFT 2-Stage LoRA')
parser.add_argument('--pretrain-checkpoint', type=str,
                    default='./checkpoints/HessGpt_pretrain.pt')
parser.add_argument('--output-checkpoint', type=str,
                    default='./checkpoints/HessGpt_sft_lora.pt')
parser.add_argument('--dry-run', action='store_true',
                    help='Vérifie datasets + masking sans entraîner')
parser.add_argument('--num-samples', type=int, default=None,
                    help='Limite samples pour debug rapide')
parser.add_argument('--stage', type=int, default=None, choices=[1, 2],
                    help='Force un stage (1 ou 2) — ignore le checkpoint stage')
args = parser.parse_args()

print('=' * 80)
print('HessGPT — SFT LoRA 2-Stage  |  LLaMA-3 128k  |  ~275k  |  0%% code')
print('=' * 80)

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # ── Modèle (cohérent avec pretrain sauf max_seq_len + YaRN)
    'vocab_size':            None,   # défini après tokenizer
    'embed_dim':             1280,
    'num_heads':             20,
    'num_layers':            24,
    'max_seq_len':           2048,
    'dropout':               0.05,
    'use_rope':              True,
    'use_yarn':              True,
    'yarn_scale':            2.0,    # 1024 → 2048
    'yarn_original_max_len': 1024,
    'use_swiglu':            True,
    'n_kv_heads':            5,
    'use_qk_norm':           True,
    'soft_cap':              30.0,
    'use_flash_attn':        True,

    # ── LoRA
    'lora_r':       64,
    'lora_alpha':   128,
    'lora_dropout': 0.05,
    'lora_target_modules': [
        'q_proj', 'k_proj', 'v_proj', 'out_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ],

    # ── Stage 1 — Cold Start (Bespoke-Stratos uniquement)
    'stage1': {
        'epochs':                1,
        'batch_size':            8,
        'gradient_accumulation': 4,
        'learning_rate':         1e-4,   # Gemini : 2e-4 trop agressif → effondrement entropie
        'warmup_ratio':          0.10,   # warmup plus long sur petit dataset
        'decay_ratio':           0.20,
        'min_lr_ratio':          0.1,
        'max_grad_norm':         1.0,
        'validate_every_steps':  100,
        'val_batches':           30,
        'save_every_steps':      200,
    },

    # ── Stage 2 — Full Mix
    'stage2': {
        'epochs':                3,
        'batch_size':            8,
        'gradient_accumulation': 4,
        'learning_rate':         5e-5,   # conservateur pour ne pas écraser Stage 1
        'warmup_ratio':          0.03,
        'decay_ratio':           0.15,
        'min_lr_ratio':          0.1,
        'max_grad_norm':         1.0,
        'validate_every_steps':  300,
        'val_batches':           50,
        'save_every_steps':      500,
    },

    # ── Data
    'val_split':    0.03,
    'anneal_factor': 5.0,    # surpondération RefGPT-Fact pendant phase decay WSD

    # ── Tailles cibles Stage 2 (total ~276k)
    # PiKa 25%, OpenThoughts3 35% (HES Top20%), Magpie 30%, AM-Math 5%, Aegis 5%
    'dataset_sizes': {
        'pika_sft':          76_000,   # ~27%
        'openthoughts3':     96_000,   # 35% + HES Top20%
        'smol_magpie_ultra': 90_000,   # ~32%
        'aegis_safety':      14_000,   # 5%
        'refgpt_fact':       10_000,   # ~4% — annealing haute qualité factuel
        'longalign':         14_000,   # ~5% — long-context YaRN (conditionnel)
        # am_deepseek_math : retiré — objectif assistant factuel, pas math
        # bespoke_stratos  : retiré — Stage 1 suffit pour le formatage <think>
        # smol_constraints : retiré
    },
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'\nCONFIG :')
print(f'  embed={CONFIG["embed_dim"]}  layers={CONFIG["num_layers"]}  '
      f'heads={CONFIG["num_heads"]}  kv={CONFIG["n_kv_heads"]}')
print(f'  max_seq_len={CONFIG["max_seq_len"]}  YaRN x{CONFIG["yarn_scale"]}')
print(f'  LoRA r={CONFIG["lora_r"]}  alpha={CONFIG["lora_alpha"]}')
print(f'  Stage1 LR={CONFIG["stage1"]["learning_rate"]:.0e}  '
      f'Stage2 LR={CONFIG["stage2"]["learning_rate"]:.0e}')
print(f'  device={device}')
if device == 'cuda':
    print(f'  GPU={torch.cuda.get_device_name(0)}  '
          f'VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB')

# ── Filtre No-Code modéré — blocs de code seulement
_CODE_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in [
    r'```\s*(?:python|javascript|js|typescript|ts|java|c\+\+|cpp|c#|'
    r'csharp|ruby|go|rust|php|swift|kotlin|scala|r|bash|sh|sql|html|css)',
    r'<code>[\s\S]{50,}</code>',
]]

def has_code_blocks(text: str) -> bool:
    return any(r.search(text) for r in _CODE_RE)

def messages_have_code(messages: List[dict]) -> bool:
    return any(has_code_blocks(m.get('content', '')) for m in messages)

# ============================================================
# TOKENIZER — global, chargé une seule fois
# ============================================================
print(f'\nLoading LLaMA-3 tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('unsloth/Meta-Llama-3-8B')
tokenizer.add_special_tokens({'additional_special_tokens': CUSTOM_TOKEN_STRINGS})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
CONFIG['vocab_size'] = len(tokenizer)

for tok_str in [
    '<|begin_of_text|>', '<|end_of_text|>',
    '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>',
] + CUSTOM_TOKEN_STRINGS:
    tid = tokenizer.convert_tokens_to_ids(tok_str)
    if tid == tokenizer.unk_token_id:
        print(f'  WARN: {tok_str} inconnu dans le vocab !')
    SPECIAL_TOKENS[tok_str] = tid

print(f'  vocab={len(tokenizer)}  pad_id={tokenizer.pad_token_id}')
for k, v in SPECIAL_TOKENS.items():
    print(f'  {k:<32s} → {v}')

# ============================================================
# FORMAT LLAMA-3
# ============================================================
def build_conversation(system: str, turns: List[tuple]) -> str:
    """turns = [(user_msg, asst_msg), ...]"""
    bos = '<|begin_of_text|>'
    sh  = '<|start_header_id|>'
    eh  = '<|end_header_id|>'
    eot = '<|eot_id|>'
    out = bos + f'{sh}system{eh}\n{system}{eot}'
    for user_msg, asst_msg in turns:
        out += f'{sh}user{eh}\n{user_msg}{eot}'
        out += f'{sh}assistant{eh}\n{asst_msg}{eot}'
    return out

def format_from_messages(messages: List[dict]) -> str:
    bos = '<|begin_of_text|>'
    sh  = '<|start_header_id|>'
    eh  = '<|end_header_id|>'
    eot = '<|eot_id|>'
    out = bos
    for msg in messages:
        role    = msg.get('role', '')
        content = msg.get('content', '')
        if role in ('system', 'user', 'assistant'):
            out += f'{sh}{role}{eh}\n{content}{eot}'
    return out

def sample_to_text(sample: dict) -> str:
    """Convertit n'importe quel format de sample en texte LLaMA-3."""
    if 'messages' in sample and sample['messages']:
        return format_from_messages(sample['messages'])
    if 'instruction' in sample and 'output' in sample:
        system = sample.get('system', 'You are a helpful assistant.')
        return build_conversation(system, [(sample['instruction'], sample['output'])])
    system = sample.get('system', 'You are a helpful assistant.')
    user   = sample.get('user',   sample.get('prompt',   ''))
    asst   = sample.get('assistant', sample.get('response', ''))
    return build_conversation(system, [(user, asst)])

def _normalize_messages(msgs: list) -> List[dict]:
    """Normalise depuis format conversations HF ou OpenAI."""
    normalized = []
    for m in msgs:
        from_field = m.get('from', m.get('role', ''))
        role = ('assistant' if from_field in ('gpt', 'assistant', 'model') else
                'user'      if from_field in ('human', 'user') else
                'system'    if from_field == 'system' else None)
        if role is None:
            continue
        content = m.get('value', m.get('content', ''))
        if content:
            normalized.append({'role': role, 'content': content})
    return normalized

def _inject_think(messages: List[dict], thinking: str) -> List[dict]:
    """Injecte <think>...</think> dans le dernier message assistant si absent."""
    if not thinking or not thinking.strip():
        return messages
    result = list(messages)
    for j in range(len(result) - 1, -1, -1):
        if result[j].get('role') == 'assistant':
            content = result[j].get('content', '')
            if '<think>' not in content:
                clean = re.sub(r'<think>.*?</think>', '', content,
                               flags=re.DOTALL).strip()
                result[j] = {
                    'role':    'assistant',
                    'content': f'<think>{thinking.strip()}</think>\n{clean}',
                }
            break
    return result

# ============================================================
# DATASET SFT — ASSISTANT-ONLY LOSS MASKING
# ============================================================
class SFTDataset(Dataset):
    """
    Machine à états pour le masking multi-turn :
      PROMPT       : labels = -100
      IN_HEADER    : accumule tokens du header
      IN_ASST_BODY : labels = target_ids (entraîné)

    Transitions :
      PROMPT       + <|start_header_id|> → IN_HEADER
      IN_HEADER    + <|end_header_id|>   → IN_ASST_BODY si header='assistant'
                                         → PROMPT sinon
      IN_ASST_BODY + <|eot_id|>          → PROMPT
                                           (<|eot_id|> inclus dans la loss)

    Le flag annealing=True est posé sur les samples haute qualité (RefGPT-Fact).
    get_annealing_weights() retourne les poids pour WeightedRandomSampler.
    """
    def __init__(self, samples: list, max_seq_len: int):
        self.samples     = samples
        self.max_seq_len = max_seq_len
        self._sh  = SPECIAL_TOKENS['<|start_header_id|>']
        self._eh  = SPECIAL_TOKENS['<|end_header_id|>']
        self._eot = SPECIAL_TOKENS['<|eot_id|>']

    def __len__(self):
        return len(self.samples)

    def get_annealing_weights(self, anneal_factor: float = 5.0):
        """
        Retourne un tenseur de poids pour WeightedRandomSampler.
        Les samples marqués annealing=True reçoivent anneal_factor × plus de poids.
        Utilisé pendant la phase decay du WSD pour le Data Annealing.
        """
        weights = []
        for s in self.samples:
            weights.append(anneal_factor if s.get('annealing') else 1.0)
        return torch.tensor(weights, dtype=torch.float)

    def __getitem__(self, idx):
        text   = sample_to_text(self.samples[idx])
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len - 1] + [SPECIAL_TOKENS['<|eot_id|>']]

        if len(tokens) < 2:
            dummy = torch.zeros(1, dtype=torch.long)
            return dummy, torch.full((1,), -100, dtype=torch.long)

        input_ids  = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:],  dtype=torch.long)
        labels     = torch.full_like(target_ids, -100)   # tout masqué par défaut

        state              = 'prompt'
        pending_header_ids = []

        for i, tok in enumerate(input_ids.tolist()):
            if state == 'prompt':
                if tok == self._sh:
                    state              = 'in_header'
                    pending_header_ids = []

            elif state == 'in_header':
                if tok == self._eh:
                    header_text = tokenizer.decode(pending_header_ids).strip()
                    state       = 'in_asst_body' if header_text == 'assistant' else 'prompt'
                    pending_header_ids = []
                else:
                    pending_header_ids.append(tok)

            elif state == 'in_asst_body':
                labels[i] = target_ids[i]   # token entraîné
                if tok == self._eot:
                    state = 'prompt'         # eot inclus, repasse en prompt

        return input_ids, labels


def make_collate_fn(pad_id: int):
    """Dynamic padding — pad au max du batch, pas au max_seq_len global."""
    def collate(batch):
        input_ids_list = [item[0] for item in batch]
        labels_list    = [item[1] for item in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100,
        )
        return input_ids, labels
    return collate

# ============================================================
# CHARGEMENT DATASETS
# ============================================================

def _make_split(samples: list):
    """Shuffle global puis split train/val."""
    rng = random.Random(42)
    rng.shuffle(samples)
    n_val   = max(50, int(len(samples) * CONFIG['val_split']))
    n_train = len(samples) - n_val
    print(f'  Total : {len(samples):,}  →  train={n_train:,}  val={n_val:,}')
    return (
        SFTDataset(samples[:n_train], CONFIG['max_seq_len']),
        SFTDataset(samples[n_train:], CONFIG['max_seq_len']),
    )


def load_stage1_dataset(num_samples_limit: Optional[int] = None):
    """Stage 1 : Bespoke-Stratos-17k uniquement."""
    print(f'\n{"="*80}\nSTAGE 1 DATASET — Bespoke-Stratos-17k\n{"="*80}')
    samples = []
    try:
        ds = load_dataset('bespokelabs/Bespoke-Stratos-17k', split='train')
        ds = ds.shuffle(seed=42)
        if num_samples_limit:
            ds = ds.select(range(min(num_samples_limit, len(ds))))

        for row in ds:
            msgs     = row.get('conversations', row.get('messages', []))
            thinking = row.get('thinking', row.get('reasoning', ''))
            if msgs:
                normalized = _normalize_messages(msgs)
                if thinking:
                    normalized = _inject_think(normalized, thinking)
                if normalized:
                    samples.append({'messages': normalized})
            else:
                instruction = row.get('instruction', row.get('prompt', ''))
                output      = row.get('output', row.get('response', ''))
                if thinking:
                    clean  = re.sub(r'<think>.*?</think>', '', output,
                                    flags=re.DOTALL).strip()
                    output = f'<think>{thinking.strip()}</think>\n{clean}'
                if instruction:
                    samples.append({'instruction': instruction, 'output': output})

        print(f'  OK : {len(samples):,} samples Stage 1')
    except Exception as e:
        raise ValueError(f'Stage 1 dataset introuvable : {e}')

    if not samples:
        raise ValueError('Stage 1 : aucun sample chargé depuis Bespoke-Stratos-17k')

    return _make_split(samples)


def compute_hes_score(text: str) -> float:
    """
    High-Entropy Sum — densité de noeuds de décision logiques.
    Heuristique textuelle pure, aucun modèle requis.
    """
    _HES_PATTERNS = [
        r'\?',
        r'\b(if|but|however|unless|although|whether|while|whereas|despite|yet)\b',
        r'\b(not|never|no|neither|nor|without|except)\b',
        r'\b(all|every|any|some|none|most|few|each|both|either)\b',
        r'\b(because|therefore|thus|hence|since|so that|consequently|as a result)\b',
        r'\b(first|second|third|then|finally|next|lastly|moreover|furthermore)\b',
        r'\b(if and only if|necessary|sufficient|implies|contradiction|proof)\b',
    ]
    words = text.split()
    if not words:
        return 0.0
    hits = sum(len(re.findall(p, text, re.IGNORECASE)) for p in _HES_PATTERNS)
    return hits / len(words)


def apply_hes_top20(samples: list, sample_to_text_fn) -> list:
    """Garde le Top 20% des samples par score HES."""
    if len(samples) < 5:
        return samples
    print(f'  HES : calcul sur {len(samples):,} samples...')
    scored = [(i, compute_hes_score(sample_to_text_fn(s))) for i, s in enumerate(samples)]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_n     = max(1, len(scored) // 5)
    top_idx   = set(i for i, _ in scored[:top_n])
    filtered  = [s for i, s in enumerate(samples) if i in top_idx]
    threshold = scored[top_n - 1][1] if top_n > 0 else 0
    print(f'  HES : {len(samples):,} → {len(filtered):,} (Top 20%)  seuil={threshold:.4f}')
    return filtered


def load_stage2_dataset(num_samples_limit: Optional[int] = None):
    """Stage 2 : Mix 5 datasets ~276k + HES Top20% sur OpenThoughts3."""
    print(f'\n{"="*80}\nSTAGE 2 DATASETS — Mix ~276k  (HES Top20% sur OpenThoughts3)\n{"="*80}')
    sizes       = CONFIG['dataset_sizes']
    all_samples = []

    # ── 1. PiKa-SFT 25% ────────────────────────────────────────
    print(f'\n[1/6] PiKa-SFT ~{sizes["pika_sft"]//1000}k (25%)...')
    try:
        ds = load_dataset('Shangjian/PiKa-SFT', split='train')
        ds = ds.shuffle(seed=42).select(range(min(sizes['pika_sft'], len(ds))))
        samples = []
        for row in ds:
            msgs = row.get('messages', row.get('conversations', []))
            if msgs:
                normalized = _normalize_messages(msgs)
                if normalized and not messages_have_code(normalized):
                    samples.append({'messages': normalized})
            else:
                instruction = row.get('instruction', row.get('input', ''))
                output      = row.get('output', row.get('response', ''))
                if instruction and not has_code_blocks(output):
                    samples.append({'instruction': instruction, 'output': output})
        all_samples.extend(samples)
        print(f'  OK : {len(samples):,} samples')
    except Exception as e:
        print(f'  SKIP PiKa-SFT : {e}')

    # ── 2. OpenThoughts3 Logic 35% + HES Top 20% ────────────────
    print(f'\n[2/6] OpenThoughts3 Logic ~{sizes["openthoughts3"]//1000}k (35%) + HES Top20%...')
    try:
        ds = load_dataset('HuggingFaceTB/smoltalk', 'openthoughts', split='train')

        KEEP_CATS   = {'science', 'puzzles', 'logic', 'reasoning', 'philosophy'}
        REJECT_CATS = {'code', 'program', 'math', 'calcul'}

        def is_logic(row):
            cat = str(row.get('category', row.get('source', ''))).lower()
            if any(x in cat for x in REJECT_CATS):
                return False
            return any(c in cat for c in KEEP_CATS)

        ds_logic = ds.filter(is_logic, num_proc=4)

        if len(ds_logic) < 500:
            print(f'  WARN filtre strict ({len(ds_logic)}) → garde Science+Logic')
            ds_logic = ds.filter(
                lambda r: any(c in str(r.get('category', '')).lower()
                              for c in ('science', 'logic')),
                num_proc=4,
            )
        if len(ds_logic) < 100:
            print(f'  WARN très peu ({len(ds_logic)}) → prend tout')
            ds_logic = ds

        # Prendre 5x plus pour que le Top 20% HES donne la bonne quantité finale
        n_before_hes = min(sizes['openthoughts3'] * 5, len(ds_logic))
        if len(ds_logic) < sizes['openthoughts3'] * 5:
            print(f'  WARN HES : {len(ds_logic):,} dispo < {sizes["openthoughts3"]*5:,} cible'
                  f' → résultat HES sera < {sizes["openthoughts3"]//1000}k')
        ds_logic = ds_logic.shuffle(seed=42).select(range(n_before_hes))

        pre_hes = []
        for row in ds_logic:
            msgs     = row.get('messages', [])
            thinking = row.get('thinking', '')
            if msgs:
                normalized = _normalize_messages(msgs)
                if thinking:
                    normalized = _inject_think(normalized, thinking)
                if normalized:
                    pre_hes.append({'messages': normalized})
            else:
                instruction = row.get('instruction', row.get('prompt', ''))
                output      = row.get('output', row.get('response', ''))
                if thinking:
                    clean  = re.sub(r'<think>.*?</think>', '', output,
                                    flags=re.DOTALL).strip()
                    output = f'<think>{thinking.strip()}</think>\n{clean}'
                if instruction:
                    pre_hes.append({'instruction': instruction, 'output': output})

        # HES Top 20% — garder uniquement les exemples les plus denses en raisonnement
        samples = apply_hes_top20(pre_hes, sample_to_text)
        if len(samples) > sizes['openthoughts3']:
            samples = samples[:sizes['openthoughts3']]

        all_samples.extend(samples)
        print(f'  OK : {len(samples):,} samples (Science/Logic/Puzzles + HES Top20%)')
    except Exception as e:
        print(f'  SKIP OpenThoughts3 : {e}')

    # ── 3. Smol-Magpie-Ultra 30% — filtre No-Code ───────────────
    print(f'\n[3/6] Smol-Magpie-Ultra ~{sizes["smol_magpie_ultra"]//1000}k (30%) + No-Code...')
    try:
        ds = load_dataset('HuggingFaceTB/smol-magpie-ultra', split='train')
        ds = ds.shuffle(seed=42)
        samples, n_scanned = [], 0
        for row in ds:
            if len(samples) >= sizes['smol_magpie_ultra']:
                break
            n_scanned += 1
            msgs = row.get('messages', [])
            if not msgs:
                continue
            if messages_have_code(msgs):
                continue
            samples.append({'messages': msgs})
        all_samples.extend(samples)
        reject_rate = (n_scanned - len(samples)) / max(n_scanned, 1) * 100
        print(f'  OK : {len(samples):,} samples  (rejet code : {reject_rate:.1f}%)')
    except Exception as e:
        print(f'  SKIP Smol-Magpie-Ultra : {e}')

    # ── 4. Aegis-AI Safety 5% ───────────────────────────────────
    # am_deepseek_math retiré — objectif assistant factuel, pas math
    print(f'\n[4/6] Aegis-AI Safety ~{sizes["aegis_safety"]//1000}k (5%)...')
    try:
        ds = load_dataset('nvidia/Aegis-AI-Content-Safety-Dataset-2.0', split='train')
        n  = min(sizes['aegis_safety'], len(ds))
        ds = ds.shuffle(seed=42).select(range(n))
        samples = []
        for row in ds:
            msgs = row.get('messages', [])
            if msgs:
                normalized = _normalize_messages(msgs)
                if normalized:
                    samples.append({'messages': normalized})
            else:
                prompt   = row.get('prompt', row.get('text', ''))
                response = row.get('response', row.get('label_text',
                           row.get('answer', '')))
                if prompt and response:
                    samples.append({'instruction': prompt, 'output': response})
        all_samples.extend(samples)
        print(f'  OK : {len(samples):,} samples')
    except Exception as e:
        print(f'  SKIP Aegis-AI Safety : {e}')

    # ── 5. RefGPT-Fact — Hallucination Mitigation (Annealing) ─────
    # Dataset de dialogues factuels basés sur Wikipedia (GPT-4 generated).
    # Uniquement le split anglais (50k). Taggé annealing=True → surpondéré
    # pendant la phase decay du WSD (Data Annealing).
    n_refgpt = CONFIG['dataset_sizes'].get('refgpt_fact', 10_000)
    print(f'\n[5/6] RefGPT-Fact ~{n_refgpt//1000}k (annealing)...')
    try:
        ds = load_dataset('Mutonix/RefGPT-Fact', split='en')
        ds = ds.shuffle(seed=42)
        n  = min(n_refgpt, len(ds))
        ds = ds.select(range(n))
        samples = []
        for row in ds:
            dialogue = row.get('dialogue', '')
            if not dialogue:
                continue
            # Parser le format <user> ... <assistant> ... multi-turn
            msgs = []
            parts = re.split(r'<(user|assistant)>', dialogue)
            # parts[0] est vide, puis alternance [role, contenu, role, contenu, ...]
            i = 1
            while i + 1 < len(parts):
                role    = parts[i].strip()
                content = parts[i + 1].strip()
                if role in ('user', 'assistant') and content:
                    msgs.append({'role': role, 'content': content})
                i += 2
            if len(msgs) >= 2:
                samples.append({'messages': msgs, 'annealing': True})
        all_samples.extend(samples)
        n_anneal = sum(1 for s in all_samples if s.get('annealing'))
        print(f'  OK : {len(samples):,} samples factuals (annealing=True)')
        print(f'  Total samples annealing dans le mix : {n_anneal:,}')
    except Exception as e:
        print(f'  SKIP RefGPT-Fact : {e}')

    # ── 6. LongAlign-10k — Long Context YaRN (conditionnel) ────────
    # Ajouté uniquement si use_yarn=True ET yarn_scale > 1.0
    # Fournit des séquences longues (8k-64k tokens, tronquées à max_seq_len)
    # pour aligner le modèle sur les contextes étendus activés par YaRN.
    if CONFIG.get('use_yarn', False) and CONFIG.get('yarn_scale', 1.0) > 1.0:
        n_long = CONFIG['dataset_sizes'].get('longalign', 14_000)
        print(f'\n[6/6] LongAlign-10k ~{n_long//1000}k '
              f'(YaRN x{CONFIG["yarn_scale"]} détecté — long-context alignment)...')
        try:
            ds = load_dataset('THUDM/LongAlign-10k', split='train')
            ds = ds.shuffle(seed=42)
            n  = min(n_long, len(ds))
            ds = ds.select(range(n))
            samples = []
            for row in ds:
                msgs = row.get('messages', row.get('conversations', []))
                if msgs:
                    normalized = _normalize_messages(msgs)
                    if normalized:
                        samples.append({'messages': normalized})
                else:
                    instruction = row.get('instruction', row.get('input', ''))
                    output      = row.get('output', row.get('response', ''))
                    if instruction and output:
                        samples.append({'instruction': instruction, 'output': output})
            all_samples.extend(samples)
            print(f'  OK : {len(samples):,} samples long-context '
                  f'(tronqués à {CONFIG["max_seq_len"]} tokens)')
        except Exception as e:
            print(f'  SKIP LongAlign-10k : {e}')
    else:
        print(f'\n[6/6] LongAlign-10k — SKIP '
              f'(use_yarn={CONFIG.get("use_yarn")}  '
              f'yarn_scale={CONFIG.get("yarn_scale", 1.0)})')

    if not all_samples:
        raise ValueError('Stage 2 : aucun dataset chargé — vérifier connexion')

    if num_samples_limit:
        all_samples = all_samples[:num_samples_limit]

    return _make_split(all_samples)

# ============================================================
# LORA
# ============================================================
class LoRALayer(nn.Module):
    """delta_W = B @ A * (alpha/r)   —   lora_B init à 0 → pas de perturbation initiale."""
    def __init__(self, in_features, out_features, r, alpha, dropout):
        super().__init__()
        self.scaling = alpha / r
        self.lora_A  = nn.Parameter(torch.empty(r, in_features))
        self.lora_B  = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(
            F.linear(self.dropout(x), self.lora_A), self.lora_B
        ) * self.scaling


class LinearWithLoRA(nn.Module):
    def __init__(self, base_layer, r, alpha, dropout):
        super().__init__()
        self.base_layer = base_layer
        self.lora       = LoRALayer(
            base_layer.in_features, base_layer.out_features,
            r=r, alpha=alpha, dropout=dropout,
        )

    def forward(self, x):
        return self.base_layer(x) + self.lora(x)


def apply_lora(model, r, alpha, dropout, target_modules):
    """Gèle tout le modèle, puis ajoute des branches LoRA sur les couches cibles."""
    for p in model.parameters():
        p.requires_grad = False

    count = 0
    for name, module in list(model.named_modules()):
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear) \
                    and not isinstance(module, LinearWithLoRA):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name  = name.split('.')[-1]
                parent      = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name,
                        LinearWithLoRA(module, r=r, alpha=alpha, dropout=dropout))
                count += 1
                break   # évite double-match sur même module

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nLoRA : {count} couches  '
          f'({trainable:,} params = {trainable/total*100:.2f}%)')
    return model, trainable

# ============================================================
# MUON + MARS-M OPTIMIZER
# ============================================================
# Même implémentation que pretrain.py.
# En SFT LoRA, seuls les adaptateurs LoRA sont trainables.
# Muon+MARS-M s'applique sur lora_A et lora_B (matrices 2D).
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
    """
    Muon + MARS-M (Momentum-Aware Rescaled Stochastic gradient)
    Même implémentation que pretrain.py — appliqué sur lora_A et lora_B.

    MARS-M corrige le biais du momentum :
        c_t = (γ / (1 - γ)) * ||g_t|| / (||g_{t-1}|| + ε)
        g̃_t = g_t + c_t * (g_t - g_{t-1})
    Puis Muon applique Newton-Schulz sur g̃_t.
    """
    def __init__(self, params, lr=0.02, momentum=0.95,
                 nesterov=True, ns_steps=3, weight_decay=0.0,
                 use_mars=True, mars_gamma=0.025):
        defaults = dict(lr=lr, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay,
                        use_mars=use_mars, mars_gamma=mars_gamma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr         = group['lr']
            momentum   = group['momentum']
            nesterov   = group['nesterov']
            ns_steps   = group['ns_steps']
            wd         = group['weight_decay']
            use_mars   = group.get('use_mars', True)
            mars_gamma = group.get('mars_gamma', 0.025)

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    continue
                state = self.state[p]

                # ── MARS-M correction ─────────────────────────────
                if use_mars:
                    if 'prev_grad' not in state:
                        state['prev_grad'] = torch.zeros_like(g)
                    prev_g    = state['prev_grad']
                    norm_g    = g.norm() + 1e-8
                    norm_prev = prev_g.norm() + 1e-8
                    c_t = (mars_gamma / (1.0 - mars_gamma)) * (norm_g / norm_prev)
                    c_t = torch.clamp(c_t, max=1.0)   # tensor, pas .item() → pas de sync CUDA
                    g   = g + c_t * (g - prev_g)
                    state['prev_grad'].copy_(p.grad)   # grad original avant correction

                # ── Momentum buffer ───────────────────────────────
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g + momentum * buf if nesterov else buf

                # ── Newton-Schulz orthogonalisation ──────────────
                g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                scale = max(g.size(0), g.size(1)) ** 0.5
                g = g * scale

                # ── Weight decay + update ─────────────────────────
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)


def make_sft_optimizers(model, lr, weight_decay=0.01):
    """
    Partitionne les paramètres LoRA en :
      - Muon+MARS-M : lora_A et lora_B (matrices 2D) des couches attention/FFN
      - AdamW       : tout le reste (scaling, biases éventuels)
    """
    muon_params  = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and ('lora_A' in name or 'lora_B' in name):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    lr_muon  = lr * 5.0
    muon_opt = Muon(
        [{'params': muon_params, 'is_muon': True}],
        lr           = lr_muon,
        momentum     = 0.95,
        nesterov     = True,
        ns_steps     = 5,
        weight_decay = 0.0,
        use_mars     = True,
        mars_gamma   = 0.025,
    )
    adamw_opt = torch.optim.AdamW(
        [{'params': adamw_params, 'is_muon': False}], lr=lr,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay,
        fused=(device == 'cuda'),
    )
    print(f'  Muon+MARS-M : {len(muon_params):3d} tenseurs LoRA 2D  lr={lr_muon:.2e}  mars_gamma=0.025')
    print(f'  AdamW       : {len(adamw_params):3d} tenseurs           lr={lr:.2e}')
    return muon_opt, adamw_opt


# ============================================================
# WSD SCHEDULER
# ============================================================
class WSDScheduler:
    """
    WSD schedule — supporte une liste d'optimizers (Muon + AdamW).
    Muon garde son ratio lr 5x par rapport à AdamW à chaque step.
    """
    def __init__(self, optimizers, max_lr, total_steps,
                 warmup_ratio=0.05, decay_ratio=0.10, min_lr_ratio=0.1):
        self.optimizers   = optimizers if isinstance(optimizers, list) else [optimizers]
        self.max_lr       = max_lr
        self.min_lr       = max_lr * min_lr_ratio
        self.total_steps  = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.decay_steps  = int(total_steps * decay_ratio)
        self.stable_steps = total_steps - self.warmup_steps - self.decay_steps
        self.current_step = 0
        print(f'  WSD : warmup={self.warmup_steps:,}  '
              f'stable={self.stable_steps:,}  decay={self.decay_steps:,}')
        print(f'  AdamW {self.min_lr:.2e} → {self.max_lr:.2e}  '
              f'| Muon {self.min_lr*5:.2e} → {self.max_lr*5:.2e}')

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
                # ✅ FIX : tag is_muon explicite — même fix que pretrain.py
                pg['lr'] = lr * 5.0 if pg.get('is_muon', False) else lr
        return lr

    def get_last_lr(self):
        return [self.get_lr()]

    def state_dict(self):
        return {'current_step': self.current_step}

    def load_state_dict(self, sd):
        self.current_step = sd['current_step']

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
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                n += 1
    finally:
        model.train()
    avg = total_loss / max(n, 1)
    return math.exp(min(avg, 10)), avg

# ============================================================
# CHECKPOINT MANAGER
# ============================================================
class CheckpointManager:
    def __init__(self, path):
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    def save(self, model, optimizers, scheduler, metadata):
        """
        Sauvegarde atomique — seuls les poids LoRA (requires_grad) sauvés.
        optimizers : [muon_opt, adamw_opt] ou optimizer unique.
        """
        lora_state = {
            name: p.data.cpu()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        if isinstance(optimizers, (list, tuple)):
            muon_opt, adamw_opt = optimizers
            opt_state = {
                'muon_state_dict':  muon_opt.state_dict(),
                'adamw_state_dict': adamw_opt.state_dict(),
            }
        else:
            opt_state = {'optimizer_state_dict': optimizers.state_dict()}
        cp = {
            'lora_state_dict':      lora_state,
            **opt_state,
            'scheduler_state_dict': scheduler.state_dict(),
            'stage':                metadata['stage'],
            'epoch':                metadata['epoch'],
            'global_step':          metadata['global_step'],
            'training_history':     metadata['training_history'],
            'config':               CONFIG,
            'special_tokens':       SPECIAL_TOKENS,
            'last_save':            datetime.now().isoformat(),
        }
        tmp = self.path + '.tmp'
        torch.save(cp, tmp)
        os.replace(tmp, self.path)   # atomique — jamais de .pt corrompu
        print(f'  💾 SAVE → stage={metadata["stage"]}  '
              f'epoch={metadata["epoch"]}  '
              f'step={metadata["global_step"]:,}  [{self.path}]')

    def load(self):
        if not os.path.exists(self.path):
            return None
        print(f'\nCheckpoint SFT trouvé : {self.path}')
        cp = torch.load(self.path, map_location='cpu', weights_only=False)
        print(f'  stage={cp.get("stage","?")}  '
              f'epoch={cp.get("epoch","?")}  '
              f'step={cp.get("global_step",0):,}  '
              f'saved={cp.get("last_save","?")}')
        return cp

# ============================================================
# TRAIN EPOCH (partagé Stage 1 et Stage 2)
# ============================================================
def _make_anneal_loader(train_ds, stage_cfg, collate_fn, anneal_factor=5.0):
    """
    Crée un DataLoader avec WeightedRandomSampler pour le Data Annealing.
    Les samples RefGPT-Fact (annealing=True) reçoivent anneal_factor × plus de poids.
    Appelé dès que le WSD scheduler entre en phase decay.
    """
    from torch.utils.data import WeightedRandomSampler
    weights  = train_ds.get_annealing_weights(anneal_factor)
    n_anneal = int((weights > 1.0).sum().item())
    sampler  = WeightedRandomSampler(
        weights       = weights,
        num_samples   = len(train_ds),
        replacement   = True,
    )
    loader = DataLoader(
        train_ds,
        batch_size  = stage_cfg['batch_size'],
        sampler     = sampler,
        collate_fn  = collate_fn,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True,
    )
    print(f'  🔥 DATA ANNEALING activé — {n_anneal:,} samples RefGPT-Fact '
          f'surpondérés ×{anneal_factor:.0f}')
    return loader


def train_epoch(
    model, train_loader, optimizers, scheduler,
    val_loader, checkpoint_manager, training_history,
    stage, epoch, global_step, stage_cfg,
):
    """
    optimizers  : [muon_opt, adamw_opt]

    Data Annealing :
      Le loader correct (normal ou anneal) est sélectionné par main() AVANT
      d'appeler train_epoch — selon si scheduler.current_step >= warmup + stable.
      Ici train_loader contient déjà le bon loader pour cette epoch.
    """
    muon_opt, adamw_opt = optimizers

    print(f'\n{"="*80}')
    print(f'STAGE {stage} — Epoch {epoch}  |  LR_adamw={scheduler.get_last_lr()[0]:.2e}'
          f'  LR_muon={scheduler.get_last_lr()[0]*5:.2e}')
    print(f'{"="*80}')

    model.train()
    epoch_loss        = 0.0
    valid_batches     = 0
    accumulated_steps = 0
    running_loss      = 0.0
    running_batches   = 0
    t_start           = time.time()
    ae                = (device == 'cuda')
    adt               = torch.bfloat16 if ae else torch.float32
    num_batches       = len(train_loader)

    pbar = tqdm(train_loader, desc=f'S{stage} Ep{epoch}', leave=True)

    for batch_idx, (x, y) in enumerate(pbar):
        try:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Skip batch entièrement masqué
            if (y != -100).sum() == 0:
                continue

            with torch.amp.autocast(device, dtype=adt, enabled=ae):
                _, loss, _ = model(x, targets=y, pad_token_id=tokenizer.pad_token_id)
                loss    = loss / stage_cfg['gradient_accumulation']

            if torch.isnan(loss) or torch.isinf(loss):
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                accumulated_steps = 0
                continue

            loss.backward()
            accumulated_steps += 1

            is_last = (batch_idx + 1 == num_batches)
            if (accumulated_steps % stage_cfg['gradient_accumulation'] == 0) or is_last:
                if accumulated_steps > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), stage_cfg['max_grad_norm']
                    )
                    muon_opt.step()
                    adamw_opt.step()
                    muon_opt.zero_grad(set_to_none=True)
                    adamw_opt.zero_grad(set_to_none=True)
                    scheduler.step()
                    accumulated_steps = 0
                    global_step += 1

                    # ── Validation ──────────────────────────────
                    if global_step % stage_cfg['validate_every_steps'] == 0:
                        val_ppl, val_loss = validate(
                            model, val_loader, stage_cfg['val_batches']
                        )
                        avg = running_loss / max(running_batches, 1)
                        print(f'\n  step={global_step:,} | '
                              f'train={avg:.4f} ppl={math.exp(min(avg,10)):.1f} | '
                              f'val={val_loss:.4f} ppl={val_ppl:.1f} | '
                              f'lr={scheduler.get_last_lr()[0]:.2e}\n')
                        training_history['validations'].append({
                            'step':       global_step,
                            'stage':      stage,
                            'epoch':      epoch,
                            'val_loss':   val_loss,
                            'val_ppl':    val_ppl,
                            'train_loss': avg,
                            'lr':         scheduler.get_last_lr()[0],
                        })
                        running_loss    = 0.0
                        running_batches = 0

                    # ── Safety save intra-epoch ──────────────────
                    if global_step % stage_cfg['save_every_steps'] == 0:
                        checkpoint_manager.save(model, optimizers, scheduler, metadata={
                            'stage':            stage,
                            'epoch':            epoch,
                            'global_step':      global_step,
                            'training_history': training_history,
                        })

            raw = loss.item() * stage_cfg['gradient_accumulation']
            epoch_loss      += raw
            valid_batches   += 1
            running_loss    += raw
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
                print(f'\n  OOM batch {batch_idx} — skip')
                torch.cuda.empty_cache()
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                accumulated_steps = 0
                gc.collect()
                model.train()
                continue
            raise

    pbar.close()

    avg_loss          = epoch_loss / max(valid_batches, 1)
    elapsed           = time.time() - t_start
    val_ppl, val_loss = validate(model, val_loader, stage_cfg['val_batches'])

    print(f'\n  Stage {stage} Epoch {epoch} terminée | '
          f'train={avg_loss:.4f} | val={val_loss:.4f} ppl={val_ppl:.1f} | '
          f'{elapsed/60:.1f}min')

    training_history['epochs'].append({
        'stage': stage, 'epoch': epoch,
        'train_loss': avg_loss, 'val_loss': val_loss, 'val_ppl': val_ppl,
        'global_step': global_step, 'time_sec': elapsed,
        'lr': scheduler.get_last_lr()[0],
    })

    return global_step

# ============================================================
# MASKING UNIT TEST — vérifié AVANT toute chose dans main()
# ============================================================
def test_masking():
    print('\n=== TEST MASKING ===')
    system = 'You are helpful.'
    user   = 'What is 2+2?'
    asst   = '<think>Simple addition.</think>\nThe answer is 4.'

    ds = SFTDataset([{'messages': [
        {'role': 'system',    'content': system},
        {'role': 'user',      'content': user},
        {'role': 'assistant', 'content': asst},
    ]}], max_seq_len=512)

    inp, lbl     = ds[0]
    n_total      = len(lbl)
    n_trained    = (lbl != -100).sum().item()
    n_masked     = n_total - n_trained
    trained_text = tokenizer.decode(inp[lbl != -100].tolist())

    print(f'  Tokens : {n_total}  masqués={n_masked}  entraînés={n_trained}')
    print(f'  Entraîné sur : {repr(trained_text[:150])}')

    ok       = n_trained > 0 and ('answer' in trained_text.lower() or '4' in trained_text)
    think_ok = '<think>' in trained_text

    print(f'  Masking assistant-only : {"✅ OK" if ok else "❌ FAIL"}')
    print(f'  <think> inclus dans loss : {"✅ OK" if think_ok else "⚠️  absent"}')
    print('=== FIN TEST MASKING ===\n')

    if not ok:
        raise RuntimeError(
            'MASKING TEST FAILED — le masking est cassé, on arrête avant de gaspiller du compute'
        )
    return True

# ============================================================
# MAIN
# ============================================================
def main():
    from HessGpt import HessGPT

    # ── Test masking AVANT tout ──────────────────────────────────
    test_masking()

    print('\n' + '='*80 + '\nCREATION MODELE\n' + '='*80)

    ckpt_mgr = CheckpointManager(args.output_checkpoint)

    model = HessGPT(
        vocab_size            = CONFIG['vocab_size'],
        embed_dim             = CONFIG['embed_dim'],
        num_heads             = CONFIG['num_heads'],
        num_layers            = CONFIG['num_layers'],
        max_seq_len           = CONFIG['max_seq_len'],
        dropout               = CONFIG['dropout'],
        use_rope              = CONFIG['use_rope'],
        use_yarn              = CONFIG['use_yarn'],
        yarn_scale            = CONFIG['yarn_scale'],
        yarn_original_max_len = CONFIG['yarn_original_max_len'],
        use_swiglu            = CONFIG['use_swiglu'],
        n_kv_heads            = CONFIG['n_kv_heads'],
        use_qk_norm           = CONFIG['use_qk_norm'],
        soft_cap              = CONFIG['soft_cap'],
        use_flash_attn        = CONFIG['use_flash_attn'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Params : {total_params/1e6:.1f}M')

    # ── Charger poids pretrain ───────────────────────────────────
    if os.path.exists(args.pretrain_checkpoint):
        print(f'\nChargement pretrain : {args.pretrain_checkpoint}')
        pt_cp = torch.load(args.pretrain_checkpoint, map_location='cpu',
                           weights_only=False)
        sd = pt_cp.get('model_state_dict', pt_cp)
        if any(k.startswith('_orig_mod.') for k in sd.keys()):
            sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f'  WARN missing : {missing[:3]}{"..." if len(missing)>3 else ""}')
        if unexpected:
            print(f'  WARN unexpected : {unexpected[:3]}{"..." if len(unexpected)>3 else ""}')
        print(f'  OK')
    else:
        print(f'\nWARN : pretrain introuvable ({args.pretrain_checkpoint}) — from scratch')

    # ── Appliquer LoRA (APRÈS chargement pretrain) ───────────────
    model, trainable_params = apply_lora(
        model,
        r              = CONFIG['lora_r'],
        alpha          = CONFIG['lora_alpha'],
        dropout        = CONFIG['lora_dropout'],
        target_modules = CONFIG['lora_target_modules'],
    )

    # ── Dry run ──────────────────────────────────────────────────
    if args.dry_run:
        print('\n=== DRY RUN ===')
        print('--- Stage 1 ---')
        tr1, v1 = load_stage1_dataset(num_samples_limit=100)
        for i in range(min(3, len(tr1))):
            inp, lbl = tr1[i]
            print(f'  s1[{i}]: tokens={len(inp)} entraînés={(lbl!=-100).sum().item()}')
        print('--- Stage 2 ---')
        tr2, v2 = load_stage2_dataset(num_samples_limit=200)
        for i in range(min(3, len(tr2))):
            inp, lbl = tr2[i]
            print(f'  s2[{i}]: tokens={len(inp)} entraînés={(lbl!=-100).sum().item()}')
        print('=== DRY RUN OK ===')
        return

    # ── État de reprise ──────────────────────────────────────────
    training_history = {
        'config': CONFIG, 'special_tokens': SPECIAL_TOKENS,
        'total_params': total_params, 'trainable_params': trainable_params,
        'epochs': [], 'validations': [], 'start_time': datetime.now().isoformat(),
    }
    global_step = 0
    start_stage = 1
    start_epoch = 1

    cp = ckpt_mgr.load()
    if cp:
        print('\nREPRISE')
        # Restaurer poids LoRA
        lora_sd = cp.get('lora_state_dict', {})
        for name, p in model.named_parameters():
            if p.requires_grad and name in lora_sd:
                p.data.copy_(lora_sd[name])
        global_step      = cp.get('global_step', 0)
        training_history = cp.get('training_history', training_history)
        saved_stage      = cp.get('stage', 1)
        saved_epoch      = cp.get('epoch', 0)

        if saved_stage == 2 and saved_epoch >= CONFIG['stage2']['epochs']:
            print(f'\n✅ SFT entièrement terminé (Stage 1 + Stage 2).')
            print(f'   Checkpoint : {ckpt_mgr.path}')
            return
        elif saved_stage == 1 and saved_epoch >= CONFIG['stage1']['epochs']:
            # Stage 1 terminé → démarre Stage 2
            start_stage = 2
            start_epoch = 1
        else:
            start_stage = saved_stage
            start_epoch = saved_epoch + 1   # reprend à l'epoch SUIVANTE

        print(f'  -> stage={start_stage}  epoch={start_epoch}  step={global_step:,}')

    # Override via --stage
    if args.stage is not None:
        start_stage = args.stage
        start_epoch = 1
        print(f'  -> --stage {args.stage} forcé')

    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    print('\n' + '='*80)
    print(f'SFT 2-STAGE  |  départ stage={start_stage} epoch={start_epoch}')
    print('='*80)

    # ══════════════════════════════════════════════════════════════
    # STAGE 1 — Cold Start
    # ══════════════════════════════════════════════════════════════
    if start_stage <= 1:
        s1 = CONFIG['stage1']
        print(f'\n{"="*80}')
        print(f'STAGE 1 — Cold Start | Bespoke-Stratos-17k | '
              f'{s1["epochs"]} epoch | LR={s1["learning_rate"]:.0e}')
        print(f'{"="*80}')

        train_ds, val_ds = load_stage1_dataset(args.num_samples)
        train_loader = DataLoader(train_ds, batch_size=s1['batch_size'],
                                  shuffle=True, collate_fn=collate_fn,
                                  num_workers=4, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=s1['batch_size'],
                                  shuffle=False, collate_fn=collate_fn,
                                  num_workers=2, pin_memory=True)
        print(f'  Loaders : train={len(train_loader):,}  val={len(val_loader):,}')

        muon_opt_s1, adamw_opt_s1 = make_sft_optimizers(model, s1['learning_rate'])
        optimizers_s1  = [muon_opt_s1, adamw_opt_s1]
        total_steps_s1 = math.ceil(len(train_loader) * s1['epochs']
                               / s1['gradient_accumulation'])
        scheduler = WSDScheduler(
            optimizers_s1, max_lr=s1['learning_rate'],
            total_steps=total_steps_s1,
            warmup_ratio=s1['warmup_ratio'],
            decay_ratio=s1['decay_ratio'],
            min_lr_ratio=s1['min_lr_ratio'],
        )

        # Restaurer optimizer/scheduler si reprise en Stage 1
        if cp and cp.get('stage') == 1:
            if 'muon_state_dict' in cp and 'adamw_state_dict' in cp:
                muon_opt_s1.load_state_dict(cp['muon_state_dict'])
                adamw_opt_s1.load_state_dict(cp['adamw_state_dict'])
            elif 'optimizer_state_dict' in cp:
                print('  WARN : checkpoint ancien format — Muon repart de zéro')
                adamw_opt_s1.load_state_dict(cp['optimizer_state_dict'])
            scheduler.load_state_dict(cp['scheduler_state_dict'])
            for pg in muon_opt_s1.param_groups:
                pg['lr'] = scheduler.get_lr() * 5.0
            for pg in adamw_opt_s1.param_groups:
                pg['lr'] = scheduler.get_lr()

        for epoch in range(start_epoch, s1['epochs'] + 1):
            try:
                global_step = train_epoch(
                    model, train_loader, optimizers_s1, scheduler,
                    val_loader, ckpt_mgr, training_history,
                    stage=1, epoch=epoch, global_step=global_step, stage_cfg=s1,
                )
            except KeyboardInterrupt:
                print('\nCTRL+C — sauvegarde Stage 1...')
                ckpt_mgr.save(model, optimizers_s1, scheduler, metadata={
                    'stage': 1, 'epoch': epoch,
                    'global_step': global_step, 'training_history': training_history,
                })
                return
            except Exception:
                print(f'\nERREUR Stage 1 epoch {epoch}:\n{traceback.format_exc()}')
                ckpt_mgr.save(model, optimizers_s1, scheduler, metadata={
                    'stage': 1, 'epoch': epoch,
                    'global_step': global_step, 'training_history': training_history,
                })
                raise

            ckpt_mgr.save(model, optimizers_s1, scheduler, metadata={
                'stage': 1, 'epoch': epoch,
                'global_step': global_step, 'training_history': training_history,
            })

        print(f'\n✅ STAGE 1 TERMINÉ — circuit <think> initialisé')
        del train_ds, val_ds, train_loader, val_loader, optimizers_s1, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════
    # STAGE 2 — Full Mix
    # ══════════════════════════════════════════════════════════════
    s2          = CONFIG['stage2']
    s2_start_ep = start_epoch if start_stage == 2 else 1

    print(f'\n{"="*80}')
    print(f'STAGE 2 — Full Mix | ~275k | {s2["epochs"]} epochs | LR={s2["learning_rate"]:.0e}')
    print(f'{"="*80}')

    train_ds, val_ds = load_stage2_dataset(args.num_samples)
    train_loader = DataLoader(train_ds, batch_size=s2['batch_size'],
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=s2['batch_size'],
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)
    print(f'  Loaders : train={len(train_loader):,}  val={len(val_loader):,}')

    muon_opt_s2, adamw_opt_s2 = make_sft_optimizers(model, s2['learning_rate'])
    optimizers_s2  = [muon_opt_s2, adamw_opt_s2]
    total_steps_s2 = math.ceil(len(train_loader) * s2['epochs']
                           / s2['gradient_accumulation'])
    scheduler = WSDScheduler(
        optimizers_s2, max_lr=s2['learning_rate'],
        total_steps=total_steps_s2,
        warmup_ratio=s2['warmup_ratio'],
        decay_ratio=s2['decay_ratio'],
        min_lr_ratio=s2['min_lr_ratio'],
    )

    # Restaurer optimizer/scheduler si reprise en Stage 2
    if cp and cp.get('stage') == 2:
        if 'muon_state_dict' in cp and 'adamw_state_dict' in cp:
            muon_opt_s2.load_state_dict(cp['muon_state_dict'])
            adamw_opt_s2.load_state_dict(cp['adamw_state_dict'])
        elif 'optimizer_state_dict' in cp:
            print('  WARN : checkpoint ancien format — Muon repart de zéro')
            adamw_opt_s2.load_state_dict(cp['optimizer_state_dict'])
        scheduler.load_state_dict(cp['scheduler_state_dict'])
        for pg in muon_opt_s2.param_groups:
            pg['lr'] = scheduler.get_lr() * 5.0
        for pg in adamw_opt_s2.param_groups:
            pg['lr'] = scheduler.get_lr()

    # Loader annealing pré-créé seulement si des samples RefGPT-Fact existent.
    # Si RefGPT a été skippé (exception), tous les poids = 1.0 → WeightedRandomSampler
    # équivalent à un shuffle ordinaire, mais le message "ANNEAL activé" serait trompeur.
    n_anneal_samples = sum(1 for s in train_ds.samples if s.get('annealing'))
    if n_anneal_samples > 0:
        anneal_loader = _make_anneal_loader(
            train_ds, s2, collate_fn,
            anneal_factor=CONFIG.get('anneal_factor', 5.0),
        )
        print(f'  Annealing prêt : {n_anneal_samples:,} samples RefGPT-Fact indexés')
    else:
        anneal_loader = None
        print('  WARN : aucun sample annealing dans le mix — Data Annealing désactivé')

    for epoch in range(s2_start_ep, s2['epochs'] + 1):
        # ── Data Annealing : choisir le bon loader pour cette epoch ──
        # On entre en phase decay quand current_step >= warmup + stable.
        in_decay = (anneal_loader is not None
                    and scheduler.current_step >= scheduler.warmup_steps
                    + scheduler.stable_steps)
        epoch_loader = anneal_loader if in_decay else train_loader
        if in_decay:
            print(f'  🔥 ANNEAL epoch {epoch} — loader RefGPT surpondéré ×'
                  f'{CONFIG.get("anneal_factor", 5.0):.0f}')
        try:
            global_step = train_epoch(
                model, epoch_loader, optimizers_s2, scheduler,
                val_loader, ckpt_mgr, training_history,
                stage=2, epoch=epoch, global_step=global_step, stage_cfg=s2,
            )
        except KeyboardInterrupt:
            print('\nCTRL+C — sauvegarde Stage 2...')
            ckpt_mgr.save(model, optimizers_s2, scheduler, metadata={
                'stage': 2, 'epoch': epoch,
                'global_step': global_step, 'training_history': training_history,
            })
            return
        except Exception:
            print(f'\nERREUR Stage 2 epoch {epoch}:\n{traceback.format_exc()}')
            ckpt_mgr.save(model, optimizers_s2, scheduler, metadata={
                'stage': 2, 'epoch': epoch,
                'global_step': global_step, 'training_history': training_history,
            })
            raise

        ckpt_mgr.save(model, optimizers_s2, scheduler, metadata={
            'stage': 2, 'epoch': epoch,
            'global_step': global_step, 'training_history': training_history,
        })

    # ── FIN ──────────────────────────────────────────────────────
    print(f'\n{"="*80}\nSFT 2-STAGE TERMINÉ\n{"="*80}')
    print(f'  Steps total : {global_step:,}')
    if training_history.get('validations'):
        last = training_history['validations'][-1]
        print(f'  Val PPL final : {last["val_ppl"]:.2f}  '
              f'Val Loss : {last["val_loss"]:.4f}')
    print(f'  Checkpoint : {ckpt_mgr.path}')

    history_path = args.output_checkpoint.replace('.pt', '_history.json')
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
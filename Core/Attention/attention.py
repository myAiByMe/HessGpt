# attention.py - v8 — Soft Cap
"""
Multi-Head Attention avec RoPE + YaRN + Flash Attention + KV Cache + Soft Cap

SOFT CAP (v8) :
  Technique Gemma 2 / Gemini — borne les scores d'attention via tanh pour
  éviter que softmax devienne trop piquée (un seul token monopolise l'attention).

  Formule : scores = soft_cap * tanh(scores / soft_cap)
    → scores bornés dans [-soft_cap, +soft_cap]
    → gradient non-nul partout (contrairement à un hard clamp)
    → stabilité training améliorée, surtout en début de run

  Valeur typique : 30.0 (Gemma 2), 50.0 (plus permissif)

  Compatibilité :
    - soft_cap=None → Flash Attention (SDPA) natif, aucun overhead, comportement identique v7
    - soft_cap actif → chemin manuel (Q@K^T → scale → tanh → masque → softmax → @V)
      PyTorch SDPA ne supporte pas le soft cap nativement. Avec torch.compile
      sur H100, le chemin manuel reste très performant (kernel fusionné).

KV CACHE :
  Utilisé uniquement en inférence (generate). Pendant le training, past_kv=None
  et le comportement est identique à v6.

  Principe :
    - Premier appel (prefill) : seq_len = prompt complet, past_kv=None
      → calcule K/V sur toute la séquence, les retourne dans kv_cache
    - Appels suivants (decode) : seq_len = 1 (un token à la fois), past_kv fourni
      → calcule K/V pour le nouveau token uniquement
      → concatène avec le cache existant pour l'attention complète
      → retourne le cache mis à jour

  Format kv_cache par layer :
    Tuple (k, v) où :
      k : [batch, n_kv_heads, total_seq_len, head_dim]
      v : [batch, n_kv_heads, total_seq_len, head_dim]

  Le cache est géré dans HessGPT.generate() — MultiHeadAttention ne fait
  que recevoir/retourner son propre (k, v).

COMPATIBILITÉ :
  - Training : past_kv=None → pas de cache, comportement identique
  - Flash Attention : utilisé en prefill (seq_len > 1) ET en decode (seq_len=1)
  - GQA : le cache stocke K/V non-répétés (n_kv_heads, pas num_heads)
    pour minimiser la mémoire
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    """RMSNorm - Plus rapide et simple que LayerNorm"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ============================================================
# RoPE + YaRN
# ============================================================

class RotaryPositionalEmbedding(nn.Module):
    """RoPE avec YaRN et support KV cache (position_offset)."""
    def __init__(self, dim, max_seq_len=2048, base=10000, device=None,
                 use_yarn=False, yarn_scale=1.0, yarn_original_max_len=1024):
        super().__init__()

        self.dim                  = dim
        self.max_seq_len          = max_seq_len
        self.base                 = base
        self.use_yarn             = use_yarn
        self.yarn_scale           = yarn_scale
        self.yarn_original_max_len = yarn_original_max_len

        if use_yarn:
            assert 0.1 <= yarn_scale <= 16.0, \
                f"yarn_scale must be in [0.1, 16.0], got {yarn_scale}"

        if use_yarn:
            inv_freq = self._compute_yarn_frequencies()
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        self.register_buffer('inv_freq', inv_freq)

        self._seq_len_cached = None
        self._cos_cached     = None
        self._sin_cached     = None

    def _compute_yarn_frequencies(self):
        freqs        = torch.arange(0, self.dim, 2).float() / self.dim
        inv_freq_base = 1.0 / (self.base ** freqs)

        if self.yarn_scale == 1.0:
            return inv_freq_base

        alpha = self.yarn_scale
        beta  = max(self.dim // 2, int(self.dim * 0.25))
        dims  = torch.arange(0, self.dim, 2).float()
        scale = torch.where(
            dims < beta,
            torch.ones_like(dims),
            1 + (alpha - 1) * (dims - beta) / (self.dim - beta)
        )
        return inv_freq_base / scale

    def _update_cos_sin_cache(self, seq_len, device, dtype):
        if (seq_len != self._seq_len_cached or
                self._cos_cached is None or
                self._cos_cached.device != device or
                self._cos_cached.dtype != dtype):
            self._seq_len_cached = seq_len
            t     = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(dtype))
            emb   = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().clone()
            self._sin_cached = emb.sin().clone()
        return self._cos_cached, self._sin_cached

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, position_offset: int = 0):
        """
        Applique RoPE à q et k.

        Args:
            q               : [batch, heads, seq_len, head_dim]
            k               : [batch, n_kv_heads, seq_len, head_dim]
            position_offset : positions déjà dans le cache KV (pour le decode)
                              → les nouvelles positions démarrent à position_offset
        """
        seq_len      = q.shape[2]
        total_len    = seq_len + position_offset

        # Cache calculé sur la longueur totale nécessaire
        cos, sin = self._update_cos_sin_cache(total_len, q.device, q.dtype)

        # Slice pour les positions du nouveau fragment
        cos = cos[position_offset : position_offset + seq_len]  # [seq_len, dim]
        sin = sin[position_offset : position_offset + seq_len]
        cos = cos[None, None, :, :]   # [1, 1, seq_len, dim]
        sin = sin[None, None, :, :]

        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot

    def forward(self, q, k, position_offset: int = 0):
        return self.apply_rotary_pos_emb(q, k, position_offset)


# ============================================================
# Multi-Head Attention + KV Cache
# ============================================================

# Type alias pour le cache d'un layer
KVCache = Tuple[torch.Tensor, torch.Tensor]   # (k, v)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention avec RoPE + YaRN + GQA + QK-Norm + Flash Attention + KV Cache.

    KV Cache :
      - past_kv=None  → training ou prefill premier token : pas de cache
      - past_kv=(k,v) → decode : on concatène le nouveau (k,v) au cache existant

    Retourne :
      output  : [batch, seq_len, embed_dim]
      kv_cache: (k, v) mis à jour — None si past_kv était None ET use_kv_cache=False
                Toujours retourné si use_kv_cache=True
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1,
                 use_rope=True, max_seq_len=2048,
                 use_yarn=False, yarn_scale=1.0, yarn_original_max_len=1024,
                 n_kv_heads=None, use_qk_norm=False, use_flash_attn=True,
                 soft_cap=None):
        super().__init__()

        assert embed_dim % num_heads == 0, \
            "embed_dim doit être divisible par num_heads"

        if soft_cap is not None:
            assert soft_cap > 0, f"soft_cap doit être > 0, got {soft_cap}"

        self.embed_dim      = embed_dim
        self.num_heads      = num_heads
        self.head_dim       = embed_dim // num_heads
        self.use_rope       = use_rope
        self.use_qk_norm    = use_qk_norm
        self.use_flash_attn = use_flash_attn
        self.soft_cap       = soft_cap   # ✅ v8 : None = désactivé

        # GQA
        self.n_kv_heads          = n_kv_heads if n_kv_heads is not None else num_heads
        assert num_heads % self.n_kv_heads == 0, \
            f"num_heads ({num_heads}) doit être divisible par n_kv_heads ({self.n_kv_heads})"
        self.num_queries_per_kv  = num_heads // self.n_kv_heads
        self.kv_dim              = self.n_kv_heads * self.head_dim

        # Projections
        self.q_proj   = nn.Linear(embed_dim, embed_dim,    bias=False)
        self.k_proj   = nn.Linear(embed_dim, self.kv_dim,  bias=False)
        self.v_proj   = nn.Linear(embed_dim, self.kv_dim,  bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim,    bias=False)
        self.dropout  = nn.Dropout(dropout)

        # QK-Norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        # RoPE
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                self.head_dim, max_seq_len,
                use_yarn             = use_yarn,
                yarn_scale           = yarn_scale,
                yarn_original_max_len = yarn_original_max_len,
            )
        else:
            self.rope = None

        # Flash Attention check
        self._flash_attn_available = False
        if use_flash_attn:
            try:
                F.scaled_dot_product_attention
                self._flash_attn_available = True
            except AttributeError:
                print("⚠️  Flash Attention non disponible (PyTorch < 2.0)")

    def forward(
        self,
        x        : torch.Tensor,
        mask     : Optional[torch.Tensor] = None,
        past_kv  : Optional[KVCache]      = None,
        use_kv_cache: bool                = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Args:
            x           : [batch, seq_len, embed_dim]
            mask        : [seq_len, seq_len] bool (True = masqué) — fallback uniquement
            past_kv     : (k_cache, v_cache) depuis les steps précédents, ou None
            use_kv_cache: si True, retourne toujours le (k, v) mis à jour

        Returns:
            output   : [batch, seq_len, embed_dim]
            kv_cache : (k, v) mis à jour si use_kv_cache, sinon None
        """
        batch_size, seq_len, _ = x.shape

        # ── Projections ──────────────────────────────────────────
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ── Reshape multi-head ───────────────────────────────────
        q = q.view(batch_size, seq_len, self.num_heads,   self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads,  self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads,  self.head_dim).transpose(1, 2)

        # ── QK-Norm ──────────────────────────────────────────────
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # ── RoPE avec position_offset ─────────────────────────────
        # position_offset = longueur du cache existant
        # → les nouvelles positions démarrent après le cache
        position_offset = past_kv[0].shape[2] if past_kv is not None else 0
        if self.use_rope:
            q, k = self.rope(q, k, position_offset=position_offset)

        # ── KV Cache : concat ────────────────────────────────────
        # Le cache stocke les K/V non-répétés (n_kv_heads) pour économiser la RAM
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)   # [batch, n_kv_heads, total_len, head_dim]
            v = torch.cat([past_kv[1], v], dim=2)

        # Cache mis à jour à retourner (avant GQA repeat qui augmenterait la mémoire)
        new_kv_cache: Optional[KVCache] = (k, v) if use_kv_cache else None

        # ── GQA : répéter K et V ─────────────────────────────────
        if self.n_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # ── Scale factor ─────────────────────────────────────────
        if self.use_rope and self.rope is not None and self.rope.use_yarn and self.rope.yarn_scale > 1.0:
            attn_scale = math.sqrt(self.rope.yarn_scale) / math.sqrt(self.head_dim)
        else:
            attn_scale = 1.0 / math.sqrt(self.head_dim)

        # ── Attention ────────────────────────────────────────────
        # Deux chemins :
        #   1. Flash (SDPA) : soft_cap=None uniquement — rapide, fusionné kernel
        #   2. Manuel       : soft_cap actif OU use_flash_attn=False
        #      Note : PyTorch SDPA ne supporte pas le soft cap nativement.
        #      Avec torch.compile sur H100, le chemin manuel reste très performant.

        if self.use_flash_attn and self._flash_attn_available and self.soft_cap is None:
            # ── Chemin 1 : Flash Attention pure (soft_cap=None) ───
            if mask is not None:
                raise ValueError(
                    "mask custom incompatible avec Flash Attention. "
                    "Passez use_flash_attn=False ou retirez le mask."
                )
            is_causal = (seq_len > 1 and past_kv is None)
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = None,
                is_causal = is_causal,
                dropout_p = self.dropout.p if self.training else 0.0,
                scale     = attn_scale,
            )

        else:
            # ── Chemin 2 : Manuel — soft_cap ou fallback ──────────
            # Gère : soft_cap actif (Flash désactivée pour ce calcul),
            #        use_flash_attn=False, ou Flash non disponible.
            scores = torch.matmul(q, k.transpose(-2, -1)) * attn_scale

            # ✅ Soft Cap : borne les scores dans [-soft_cap, +soft_cap]
            # Empêche softmax trop piquée → meilleure stabilité training
            if self.soft_cap is not None:
                scores = self.soft_cap * torch.tanh(scores / self.soft_cap)

            # Masque causal :
            #   - soft_cap actif  → masque construit ici (Flash bypassé)
            #   - fallback pur    → masque fourni par HessGpt.forward via 'mask'
            #   - sécurité        → si ni l'un ni l'autre, on masque quand même
            if seq_len > 1 and past_kv is None:
                if mask is not None:
                    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                else:
                    # soft_cap actif ou fallback sans masque externe
                    total_len   = k.shape[2]
                    causal_bool = torch.triu(
                        torch.ones(seq_len, total_len, device=q.device, dtype=torch.bool),
                        diagonal=1
                    )
                    scores = scores.masked_fill(causal_bool.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            if self.training and self.dropout.p > 0:
                attn_weights = self.dropout(attn_weights)
            output = torch.matmul(attn_weights, v)

        # ── Reshape + projection ─────────────────────────────────
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output, new_kv_cache

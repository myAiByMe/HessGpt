# transformer_block.py - v7 — KV Cache
import torch
import torch.nn as nn
from typing import Optional, Tuple

from attention import MultiHeadAttention, RMSNorm, KVCache
from feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Transformer Block avec RMSNorm + RoPE + SwiGLU + GQA + Flash Attention + KV Cache.

    Le KV cache est transparent pour le block : il reçoit past_kv et retourne new_kv_cache,
    qu'il délègue entièrement à MultiHeadAttention.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1,
                 use_rope=True, max_seq_len=2048,
                 use_yarn=False, yarn_scale=1.0, yarn_original_max_len=1024,
                 use_swiglu=True, n_kv_heads=None, use_qk_norm=False,
                 use_flash_attn=True, soft_cap=None):
        super().__init__()

        self.embed_dim      = embed_dim
        self.num_heads      = num_heads
        self.use_rope       = use_rope
        self.use_swiglu     = use_swiglu
        self.n_kv_heads     = n_kv_heads
        self.use_qk_norm    = use_qk_norm
        self.use_flash_attn = use_flash_attn
        self.soft_cap       = soft_cap

        self.ln1 = RMSNorm(embed_dim)

        self.attention = MultiHeadAttention(
            embed_dim, num_heads, dropout,
            use_rope              = use_rope,
            max_seq_len           = max_seq_len,
            use_yarn              = use_yarn,
            yarn_scale            = yarn_scale,
            yarn_original_max_len = yarn_original_max_len,
            n_kv_heads            = n_kv_heads,
            use_qk_norm           = use_qk_norm,
            use_flash_attn        = use_flash_attn,
            soft_cap              = soft_cap,   # ✅ v8 : propagé à MultiHeadAttention
        )

        self.ln2 = RMSNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout, use_swiglu=use_swiglu)

    def forward(
        self,
        x            : torch.Tensor,
        mask         : Optional[torch.Tensor] = None,
        past_kv      : Optional[KVCache]      = None,
        use_kv_cache : bool                   = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Args:
            x            : [batch, seq_len, embed_dim]
            mask         : [seq_len, seq_len] bool — fallback uniquement
            past_kv      : cache KV de ce layer depuis les steps précédents
            use_kv_cache : si True, retourne le cache mis à jour

        Returns:
            output    : [batch, seq_len, embed_dim]
            new_kv    : cache KV mis à jour, ou None
        """
        # Attention block (pre-norm)
        residual = x
        x, new_kv = self.attention(
            self.ln1(x),
            mask         = mask,
            past_kv      = past_kv,
            use_kv_cache = use_kv_cache,
        )
        x = residual + x

        # FFN block (pre-norm)
        residual = x
        x        = self.ffn(self.ln2(x))
        x        = residual + x

        return x, new_kv
# HessGpt.py - v7 — KV Cache + top_p
"""
HessGPT - Architecture Transformer moderne v7

NOUVEAUTÉS v7 :
  ✅ KV Cache en inférence
      - Prefill  : premier appel avec le prompt complet, cache initialisé
      - Decode   : appels suivants avec 1 token, cache concaténé
      - Mémoire  : cache stocke K/V non-répétés (n_kv_heads) pour économiser la RAM
      - Format   : List[KVCache] de longueur num_layers, chaque entry = (k, v)
      - Training : past_kv=None partout → comportement identique à v6

  ✅ top_p (nucleus sampling)
      - Appliqué après top_k si les deux sont fournis
      - Trie les probs, cumule, masque tout ce qui dépasse p
      - Compatible temperature=0 (greedy, top_p ignoré)

BUGS FIXÉS v6 (conservés) :
  - Masque causal pré-alloué à l'init (compatible torch.compile)
  - Source de vérité flash : blocks[0].attention.use_flash_attn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple

from transformer_block import TransformerBlock
from attention import RMSNorm, KVCache


class HessGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim             = 768,
        num_heads             = 12,
        num_layers            = 12,
        max_seq_len           = 2048,
        dropout               = 0.1,
        use_rope              = True,
        use_yarn              = False,
        yarn_scale            = 1.0,
        yarn_original_max_len = 1024,
        use_swiglu            = True,
        n_kv_heads            = None,
        use_qk_norm           = False,
        soft_cap              = None,
        use_flash_attn        = True,
    ):
        super().__init__()

        # ── Validation ───────────────────────────────────────────
        assert vocab_size > 0,         "vocab_size must be positive"
        assert embed_dim > 0,          "embed_dim must be positive"
        assert num_layers > 0,         "num_layers must be positive"
        assert max_seq_len > 0,        "max_seq_len must be positive"
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        if n_kv_heads is not None:
            assert n_kv_heads > 0
            assert num_heads   % n_kv_heads == 0, \
                f"num_heads ({num_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            assert embed_dim % n_kv_heads == 0, \
                f"embed_dim ({embed_dim}) must be divisible by n_kv_heads ({n_kv_heads})"

        if use_rope and use_yarn:
            assert yarn_original_max_len > 0
            assert yarn_original_max_len <= max_seq_len, \
                f"yarn_original_max_len ({yarn_original_max_len}) must be <= max_seq_len ({max_seq_len})"
            assert 0.1 <= yarn_scale <= 16.0, \
                f"yarn_scale must be in [0.1, 16.0], got {yarn_scale}"

        if soft_cap is not None:
            assert 0 < soft_cap <= 100, \
                f"soft_cap must be in (0, 100], got {soft_cap}"

        if not use_yarn and yarn_scale != 1.0:
            print(f"⚠️  Warning: yarn_scale={yarn_scale} ignored (use_yarn=False)")

        # ── Attributs ────────────────────────────────────────────
        self.vocab_size            = vocab_size
        self.embed_dim             = embed_dim
        self.num_heads             = num_heads
        self.num_layers            = num_layers
        self.max_seq_len           = max_seq_len
        self.use_rope              = use_rope
        self.use_yarn              = use_yarn
        self.yarn_scale            = yarn_scale
        self.yarn_original_max_len = yarn_original_max_len
        self.use_swiglu            = use_swiglu
        self.n_kv_heads            = n_kv_heads
        self.use_qk_norm           = use_qk_norm
        self.soft_cap              = soft_cap
        self.use_flash_attn        = use_flash_attn

        # ── Embeddings ───────────────────────────────────────────
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)

        if not use_rope:
            self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
        else:
            self.position_embeddings = None

        self.dropout = nn.Dropout(dropout)

        # ── Transformer Blocks ───────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, dropout,
                use_rope              = use_rope,
                max_seq_len           = max_seq_len,
                use_yarn              = use_yarn,
                yarn_scale            = yarn_scale,
                yarn_original_max_len = yarn_original_max_len,
                use_swiglu            = use_swiglu,
                n_kv_heads            = n_kv_heads,
                use_qk_norm           = use_qk_norm,
                use_flash_attn        = use_flash_attn,
                soft_cap              = soft_cap,   # ✅ v8 : propagé à chaque layer
            )
            for _ in range(num_layers)
        ])

        # ── Final norm + head ────────────────────────────────────
        self.ln_final    = RMSNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # ── Masque causal pré-alloué (compile-safe) ──────────────
        # Slice [:seq_len, :seq_len] dans forward → pas de logique conditionnelle
        # que torch.compile pourrait figer.
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer('_causal_mask', causal_mask, persistent=False)

        # ── Init ─────────────────────────────────────────────────
        self.apply(self._init_weights)
        self.output_head.weight = self.token_embeddings.weight   # weight tying

    # ─────────────────────────────────────────────────────────────
    # Init
    # ─────────────────────────────────────────────────────────────
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    # ─────────────────────────────────────────────────────────────
    # Masque causal
    # ─────────────────────────────────────────────────────────────
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Retourne un slice du masque pré-alloué [seq_len, seq_len].
        Pas de branche conditionnelle → compile-safe.
        """
        return self._causal_mask[:seq_len, :seq_len]

    # ─────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids    : torch.Tensor,
        targets      : Optional[torch.Tensor]    = None,
        pad_token_id : Optional[int]             = None,
        past_kv      : Optional[List[KVCache]]   = None,
        use_kv_cache : bool                      = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[KVCache]]]:
        """
        Args:
            input_ids    : [batch, seq_len]
            targets      : [batch, seq_len] — optionnel, pour le calcul de la loss
            pad_token_id : token ignoré dans la loss
            past_kv      : List[KVCache] de longueur num_layers — None en training
            use_kv_cache : si True, retourne new_past_kv

        Returns:
            logits      : [batch, seq_len, vocab_size]
            loss        : scalar ou None
            new_past_kv : List[KVCache] mis à jour, ou None
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # ── Embeddings ───────────────────────────────────────────
        token_embeds = self.token_embeddings(input_ids)

        if self.use_rope:
            x = self.dropout(token_embeds)
        else:
            assert seq_len <= self.max_seq_len, \
                f"seq_len ({seq_len}) > max_seq_len ({self.max_seq_len})"
            positions  = torch.arange(0, seq_len, device=device).unsqueeze(0)
            pos_embeds = self.position_embeddings(positions)
            x          = self.dropout(token_embeds + pos_embeds)

        # ── Masque causal ────────────────────────────────────────
        # Source de vérité = premier block (évite désync runtime)
        # En decode avec KV cache (seq_len=1) : pas de masque nécessaire
        mask = None
        if not self.blocks[0].attention.use_flash_attn:
            if seq_len > 1:
                mask = self._get_causal_mask(seq_len, device)

        # ── Transformer Blocks ───────────────────────────────────
        new_past_kv: Optional[List[KVCache]] = [] if use_kv_cache else None

        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            x, new_kv  = block(
                x,
                mask         = mask,
                past_kv      = layer_past,
                use_kv_cache = use_kv_cache,
            )
            if use_kv_cache:
                new_past_kv.append(new_kv)

        # ── Final norm + logits ──────────────────────────────────
        x      = self.ln_final(x)
        logits = self.output_head(x)

        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap

        # ── Loss ─────────────────────────────────────────────────
        loss = None
        if targets is not None:
            ignore_index = pad_token_id if pad_token_id is not None else -100
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=ignore_index,
            )

        return logits, loss, new_past_kv

    # ─────────────────────────────────────────────────────────────
    # Génération autoregressive — KV Cache + top_k + top_p
    # ─────────────────────────────────────────────────────────────
    def generate(
        self,
        input_ids    : torch.Tensor,
        max_new_tokens: int   = 50,
        temperature  : float  = 1.0,
        top_k        : Optional[int]   = None,
        top_p        : Optional[float] = None,
        eos_token_id : Optional[int]   = None,
    ) -> torch.Tensor:
        """
        Génération autoregressive avec KV Cache, top_k et top_p.

        Stratégie KV Cache :
          1. Prefill  : forward sur le prompt complet → cache initialisé
          2. Decode   : forward sur 1 token à la fois → cache concaténé à chaque step

        Args:
            input_ids      : [batch, prompt_len]
            max_new_tokens : nombre maximum de tokens à générer
            temperature    : température (0.0 = greedy)
            top_k          : garde les k tokens les plus probables
            top_p          : nucleus sampling — garde le noyau de proba cumulée >= p
            eos_token_id   : stop dès que ce token est généré (optionnel)

        Returns:
            input_ids : [batch, prompt_len + nb_tokens_générés]
        """
        was_training = self.training
        self.eval()
        device = input_ids.device

        with torch.no_grad():

            # ── Tronquer si prompt dépasse max_seq_len ───────────
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]

            # ── PREFILL : forward sur le prompt complet ──────────
            # past_kv=None, use_kv_cache=True → initialise le cache
            # Les logits[:, -1, :] donnent directement la distribution
            # du PREMIER token à générer → on ne re-traite pas le
            # dernier token du prompt dans la boucle de decode.
            prefill_logits, _, past_kv = self.forward(
                input_ids,
                use_kv_cache=True,
            )
            # past_kv : List[KVCache], chaque entry = (k, v)
            #   shape : [batch, n_kv_heads, prompt_len, head_dim]

            # Logits du premier token à générer (depuis le prefill)
            next_logits = prefill_logits[:, -1, :]   # [batch, vocab_size]

            for step in range(max_new_tokens):

                logits = next_logits   # [batch, vocab_size]

                # ── Greedy ───────────────────────────────────────
                if temperature == 0.0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    # ── Temperature ──────────────────────────────
                    logits = logits / temperature

                    # ── top_k ────────────────────────────────────
                    if top_k is not None:
                        k         = min(top_k, logits.size(-1))
                        topk_v, _ = torch.topk(logits, k)
                        logits    = logits.masked_fill(logits < topk_v[:, [-1]], float('-inf'))

                    # ── top_p (nucleus sampling) ──────────────────
                    # Appliqué après top_k si les deux sont fournis.
                    # Algorithme :
                    #   1. Trier les logits par ordre décroissant
                    #   2. Convertir en probs (softmax)
                    #   3. Calculer la somme cumulée
                    #   4. Masquer tout ce qui fait dépasser le seuil p
                    #   5. Remettre dans l'ordre original
                    if top_p is not None and top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                        sorted_probs              = F.softmax(sorted_logits, dim=-1)
                        cumulative_probs          = torch.cumsum(sorted_probs, dim=-1)

                        # Décaler d'un cran pour inclure le premier token qui dépasse p
                        # Ex: probs=[0.6, 0.3, 0.1], p=0.8
                        #   cumsum = [0.6, 0.9, 1.0]
                        #   shifted = [0.0, 0.6, 0.9]
                        #   mask    = [F,   F,   T]   → garde les 2 premiers ✅
                        remove_mask   = (cumulative_probs - sorted_probs) >= top_p
                        sorted_logits = sorted_logits.masked_fill(remove_mask, float('-inf'))

                        # Remettre dans l'ordre original des tokens
                        logits = torch.zeros_like(logits).scatter_(
                            dim=1, index=sorted_idx, src=sorted_logits
                        )

                    # ── Sampling ─────────────────────────────────
                    probs      = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)   # [batch, 1]

                input_ids = torch.cat([input_ids, next_token], dim=1)

                # ── Stop sur EOS ─────────────────────────────────
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

                # ── DECODE : forward sur le token généré ─────────
                # Met à jour next_logits ET past_kv pour le step suivant.
                # C'est ici que le KV cache est utilisé : on ne passe que
                # 1 token, le cache contient tout le contexte précédent.
                decode_logits, _, past_kv = self.forward(
                    next_token,
                    past_kv      = past_kv,
                    use_kv_cache = True,
                )
                next_logits = decode_logits[:, -1, :]   # [batch, vocab_size]

        if was_training:
            self.train()

        return input_ids

    # ─────────────────────────────────────────────────────────────
    # Utilitaires
    # ─────────────────────────────────────────────────────────────
    def resize_token_embeddings(self, new_vocab_size: int):
        if new_vocab_size == self.vocab_size:
            return
        print(f"📝 Resizing embeddings: {self.vocab_size} → {new_vocab_size}")

        old_emb = self.token_embeddings
        self.token_embeddings = nn.Embedding(new_vocab_size, self.embed_dim)
        n = min(old_emb.num_embeddings, new_vocab_size)
        with torch.no_grad():
            self.token_embeddings.weight.data[:n] = old_emb.weight.data[:n]

        self.output_head        = nn.Linear(self.embed_dim, new_vocab_size, bias=False)
        self.output_head.weight = self.token_embeddings.weight
        self.vocab_size         = new_vocab_size
        print(f"   ✅ Embeddings resized to {new_vocab_size}")

    def count_parameters(self) -> dict:
        token_params = self.token_embeddings.weight.numel()
        pos_params   = (self.position_embeddings.weight.numel()
                        if self.position_embeddings is not None else 0)
        block_params = sum(p.numel() for b in self.blocks for p in b.parameters())
        ln_params    = sum(p.numel() for p in self.ln_final.parameters())
        total        = token_params + pos_params + block_params + ln_params
        return {
            'token_embeddings':    token_params,
            'position_embeddings': pos_params,
            'transformer_blocks':  block_params,
            'final_ln':            ln_params,
            'output_head':         0,
            'total':               total,
        }

    def get_config(self) -> dict:
        return {
            'vocab_size':            self.vocab_size,
            'embed_dim':             self.embed_dim,
            'num_heads':             self.num_heads,
            'num_layers':            self.num_layers,
            'max_seq_len':           self.max_seq_len,
            'use_rope':              self.use_rope,
            'use_yarn':              self.use_yarn,
            'yarn_scale':            self.yarn_scale,
            'yarn_original_max_len': self.yarn_original_max_len,
            'use_swiglu':            self.use_swiglu,
            'n_kv_heads':            self.n_kv_heads,
            'use_qk_norm':           self.use_qk_norm,
            'soft_cap':              self.soft_cap,
            'use_flash_attn':        self.use_flash_attn,
        }
from typing import Optional, Tuple
import math
import warnings

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.mistral.modeling_mistral import MistralAttention, repeat_kv, apply_rotary_pos_emb
from transformers.models.mixtral.modeling_mixtral import MixtralAttention


def _mistral_shape(self):
    num_heads = getattr(self, "num_heads", getattr(self, "num_attention_heads", self.config.num_attention_heads))
    num_key_value_heads = getattr(self, "num_key_value_heads", self.config.num_key_value_heads)
    num_key_value_groups = getattr(self, "num_key_value_groups", num_heads // num_key_value_heads)
    hidden_size = getattr(self, "hidden_size", self.config.hidden_size)
    return num_heads, num_key_value_heads, num_key_value_groups, hidden_size


def _rotary_emb(self, value_states, kv_seq_len, position_ids, kwargs):
    position_embeddings = kwargs.get("position_embeddings")
    if position_embeddings is not None:
        return position_embeddings
    try:
        return self.rotary_emb(value_states, seq_len=kv_seq_len)
    except TypeError:
        return self.rotary_emb(value_states, position_ids)


def _apply_rotary(query_states, key_states, cos, sin, position_ids):
    try:
        return apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    except TypeError:
        return apply_rotary_pos_emb(query_states, key_states, cos, sin)


def _mask_value(attn_weights, attention_mask):
    if attention_mask is None or attention_mask.dtype == torch.bool:
        return torch.finfo(attn_weights.dtype).min
    return torch.min(attention_mask)


def get_energy_forward(args):
    def modified_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()
        num_heads, num_key_value_heads, num_key_value_groups, hidden_size = _mistral_shape(self)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = _rotary_emb(self, value_states, kv_seq_len, position_ids, kwargs)
        query_states, key_states = _apply_rotary(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        L = attn_weights.shape[-1]
        keep_budget = int(args.keep_ratio * L)
        recent_ratio = args.recent_ratio if args.recent_ratio != -1 else args.keep_ratio
        recent_budget = int(recent_ratio * L)

        q = query_states.float()
        Rq = torch.einsum("bhtd,bhte->bhde", q, q) / q.shape[2]
        w, U = torch.linalg.eigh(Rq)
        w = w.clamp_min(w.amax(-1, keepdim=True) * 1e-6)
        Rq_half = (U * w.sqrt().unsqueeze(-2)) @ U.transpose(-1, -2)

        k = key_states.float()
        kt = torch.einsum("bhld,bhde->bhle", k, Rq_half)
        energy = (kt * kt).sum(-1)

        neg = torch.finfo(attn_weights.dtype).min
        score = energy[:, :, None, :].expand(-1, -1, q_len, -1).clone()
        causal = torch.tril(
            torch.ones(q_len, L, dtype=torch.bool, device=k.device),
            diagonal=L - q_len,
        )
        score = score.masked_fill(~causal, neg)

        if 0 < keep_budget < L:
            idx = score.topk(keep_budget, dim=-1).indices
            keep = torch.zeros_like(attn_weights, dtype=torch.bool).scatter_(-1, idx, True)
        else:
            keep = torch.ones_like(attn_weights, dtype=torch.bool)

        recent = torch.triu(torch.ones_like(attn_weights, dtype=torch.bool), diagonal=-recent_budget)
        keep = torch.logical_or(keep, recent)
        keep = torch.tril(keep, diagonal=0)

        attn_weights[~keep] = _mask_value(attn_weights, attention_mask)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return modified_forward


def make_mistral_attention_energy(args):
    recent_ratio = args.recent_ratio if args.recent_ratio != -1 else args.keep_ratio
    print("Modifying Mistral/Mixtral Attention -> Energy evictor")
    print(f"keep_ratio={args.keep_ratio}  recent_ratio={recent_ratio}")
    MistralAttention.forward = get_energy_forward(args)
    MixtralAttention.forward = get_energy_forward(args)

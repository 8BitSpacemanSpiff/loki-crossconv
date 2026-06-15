from typing import Optional, Tuple
import math
import warnings

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.mistral.modeling_mistral import MistralAttention, repeat_kv, apply_rotary_pos_emb

from .selectors import build_keep_mask


_EVICT = {"masks": {}}


def reset_evict():
    _EVICT["masks"].clear()


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


def get_eviction_forward(args):
    def modified_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

        Lk = attn_weights.shape[-1]
        is_prefill = q_len > 1

        if is_prefill:
            with torch.no_grad():
                probs = torch.softmax(attn_weights.float(), dim=-1)
                W = min(args.obs_window, q_len)
                if bsz != 1:
                    raise ValueError("eviction_longctx_eval currently expects batch size 1")
                A_win = probs[0, :, -W:, :].mean(1)
                A_acc = probs[0].sum(1)
                keep = build_keep_mask(A_win, A_acc, value_states[0], args, num_key_value_groups)
            _EVICT["masks"][self.layer_idx] = keep
        else:
            keep = _EVICT["masks"][self.layer_idx]
            if keep.shape[1] < Lk:
                pad = torch.ones(
                    keep.shape[0],
                    Lk - keep.shape[1],
                    dtype=torch.bool,
                    device=keep.device,
                )
                keep = torch.cat([keep, pad], dim=1)
            m = keep[None, :, None, :]
            attn_weights = attn_weights.masked_fill(~m, torch.finfo(attn_weights.dtype).min)

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

        return attn_output, attn_weights

    return modified_forward


def make_mistral_attention_eviction(args):
    print(
        f"Eviction: method={args.evict_method} budget={args.evict_budget} "
        f"sinks={args.sink_tokens} recent={args.recent_tokens} obs_window={args.obs_window}"
    )
    MistralAttention.forward = get_eviction_forward(args)

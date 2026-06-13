
from typing import List, Optional, Tuple, Union
import math
import warnings
from transformers.models.mistral.modeling_mistral import MistralAttention, repeat_kv, apply_rotary_pos_emb, MistralRotaryEmbedding
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralAttention
from transformers.cache_utils import Cache
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from .utils import mask_attn_pca_topk, get_pca_components
from .crosscov_utils import mask_attn_crosscov, get_crosscov_components
from .crosscov_ext import (
    mask_attn_crosscov_compress,
    mask_attn_crosscov_evict,
    mask_attn_keydiff,
    mask_attn_keynorm,
    get_rq_gram,
)
import methods

try:
    from axonn import axonn as ax
    from axonn.intra_layer import drop
    AXONN_AVAILABLE=True
except ImportError:
    AXONN_AVAILABLE=False

def log_attention_output_error(args, layer_idx, dense_output, sparse_output):
    diff = (sparse_output.float() - dense_output.float()).reshape(dense_output.shape[0], -1)
    ref = dense_output.float().reshape(dense_output.shape[0], -1)
    rel_l2 = (torch.linalg.vector_norm(diff, dim=-1) / torch.linalg.vector_norm(ref, dim=-1).clamp_min(1e-12)).mean().item()
    cosine = F.cosine_similarity(sparse_output.float().reshape(dense_output.shape[0], -1), ref, dim=-1).mean().item()
    if getattr(args, "quiet_diagnostics", False):
        methods.record_diagnostic("attn_out_rel_l2", layer_idx, rel_l2)
        methods.record_diagnostic("attn_out_cos", layer_idx, cosine)
    else:
        print(f"LayerId:{layer_idx}|AttnOutRelL2:{rel_l2:.6f}|AttnOutCos:{cosine:.6f}")
    if methods.LOGGER is not None:
        methods.LOGGER.log({
            f"attn_out_rel_l2_layer_{layer_idx}": rel_l2,
            f"attn_out_cos_layer_{layer_idx}": cosine,
        })

def get_pca_forward(args):
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

        use_crosscov = getattr(args, "use_crosscov", False)
        crosscov_mode = getattr(args, "crosscov_mode", "select")
        needs_basis = crosscov_mode in ("select", "compress", "evict")
        if use_crosscov:
            if needs_basis and not hasattr(self, "cc_key_r"):
                (self.cc_key_full, self.cc_key_r,
                 self.cc_query_full, self.cc_query_r) = get_crosscov_components(
                    args, self.layer_idx, self.head_dim, args.top_r, self.num_key_value_groups, repeat_kv)
            if crosscov_mode == "evict" and not hasattr(self, "cc_rq_gram"):
                self.cc_rq_gram = get_rq_gram(
                    args, self.layer_idx, self.head_dim, args.top_r,
                    self.num_key_value_groups, repeat_kv, self.cc_key_r)
        else:
            if not hasattr(self, "pca_components"):
                self.pca_means, self.pca_components, self.pca_components_r_key = get_pca_components(args, self.layer_idx, self.head_dim, args.top_r, self.num_key_value_groups, repeat_kv)
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if args.top_k <= 1:
            topk = int(args.top_k * key_states.shape[-2])
        else:
            topk = int(args.top_k)

        dense_attn_output = None
        if getattr(args, "log_output_error", False):
            dense_attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                dense_attn_weights = dense_attn_weights + attention_mask
            dense_attn_weights = nn.functional.softmax(dense_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            dense_attn_output = torch.matmul(dense_attn_weights, value_states)

        if use_crosscov:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
            if crosscov_mode == "select":
                self.cc_key_r = self.cc_key_r.to(key_states.dtype)
                self.cc_query_r = self.cc_query_r.to(key_states.dtype)
                attn_weights, alpha = mask_attn_crosscov(
                    args, self.layer_idx, attn_weights, attention_mask, query_states, key_states,
                    self.cc_key_r, self.cc_query_r, args.top_r, topk)
            elif crosscov_mode == "compress":
                self.cc_key_r = self.cc_key_r.to(key_states.dtype)
                self.cc_query_r = self.cc_query_r.to(key_states.dtype)
                attn_weights, alpha = mask_attn_crosscov_compress(
                    args, self.layer_idx, attn_weights, attention_mask, query_states, key_states,
                    self.cc_key_r, self.cc_query_r, args.top_r, topk)
            elif crosscov_mode == "evict":
                self.cc_key_r = self.cc_key_r.to(key_states.dtype)
                self.cc_rq_gram = self.cc_rq_gram.to(key_states.dtype)
                attn_weights, alpha = mask_attn_crosscov_evict(
                    args, self.layer_idx, attn_weights, attention_mask, query_states, key_states,
                    self.cc_key_r, self.cc_rq_gram, args.top_r,
                    getattr(args, "evict_ratio", 0.5), getattr(args, "sink_tokens", 16),
                    getattr(args, "recent_window", 64))
            elif crosscov_mode == "keydiff":
                attn_weights, alpha = mask_attn_keydiff(
                    args, self.layer_idx, attn_weights, attention_mask, query_states, key_states,
                    getattr(args, "evict_ratio", 0.5), getattr(args, "sink_tokens", 16),
                    getattr(args, "recent_window", 64))
            elif crosscov_mode == "keynorm":
                attn_weights, alpha = mask_attn_keynorm(
                    args, self.layer_idx, attn_weights, attention_mask, query_states, key_states,
                    getattr(args, "evict_ratio", 0.5), getattr(args, "sink_tokens", 16),
                    getattr(args, "recent_window", 64))
            else:
                raise ValueError(f"unknown crosscov_mode: {crosscov_mode}")
        else:
            self.pca_means = self.pca_means.to(key_states.dtype)
            self.pca_components_r_key = self.pca_components_r_key.to(key_states.dtype)
            self.pca_components = self.pca_components.to(key_states.dtype)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )

                attn_weights = attn_weights + attention_mask

            attn_weights, alpha = mask_attn_pca_topk(args, self.layer_idx, attn_weights, attention_mask, query_states, key_states, self.pca_components, self.pca_components_r_key, args.top_r, topk)

        assert alpha is not None, "alpha is None"

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if dense_attn_output is not None:
            log_attention_output_error(args, self.layer_idx, dense_attn_output, attn_output)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    return modified_forward

def make_mistral_attention_pca_topk(args):
    print ("Modifying Mistral & Mixtral Attention -> PCA Attention")
    print ("Top R:", args.top_r)
    print ("Top K:", args.top_k)
    MistralAttention.forward = get_pca_forward(args)
    MixtralAttention.forward = get_pca_forward(args)

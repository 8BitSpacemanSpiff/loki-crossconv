from typing import List, Optional, Tuple, Union
import math
import warnings
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv, apply_rotary_pos_emb
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, ACT2FN
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

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if args.top_k <= 1:
            topk = int(args.top_k * key_states.shape[-2])
        else:
            topk = int(args.top_k)

        if use_crosscov:
            # Exact reference scores from RAW q.k -- the asymmetric projection is not a
            # rotation, so the final attention and the recall ground truth use these.
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
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

            # Compute attention after projecting the key_states and query_states to PCA space
            key_states_pca  = torch.matmul(key_states, self.pca_components)
            query_states_pca = torch.matmul(query_states, self.pca_components)
            attn_weights = (torch.matmul(query_states_pca, key_states_pca.transpose(2, 3))) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights, alpha = mask_attn_pca_topk(args, self.layer_idx, attn_weights, attention_mask, query_states, key_states, self.pca_components, self.pca_components_r_key, args.top_r, topk)


        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    return modified_forward

def make_llama_attention_pca_topk(args):
    print ("Modifying Llama Attention -> PCA TopK Attention")
    print ("Top R:", args.top_r)
    print ("Top K:", args.top_k)
    LlamaAttention.forward = get_pca_forward(args)

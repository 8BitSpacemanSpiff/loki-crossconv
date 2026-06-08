import math
import os

import torch

import methods
from .utils import PCA_DATA_PATH

try:
    from axonn import axonn as ax
    from axonn.intra_layer import drop
    AXONN_AVAILABLE = True
except ImportError:
    AXONN_AVAILABLE = False


def _load_side(side, args, layer_idx, head_dim, top_r, num_key_value_groups, repeat_kv, device):
    # side in {"key", "query"}. Mirrors get_pca_components but for one side; reads the
    # CrossCov-SVD bases written by pca_analysis/crosscov.py into <...>/{rotary_type}/{side}.
    model_folder_name = args.model_id.split("/")[-1] + "-PCA"
    base = os.path.join(PCA_DATA_PATH, f"{model_folder_name}/{args.transform_dataset}/{args.rotary_type}/{side}")
    comps = torch.load(f"{base}/pca_components/pca_components_layer_{layer_idx}.pt").to(device)
    expl = torch.load(f"{base}/pca_explained_variance/pca_explained_variance_layer_{layer_idx}.pt").to(device)

    # (num_heads, head_dim, head_dim) rows->cols so x @ comps projects; matches Loki loader.
    comps = comps.reshape(1, -1, head_dim, head_dim).transpose(2, 3)

    if top_r < 1:
        top_correct_r = (expl.cumsum(-1) < top_r).sum(-1).max().item()
    else:
        top_correct_r = int(top_r)
    comps_r = comps[:, :, :, :top_correct_r]

    if repeat_kv is not None:
        comps = repeat_kv(comps, num_key_value_groups)
        comps_r = repeat_kv(comps_r, num_key_value_groups)
    if AXONN_AVAILABLE and ax.is_initialized:
        comps = drop(comps, transpose=True, skip_batch=True, dim=1)
        comps_r = drop(comps_r, transpose=True, skip_batch=True, dim=1)
    return comps, comps_r, top_correct_r


def get_crosscov_components(args, layer_idx, head_dim, top_r, num_key_value_groups, repeat_kv, device=None):
    if device is None:
        device = "cuda"
    key_full, key_r, r = _load_side("key", args, layer_idx, head_dim, top_r, num_key_value_groups, repeat_kv, device)
    query_full, query_r, _ = _load_side("query", args, layer_idx, head_dim, top_r, num_key_value_groups, repeat_kv, device)
    print(f"{layer_idx}: CrossCov key_r {tuple(key_r.shape)} query_r {tuple(query_r.shape)}  ratio {r/head_dim:.3f}")
    if methods.LOGGER is not None:
        methods.LOGGER.log({"compression_ratio": r / head_dim})
    return key_full, key_r, query_full, query_r


def mask_attn_crosscov(args, layer_idx, attn_weights, attention_mask, query_states,
                       key_states, comps_key_r, comps_query_r, top_r, top_k):
    # attn_weights here MUST be the EXACT raw-q.k scores (the asymmetric projection is
    # not a rotation, so projected full-rank scores are not exact). Selection happens in
    # the reduced space; the gathered values use the exact scores.
    head_dim = key_states.shape[-1]
    key_sparse = torch.matmul(key_states, comps_key_r).to(query_states.dtype)
    query_sparse = torch.matmul(query_states, comps_query_r).to(query_states.dtype)

    s_hat = torch.matmul(query_sparse, key_sparse.transpose(-1, -2)) / math.sqrt(head_dim)
    if attention_mask is not None:
        s_hat = s_hat + attention_mask[:, :, :, : key_states.shape[-2]]

    if top_k >= key_states.shape[-2]:
        top_k = key_states.shape[-2]
    i2 = torch.topk(s_hat, top_k, dim=-1).indices

    if getattr(args, "log_recall", False):
        i2_ground = torch.topk(attn_weights, top_k, dim=-1).indices
        zeros = torch.zeros_like(attn_weights)
        pred = torch.tril(zeros.scatter(-1, i2, 1))
        grnd = torch.tril(zeros.scatter(-1, i2_ground, 1))
        inter = torch.logical_and(pred, grnd).sum(-1).float()
        recall = (inter / top_k)[:, :, top_k:].mean().item()
        print(f"LayerId:{layer_idx}|Recall@{top_k}:{recall:.4f}")
        if methods.LOGGER is not None:
            methods.LOGGER.log({f"recall_layer_{layer_idx}": recall})

    mask = torch.full_like(attn_weights, fill_value=float("-inf"))
    mask.scatter_(-1, i2, attn_weights.gather(-1, i2))
    alpha = torch.sum(torch.gather(torch.softmax(s_hat, dim=-1), -1, i2), -1, True)
    return mask, alpha

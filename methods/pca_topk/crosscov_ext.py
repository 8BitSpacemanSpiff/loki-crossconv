"""
CrossCov runtime extensions: KV-cache compression and KV eviction.

Both reuse the CrossCov-U key basis already produced by pca_analysis/crosscov.py
and loaded by crosscov_utils.get_crosscov_components. They are selected as
sub-modes of the existing --use-crosscov path:

  --crosscov-mode select    (default) : current behaviour, full cache + top-k select
  --crosscov-mode compress            : SALS-style rank-r cache, reconstruct selected
  --crosscov-mode evict               : static query-weighted score-energy eviction

Design notes
------------
* compress(): the cache is modelled as the rank-r projection of the keys,
  K_hat = K @ U_r @ U_r^T. Selection happens in latent space (cheap r-dim score),
  but unlike select-mode the gathered attention scores are recomputed from the
  RECONSTRUCTED keys, not the exact keys -- this is the faithful accuracy cost of
  only storing the rank-r cache. Memory ratio r/head_dim is logged.

* evict(): each token gets a query-AGNOSTIC importance s_j = k_j^T Rq k_j, the
  expected query energy of its score (Rq = E[q q^T], pooled over the GQA group).
  Computed in latent coords as k~_j^T G k~_j with G = U_r^T Rq U_r. A static
  retained set keeps `sink` initial + `recent` final tokens and the highest-score
  middle tokens up to the budget; evicted columns are masked for ALL query rows
  (irreversible), subject to the causal mask.

The math behind both is unit-checked in tests/test_crosscov_ext.py (numpy).
"""
import math
import os

import torch
import torch.nn.functional as F

import methods
from .utils import PCA_DATA_PATH


def _rq_gram_path(side, args, layer_idx):
    model_folder_name = args.model_id.split("/")[-1] + "-PCA"
    return os.path.join(
        PCA_DATA_PATH,
        f"{model_folder_name}/{args.transform_dataset}/{args.rotary_type}/{side}/rq_gram/rq_gram_layer_{layer_idx}.pt",
    )


def get_rq_gram(args, layer_idx, head_dim, top_r, num_key_value_groups, repeat_kv, comps_key_r, device=None):
    """Return the per-head latent query gram G_r = U_r^T Rq U_r, shape (1, H, r, r).

    If a precomputed Rq (head_dim x head_dim, pooled over the GQA group) was emitted
    offline (crosscov.py --emit-rq), project it into the loaded latent basis. Falls
    back to identity (pure key-norm eviction) if absent, so the path degrades to a
    query-agnostic geometric scorer rather than failing.
    """
    if device is None:
        device = "cuda"
    path = _rq_gram_path("key", args, layer_idx)
    r = comps_key_r.shape[-1]
    if not os.path.exists(path):
        if not getattr(args, "quiet_diagnostics", False):
            print(f"[crosscov-evict] no Rq at layer {layer_idx}; using identity gram (key-norm eviction)")
        H = comps_key_r.shape[1]
        return torch.eye(r, device=device).reshape(1, 1, r, r).expand(1, H, r, r).contiguous()
    Rq = torch.load(path).to(device)                       # (num_kv_heads, head_dim, head_dim), pooled per group
    Rq = Rq.reshape(1, -1, head_dim, head_dim)
    if repeat_kv is not None:
        Rq = repeat_kv(Rq, num_key_value_groups)           # (1, H, d, d) aligned to comps_key_r heads
    # G_r = U_r^T Rq U_r ; comps_key_r columns are the latent basis (x @ comps -> latent)
    U_r = comps_key_r                                      # (1, H, d, r)
    G = torch.matmul(U_r.transpose(-1, -2), torch.matmul(Rq, U_r))  # (1, H, r, r)
    return G.to(comps_key_r.dtype)


def mask_attn_crosscov_compress(args, layer_idx, attn_weights, attention_mask, query_states,
                                key_states, comps_key_r, comps_query_r, top_r, top_k):
    """SALS-style: select in latent space, score selected tokens from reconstructed keys."""
    head_dim = key_states.shape[-1]
    key_lat = torch.matmul(key_states, comps_key_r).to(query_states.dtype)     # (.., S, r)
    query_lat = torch.matmul(query_states, comps_query_r).to(query_states.dtype)

    s_hat = torch.matmul(query_lat, key_lat.transpose(-1, -2)) / math.sqrt(head_dim)
    if attention_mask is not None:
        s_hat = s_hat + attention_mask[:, :, :, : key_states.shape[-2]]

    if top_k >= key_states.shape[-2]:
        top_k = key_states.shape[-2]
    i2 = torch.topk(s_hat, top_k, dim=-1).indices

    # Reconstructed-key scores: q . (P k) with P = U_r U_r^T. Equivalent to scoring
    # the rank-r cache after reconstruction, which is what a compressed cache yields.
    key_recon = torch.matmul(key_lat, comps_key_r.transpose(-1, -2)).to(query_states.dtype)  # (.., S, d)
    recon_weights = torch.matmul(query_states, key_recon.transpose(-1, -2)) / math.sqrt(head_dim)
    if attention_mask is not None:
        recon_weights = recon_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    if getattr(args, "log_recall", False):
        i2_ground = torch.topk(attn_weights, top_k, dim=-1).indices
        zeros = torch.zeros_like(attn_weights)
        pred = torch.tril(zeros.scatter(-1, i2, 1))
        grnd = torch.tril(zeros.scatter(-1, i2_ground, 1))
        inter = torch.logical_and(pred, grnd).sum(-1).float()
        recall = (inter / top_k)[:, :, top_k:].mean().item()
        _log(args, layer_idx, "recall", recall, f"Recall@{top_k}")

    if getattr(args, "log_mass_recall", False):
        true_probs = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
        mass = torch.gather(true_probs, -1, i2).sum(-1)[:, :, top_k:].mean().item()
        _log(args, layer_idx, "mass_recall", mass, f"MassRecall@{top_k}")

    if methods.LOGGER is not None:
        methods.LOGGER.log({"kv_compress_ratio": comps_key_r.shape[-1] / head_dim})

    mask = torch.full_like(attn_weights, fill_value=float("-inf"))
    mask.scatter_(-1, i2, recon_weights.gather(-1, i2))   # gathered scores use reconstructed keys
    alpha = torch.sum(torch.gather(torch.softmax(s_hat, dim=-1), -1, i2), -1, True)
    return mask, alpha


def _static_evict_mask(args, layer_idx, attn_weights, score, evict_ratio,
                       sink_tokens, recent_window, mass_label):
    """Shared static-eviction protocol. `score` is per-token importance (b, H, S);
    higher = keep. All eviction-family methods (crosscov / keydiff / keynorm) call
    this with the SAME protection + budget + diagnostics so the comparison is fair.

    Budget: --keep-tokens N (absolute, matches KeyDiff's budget protocol) overrides
    --evict-ratio when set > 0.
    """
    bsz, n_heads, q_len, S = attn_weights.shape
    keep_tokens = getattr(args, "keep_tokens", 0)
    if keep_tokens and keep_tokens > 0:
        keep = min(int(keep_tokens), S)
    else:
        keep = max(1, int(round((1.0 - evict_ratio) * S)))

    protect = torch.zeros_like(score)
    if sink_tokens > 0:
        protect[..., :min(sink_tokens, S)] = float("inf")
    if recent_window > 0:
        protect[..., max(0, S - recent_window):] = float("inf")
    score_protected = torch.where(torch.isinf(protect), protect, score)

    keep_idx = torch.topk(score_protected, min(keep, S), dim=-1).indices
    retained = torch.zeros_like(score, dtype=torch.bool)
    retained.scatter_(-1, keep_idx, True)                                      # (b, H, S)

    if getattr(args, "log_mass_recall", False):
        true_probs = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)  # (b,H,q,S)
        retained_q = retained.unsqueeze(2).expand(bsz, n_heads, q_len, S)
        mass = (true_probs * retained_q).sum(-1).mean().item()
        _log(args, layer_idx, mass_label, mass, mass_label)
    if methods.LOGGER is not None:
        methods.LOGGER.log({f"evict_ratio_actual_layer_{layer_idx}": 1.0 - keep / S})

    col_mask = retained.unsqueeze(2)                                           # (b, H, 1, S)
    mask = torch.where(col_mask, attn_weights, torch.full_like(attn_weights, float("-inf")))
    alpha = torch.sum(torch.softmax(mask, dim=-1), -1, True)
    return mask, alpha


def mask_attn_crosscov_evict(args, layer_idx, attn_weights, attention_mask, query_states,
                             key_states, comps_key_r, rq_gram, top_r, evict_ratio,
                             sink_tokens, recent_window):
    """OURS: static query-weighted score-energy eviction, score_j = k~_j^T G k~_j."""
    key_lat = torch.matmul(key_states, comps_key_r).to(query_states.dtype)     # (b, H, S, r)
    Gk = torch.matmul(key_lat, rq_gram)                                        # (b, H, S, r)
    score = torch.sum(Gk * key_lat, dim=-1)                                    # (b, H, S)
    return _static_evict_mask(args, layer_idx, attn_weights, score, evict_ratio,
                              sink_tokens, recent_window, "evict_mass_kept")


def mask_attn_keydiff(args, layer_idx, attn_weights, attention_mask, query_states,
                      key_states, evict_ratio, sink_tokens, recent_window):
    """BASELINE (KeyDiff, arXiv 2504.15364): evict keys most similar to the anchor
    = mean of L2-normalized keys. Keep geometrically distinctive (low-similarity)
    keys. Query-agnostic, no calibration. Static (anchor over the full sequence) to
    match the single-forward harness; the streaming anchor is a separate efficiency axis.
    """
    kn = F.normalize(key_states.float(), dim=-1)                               # (b, H, S, d)
    anchor = kn.mean(dim=-2, keepdim=True)                                     # (b, H, 1, d)
    cos = (kn * anchor).sum(-1)                                                # (b, H, S), unnorm. anchor ok for ranking
    score = -cos                                                              # higher = more distinctive = keep
    return _static_evict_mask(args, layer_idx, attn_weights, score, evict_ratio,
                              sink_tokens, recent_window, "keydiff_mass_kept")


def mask_attn_keynorm(args, layer_idx, attn_weights, attention_mask, query_states,
                      key_states, evict_ratio, sink_tokens, recent_window):
    """CONTROL: score_j = ||k_j||^2, the isotropic (Rq = I) limit of crosscov-evict.
    Separates 'query distribution matters' (crosscov beats this) from 'magnitude matters'.
    """
    score = (key_states.float() ** 2).sum(-1)                                  # (b, H, S)
    return _static_evict_mask(args, layer_idx, attn_weights, score, evict_ratio,
                              sink_tokens, recent_window, "keynorm_mass_kept")
def _log(args, layer_idx, key, value, label):
    if getattr(args, "quiet_diagnostics", False):
        methods.record_diagnostic(key, layer_idx, value)
    else:
        print(f"LayerId:{layer_idx}|{label}:{value:.4f}")
    if methods.LOGGER is not None:
        methods.LOGGER.log({f"{key}_layer_{layer_idx}": value})

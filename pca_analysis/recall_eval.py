# Offline recall@k harness for sparse-attention scorers.
#
# This is the PRIMARY instrument. It needs only the saved key/query tensors and a
# precomputed basis -- no model, no monkeypatch, no pinned transformers. It measures
# top-k SET AGREEMENT against the true (post-RoPE, full-precision) attention, which
# is the objective these methods actually need to preserve.
#
# Ground truth is ALWAYS the exact top-k from raw q.k -- never from projected scores.
# This is the guard against flattering a pre-RoPE basis on its own biased target:
# run the harness on POST-rotary tensors so the exact top-k is the real one.
#
# Methods:
#   loki      : shared basis P for q and k  (key components used for both sides)
#   crosscov  : asymmetric P_q (query components) and P_k (key components)
#
# Usage:
#   python recall_eval.py \
#       --tensor-root  <dir with key/ and query/ saved tensors> \
#       --key-basis    <dir with pca_components/pca_components_layer_*.pt for keys> \
#       --query-basis  <dir ... for queries>      # only needed for --method crosscov \
#       --num-layers   32 \
#       --top-r        8 16 32 64 \
#       --top-k        0.125 \
#       --method       crosscov

import argparse
import glob
import os
import sys

import torch
from parse import parse

NEG_INF = float("-inf")


def get_file_idx(file_name, tensor_type):
    file_template = "tensor_key_{:d}_{:d}.pt".replace("key", tensor_type)
    config = parse(file_template, file_name.split("/")[-1])
    if config is None:
        print(f"[ERROR] bad filename: {file_name}")
        sys.exit(1)
    return config.fixed[1]


def load_layer(layer_id, folder, tensor_type):
    files = glob.glob(os.path.join(folder, f"tensor_{tensor_type}_{layer_id}_*.pt"))
    files = sorted(files, key=lambda x: get_file_idx(x, tensor_type))
    if not files:
        return None
    return torch.stack([torch.load(f, map_location="cpu") for f in files][:-1], dim=0)


def to_columns(comps, top_r):
    # comps: (num_heads, head_dim, head_dim), rows = basis vectors (pca.py convention).
    # Transpose to put basis vectors in columns, then keep the top_r leading columns.
    cols = comps.transpose(-1, -2)
    return cols[:, :, :top_r].contiguous()


def repeat_heads_to(target_heads, tensor, head_dim=-3):
    heads = tensor.shape[head_dim]
    if heads == target_heads:
        return tensor
    if target_heads % heads != 0:
        raise ValueError(f"cannot repeat {heads} heads to {target_heads} heads")
    repeats = target_heads // heads
    return tensor.repeat_interleave(repeats, dim=head_dim)


def topk_indices(scores, k):
    # scores: (num_heads, q_len, k_len) with causal -inf already applied.
    return torch.topk(scores, k, dim=-1).indices


def recall_for_sequence(K, Q, key_cols, query_cols, top_k_frac, runtime_fixed_k=False):
    # K, Q : (num_heads, seq, head_dim) post-rotary, one sequence.
    # key_cols/query_cols : (num_heads, head_dim, top_r) projection columns.
    H, S, d = K.shape
    device = K.device
    scale = d ** 0.5

    causal = torch.triu(torch.full((S, S), NEG_INF, device=device), diagonal=1).unsqueeze(0)  # (1,S,S)

    # Exact scores -> exact top-k (the ground truth).
    exact = torch.matmul(Q, K.transpose(-1, -2)) / scale + causal

    # Approximate scores in the reduced space.
    Kr = torch.matmul(K, key_cols)
    Qr = torch.matmul(Q, query_cols)
    approx = torch.matmul(Qr, Kr.transpose(-1, -2)) / scale + causal

    recalls = []
    fixed_k = int(top_k_frac * S) if top_k_frac <= 1 else int(top_k_frac)
    fixed_k = max(1, min(fixed_k, S))
    for m in range(S):
        valid = m + 1                      # causal: positions 0..m are visible
        if runtime_fixed_k:
            k = fixed_k
        else:
            k = int(top_k_frac * valid) if top_k_frac <= 1 else int(top_k_frac)
            k = max(1, min(k, valid))
        if valid < 2 or k >= valid:
            continue                       # nothing to select among / trivial
        g = torch.topk(exact[:, m, :valid], k, dim=-1).indices    # (H, k)
        a = torch.topk(approx[:, m, :valid], k, dim=-1).indices   # (H, k)
        # recall@k per head = |a ∩ g| / k
        gset = torch.zeros(H, valid, dtype=torch.bool, device=device)
        aset = torch.zeros(H, valid, dtype=torch.bool, device=device)
        gset.scatter_(-1, g, True)
        aset.scatter_(-1, a, True)
        inter = (gset & aset).sum(-1).float()
        recalls.append((inter / k).mean().item())
    return sum(recalls) / len(recalls) if recalls else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-root", required=True)
    ap.add_argument("--key-basis", required=True)
    ap.add_argument("--query-basis", default=None)
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--top-r", type=int, nargs="+", required=True)
    ap.add_argument("--top-k", type=float, default=0.125)
    ap.add_argument("--method", choices=["loki", "crosscov"], default="crosscov")
    ap.add_argument("--max-seqs", type=int, default=64, help="cap sequences per layer for speed")
    ap.add_argument("--device", default="cpu", help="device for recall matmuls, e.g. cpu or cuda")
    ap.add_argument("--runtime-fixed-k", action="store_true",
                    help="match runtime maskers: use k=int(top_k * full sequence length) for all positions")
    ap.add_argument("--per-layer", action="store_true", help="print per-layer recall as well as overall recall")
    args = ap.parse_args()

    if args.method == "crosscov" and args.query_basis is None:
        ap.error("--query-basis is required for --method crosscov")

    print(f"method={args.method} top_k={args.top_k}")
    print(f"{'top_r':>6} | {'recall@k':>9}")

    for top_r in args.top_r:
        per_layer = []
        for layer_id in range(args.num_layers):
            K_all = load_layer(layer_id, f"{args.tensor_root}/key", "key")
            Q_all = load_layer(layer_id, f"{args.tensor_root}/query", "query")
            if K_all is None or Q_all is None:
                continue
            q_heads = Q_all.shape[-3]
            K_all = repeat_heads_to(q_heads, K_all, head_dim=-3)
            kc_file = f"{args.key_basis}/pca_components/pca_components_layer_{layer_id}.pt"
            key_cols = to_columns(torch.load(kc_file, map_location="cpu").float(), top_r)
            key_cols = repeat_heads_to(q_heads, key_cols, head_dim=0).to(args.device)
            if args.method == "loki":
                query_cols = key_cols                      # shared basis
            else:
                qc_file = f"{args.query_basis}/pca_components/pca_components_layer_{layer_id}.pt"
                query_cols = to_columns(torch.load(qc_file, map_location="cpu").float(), top_r)
                query_cols = repeat_heads_to(q_heads, query_cols, head_dim=0).to(args.device)

            # Flatten (num_files, batch, ...) -> a list of sequences.
            nf, b, H, seq, d = K_all.shape
            K_flat = K_all.reshape(nf * b, H, seq, d).float().to(args.device)
            Q_flat = Q_all.reshape(nf * b, H, seq, d).float().to(args.device)
            n_seq = min(args.max_seqs, K_flat.shape[0])
            seq_recalls = []
            for s in range(n_seq):
                r = recall_for_sequence(K_flat[s], Q_flat[s], key_cols, query_cols, args.top_k, args.runtime_fixed_k)
                if r == r:  # not nan
                    seq_recalls.append(r)
            if seq_recalls:
                layer_recall = sum(seq_recalls) / len(seq_recalls)
                per_layer.append(layer_recall)
                if args.per_layer:
                    print(f"  layer {layer_id:>2}: {layer_recall:.4f}")
        overall = sum(per_layer) / len(per_layer) if per_layer else float("nan")
        print(f"{top_r:>6} | {overall:>9.4f}")


if __name__ == "__main__":
    main()

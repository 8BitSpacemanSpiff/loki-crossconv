# Computes the query-key CROSS-covariance SVD basis (CrossCov-SVD).
#
# Loki computes PCA of the KEY auto-covariance E[k k^T] and uses a single shared
# basis for queries and keys. This script instead forms the cross-covariance
#       C = E[k q^T] ~= (1/N) * K^T @ Q          (head_dim x head_dim, NON-symmetric)
# takes its SVD  C = U S V^T, and emits ASYMMETRIC bases:
#       keys    -> top-d LEFT  singular vectors  (U)   -> written to <out>/key
#       queries -> top-d RIGHT singular vectors  (V)   -> written to <out>/query
#
# Output format matches pca_analysis/pca.py exactly (per-head rows = basis vectors,
# shape (num_heads, head_dim, head_dim)), so methods/pca_topk gets_pca_components-style
# loading works unchanged. The query/ subtree is the only new thing the loader needs.
#
# Sanity check (Loki collapse): if you point this at key tensors for BOTH sides, C
# becomes the key auto-covariance E[k k^T] (symmetric), U == V, and you recover
# Loki's PCA basis. recall_eval.py / a unit test should confirm this.
#
# Usage:
#   python crosscov.py <num_layers> <tensor_root> <output_dir> [--whiten] [--pool-gqa]
#     tensor_root : directory holding 'key/' and 'query/' subfolders of saved
#                   tensors (the {rotary_type} level produced by --save-tensors).
#     output_dir  : where to write 'key/' and 'query/' basis subtrees.
#     --whiten    : use the exact Eckart-Young version, SVD of Rk^{-1/2} C Rq^{-1/2}.
#     --pool-gqa  : for GQA/MQA, write one deployable shared U basis per KV head by
#                   pooling query-head cross-covariance energy within each KV group.

import glob
import os
import sys

import torch
from parse import parse

EPS = 1e-6  # eigenvalue floor for the whitening inverse-sqrt


def get_file_idx(file_name, tensor_type="key"):
    file_template = "tensor_key_{:d}_{:d}.pt".replace("key", tensor_type)
    file_name = file_name.split("/")[-1]
    config = parse(file_template, file_name)
    if config is None:
        print(f"[ERROR] Incorrect filename format: {file_name}")
        sys.exit(1)
    _, idx = config.fixed
    return idx


def load_tensors_for_layer(layer_id, folder_path, tensor_type="key"):
    files = glob.glob(os.path.join(folder_path, f"tensor_{tensor_type}_{layer_id}_*.pt"))
    files = sorted(files, key=lambda x: get_file_idx(x, tensor_type))
    if not files:
        print(f"No files found for layer_id {layer_id} in {folder_path}")
        return None
    tensors = []
    for f in files:
        try:
            tensors.append(torch.load(f, map_location="cpu"))
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return tensors


def head_matrix(stacked, head):
    # stacked: (num_files, batch, num_heads, num_tokens, head_dim)
    # returns (N, head_dim) for one head, rows = individual (token,position) vectors
    t = stacked[:, :, head, :, :]
    return t.reshape(-1, t.shape[-1]).to(torch.float64)


def repeat_kv_to_query_heads(K_all, Q_all):
    # GQA/MQA models save fewer key/value heads than query heads. Runtime attention
    # repeats KV heads before scoring, so the offline calibration should do the same.
    k_heads = K_all.shape[-3]
    q_heads = Q_all.shape[-3]
    if k_heads == q_heads:
        return K_all
    if q_heads % k_heads != 0:
        raise ValueError(f"cannot repeat {k_heads} key heads to {q_heads} query heads")
    repeats = q_heads // k_heads
    return K_all.repeat_interleave(repeats, dim=-3)


def inv_sqrt_psd(mat):
    # symmetric PSD inverse square root via eigendecomposition, with a floor
    evals, evecs = torch.linalg.eigh(mat)
    evals = torch.clamp(evals, min=EPS)
    return (evecs * evals.rsqrt()) @ evecs.transpose(-1, -2)


def crosscov_basis(K, Q, whiten):
    # K, Q : (N, head_dim) aligned row-for-row (same forward passes / positions)
    # returns key_rows, query_rows (each head_dim x head_dim, rows = basis vectors,
    # ordered by descending singular value) and the explained-variance proxy.
    N = K.shape[0]
    d = K.shape[1]
    C = (K.transpose(0, 1) @ Q) / N                       # E[k q^T], (d, d)

    if whiten:
        Rk = (K.transpose(0, 1) @ K) / N
        Rq = (Q.transpose(0, 1) @ Q) / N
        Rk_ih = inv_sqrt_psd(Rk)
        Rq_ih = inv_sqrt_psd(Rq)
        C_in = Rk_ih @ C @ Rq_ih
    else:
        Rk_ih = torch.eye(d, dtype=C.dtype, device=C.device)
        Rq_ih = torch.eye(d, dtype=C.dtype, device=C.device)
        C_in = C

    U, S, Vh = torch.linalg.svd(C_in)                     # C_in = U diag(S) Vh
    # Effective projection matrices (columns = basis applied via  x @ cols):
    key_cols = Rk_ih @ U                                  # P_k  (= U when raw)
    query_cols = Rq_ih @ Vh.transpose(0, 1)               # P_q  (= V when raw)
    # Save rows = basis vectors, matching pca.py / sklearn components_ convention.
    key_rows = key_cols.transpose(0, 1).contiguous()
    query_rows = query_cols.transpose(0, 1).contiguous()
    explained = (S ** 2) / (S ** 2).sum()
    return key_rows, query_rows, explained


def pooled_gqa_u_basis(K_all, Q_all, kv_head):
    # Deployable GQA variant: one shared key/query basis per KV head. For a KV
    # group, pool the key-side cross-covariance energy across all query heads:
    #     A = sum_g C_g C_g^T, C_g = E[k q_g^T]
    # and keep the leading eigenvectors of A. This avoids per-query-head bases.
    k_heads = K_all.shape[-3]
    q_heads = Q_all.shape[-3]
    if q_heads % k_heads != 0:
        raise ValueError(f"cannot pool {q_heads} query heads over {k_heads} key heads")

    group = q_heads // k_heads
    K = head_matrix(K_all, kv_head)
    N, d = K.shape
    A = torch.zeros(d, d, dtype=torch.float64, device=K.device)

    for offset in range(group):
        q_head = kv_head * group + offset
        Q = head_matrix(Q_all, q_head)
        C = (K.transpose(0, 1) @ Q) / N
        A = A + C @ C.transpose(0, 1)

    evals, evecs = torch.linalg.eigh(A)
    order = torch.argsort(evals, descending=True)
    evals = torch.clamp(evals[order], min=0)
    evecs = evecs[:, order]
    rows = evecs.transpose(0, 1).contiguous()
    explained = evals / evals.sum().clamp_min(EPS)
    return rows, explained


def pooled_gqa_rq(Q_all, kv_head, group):
    # Pooled query second moment for one KV head: sum over the query heads in its
    # group of E[q q^T]. This matches the group-summed score-energy objective used
    # for query-weighted eviction, so a single (head_dim x head_dim) Rq per KV head
    # scores keys for the whole group.
    d = Q_all.shape[-1]
    Rq = torch.zeros(d, d, dtype=torch.float64, device=Q_all.device)
    for offset in range(group):
        q_head = kv_head * group + offset
        Q = head_matrix(Q_all, q_head)
        N = Q.shape[0]
        Rq = Rq + (Q.transpose(0, 1) @ Q) / N
    return Rq


def save_rq(out_side_dir, layer_id, rq):
    os.makedirs(f"{out_side_dir}/rq_gram", exist_ok=True)
    torch.save(rq.to(torch.float32), f"{out_side_dir}/rq_gram/rq_gram_layer_{layer_id}.pt")


def save_layer(out_side_dir, layer_id, comps, means, explained):
    os.makedirs(f"{out_side_dir}/pca_components", exist_ok=True)
    os.makedirs(f"{out_side_dir}/pca_means", exist_ok=True)
    os.makedirs(f"{out_side_dir}/pca_explained_variance", exist_ok=True)
    torch.save(comps.to(torch.float32), f"{out_side_dir}/pca_components/pca_components_layer_{layer_id}.pt")
    torch.save(means.to(torch.float32), f"{out_side_dir}/pca_means/pca_means_layer_{layer_id}.pt")
    torch.save(explained.to(torch.float32), f"{out_side_dir}/pca_explained_variance/pca_explained_variance_layer_{layer_id}.pt")


def main():
    if len(sys.argv) < 4:
        print("Usage: python crosscov.py <num_layers> <tensor_root> <output_dir> [--whiten] [--pool-gqa] [--emit-rq] [--device cpu|cuda]")
        sys.exit(1)
    num_layers = int(sys.argv[1])
    tensor_root = sys.argv[2]
    output_dir = sys.argv[3]
    whiten = "--whiten" in sys.argv[4:]
    pool_gqa = "--pool-gqa" in sys.argv[4:]
    emit_rq = "--emit-rq" in sys.argv[4:]
    if pool_gqa and whiten:
        print("[ERROR] --pool-gqa currently supports the raw CrossCov-U shared basis only, not --whiten")
        sys.exit(1)
    device = "cpu"
    if "--device" in sys.argv[4:]:
        device = sys.argv[sys.argv.index("--device") + 1]
    print(f"CrossCov-SVD | layers={num_layers} | whiten={whiten} | pool_gqa={pool_gqa} | device={device}")
    print(f"  tensors: {tensor_root}/key , {tensor_root}/query")
    print(f"  output:  {output_dir}/key , {output_dir}/query")

    for layer_id in range(num_layers):
        kt = load_tensors_for_layer(layer_id, f"{tensor_root}/key", "key")
        qt = load_tensors_for_layer(layer_id, f"{tensor_root}/query", "query")
        if kt is None or qt is None:
            print(f"[skip] layer {layer_id}: missing tensors")
            continue
        # Drop the trailing (possibly partial) batch, matching pca.py.
        K_all = torch.stack(kt[:-1], dim=0).to(device)
        Q_all = torch.stack(qt[:-1], dim=0).to(device)
        if not pool_gqa:
            K_all = repeat_kv_to_query_heads(K_all, Q_all)
            assert K_all.shape == Q_all.shape, f"key/query shape mismatch: {K_all.shape} vs {Q_all.shape}"
        num_heads = K_all.shape[-3]
        d = K_all.shape[-1]

        key_comps = torch.zeros(num_heads, d, d)
        query_comps = torch.zeros(num_heads, d, d)
        key_expl = torch.zeros(num_heads, d)
        query_expl = torch.zeros(num_heads, d)

        for h in range(num_heads):
            if pool_gqa:
                kr, expl = pooled_gqa_u_basis(K_all, Q_all, h)
                qr = kr
            else:
                K = head_matrix(K_all, h)
                Q = head_matrix(Q_all, h)
                kr, qr, expl = crosscov_basis(K, Q, whiten)
            key_comps[h], query_comps[h] = kr.cpu(), qr.cpu()
            key_expl[h] = expl.cpu()
            query_expl[h] = expl.cpu()  # same singular spectrum drives both sides

        # Means are not subtracted at inference (Loki's forward projects raw states),
        # so we store zeros to keep the loader happy and the operator uncentered (= E[k q^T]).
        zeros = torch.zeros(num_heads, d)
        save_layer(f"{output_dir}/key", layer_id, key_comps, zeros, key_expl)
        save_layer(f"{output_dir}/query", layer_id, query_comps, zeros, query_expl)

        if emit_rq:
            # One pooled Rq per KV head (head_dim x head_dim), used by query-weighted
            # eviction. When not pooling, num_heads == num query heads and group == 1.
            n_kv = K_all.shape[-3] if pool_gqa else (Q_all.shape[-3])
            group = (Q_all.shape[-3] // n_kv) if pool_gqa else 1
            rq = torch.zeros(n_kv, d, d)
            for kv in range(n_kv):
                rq[kv] = pooled_gqa_rq(Q_all, kv, group).cpu()
            save_rq(f"{output_dir}/key", layer_id, rq)

        print(f"  layer {layer_id}: heads={num_heads} d={d} N={head_matrix(K_all,0).shape[0]} saved")


if __name__ == "__main__":
    main()

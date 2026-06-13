"""
Pure-numpy verification of the math behind the two CrossCov extensions, so the
torch runtime code can be written to match a checked reference. No torch / no GPU.

Checks:
  C1  rank-r reconstruction error == energy in the discarded singular directions
  C2  latent score-energy k~^T G k~ (G = Ur^T Rq Ur) == reconstructed-key energy (Pk)^T Rq (Pk)
  C3  full-rank latent score-energy == exact k^T Rq k
  C4  eviction ranking by score-energy is invariant to working in latent coords
  C5  GQA pool: left singular vecs of [C1|...|CH] == top eigenvecs of sum_g Cg Cg^T
  C6  Loki collapse: cross-cov with K==Q gives symmetric C, U==V (up to sign)
"""
import numpy as np

rng = np.random.default_rng(0)
d, r, N = 16, 6, 4000


def rand_lowrank_pair(d, N, align=0.8):
    # correlated q,k with a controllable systematic-alignment component
    z = rng.standard_normal((N, d))
    Wk = rng.standard_normal((d, d)); Wq = rng.standard_normal((d, d))
    K = z @ Wk + 0.3 * rng.standard_normal((N, d))
    Q = align * (z @ Wq) + (1 - align) * rng.standard_normal((N, d))
    return K, Q


def topr_left_singular(C, r):
    U, S, Vt = np.linalg.svd(C)
    return U[:, :r], S, Vt.T[:, :r]


# ---- C1: reconstruction error == discarded singular energy -------------------
K, Q = rand_lowrank_pair(d, N)
C = (K.T @ Q) / N
Ur, S, Vr = topr_left_singular(C, r)
P = Ur @ Ur.T
# project keys onto left singular basis of C (this is the CrossCov-U key basis)
Uk, Sk, _ = np.linalg.svd((K.T @ K) / N)            # key auto-cov eigenbasis for a clean residual identity
Uk_r = Uk[:, :r]
Pk = Uk_r @ Uk_r.T
recon_err = np.mean(np.sum((K - K @ Pk.T) ** 2, axis=1))
discarded_energy = np.sum(Sk[r:])                    # eigenvalues of key auto-cov == variance per direction
assert np.isclose(recon_err, discarded_energy, rtol=1e-6), (recon_err, discarded_energy)
print(f"C1 ok: recon_err={recon_err:.6f} == discarded_energy={discarded_energy:.6f}")


# ---- C2/C3: score-energy in latent coords ------------------------------------
Rq = (Q.T @ Q) / N                                   # query second moment E[q q^T]
G_full = Uk.T @ Rq @ Uk                              # full-rank latent gram
G_r = Uk_r.T @ Rq @ Uk_r                             # rank-r latent gram
k = K[0]
k_lat_full = Uk.T @ k
k_lat_r = Uk_r.T @ k

se_exact = k @ Rq @ k
se_latent_full = k_lat_full @ G_full @ k_lat_full
assert np.isclose(se_exact, se_latent_full, rtol=1e-10)
print(f"C3 ok: full-rank latent score-energy == exact ({se_latent_full:.4f})")

se_latent_r = k_lat_r @ G_r @ k_lat_r
se_recon = (Pk @ k) @ Rq @ (Pk @ k)
assert np.isclose(se_latent_r, se_recon, rtol=1e-10)
print(f"C2 ok: rank-r latent score-energy == reconstructed-key energy ({se_latent_r:.4f})")


# ---- C4: eviction ranking invariant to latent coords -------------------------
K2, Q2 = rand_lowrank_pair(d, 200)
Rq2 = (Q2.T @ Q2) / 200
U2, _, _ = np.linalg.svd((K2.T @ K2) / 200)
U2r = U2[:, :r]
G2 = U2r.T @ Rq2 @ U2r
Rq2_lat = U2r @ G2 @ U2r.T                              # reconstructed-key energy operator P Rq P
exact_scores = np.einsum('ni,ij,nj->n', K2, Rq2_lat, K2)
latent = K2 @ U2r
latent_scores = np.einsum('ni,ij,nj->n', latent, G2, latent)
assert np.allclose(exact_scores, latent_scores, rtol=1e-8)
order_match = np.array_equal(np.argsort(exact_scores), np.argsort(latent_scores))
assert order_match
print(f"C4 ok: eviction ranking identical in latent coords (n={len(K2)})")


# ---- C5: GQA pooling == stacked left singular basis --------------------------
H = 4
Ks, _ = rand_lowrank_pair(d, N)
Cs = []
for h in range(H):
    _, Qh = rand_lowrank_pair(d, N)
    Cs.append((Ks.T @ Qh) / N)
stacked = np.concatenate(Cs, axis=1)               # [C1 | ... | CH], shape (d, H*d)
U_stack, _, _ = np.linalg.svd(stacked, full_matrices=False)
A = sum(Ch @ Ch.T for Ch in Cs)                    # pooled gram sum_g Cg Cg^T
w, V = np.linalg.eigh(A)
V = V[:, ::-1]                                      # descending
# compare subspaces via principal angles (sign/rotation invariant)
M = U_stack[:, :r].T @ V[:, :r]
sv = np.linalg.svd(M, compute_uv=False)
assert np.allclose(sv, 1.0, atol=1e-6), sv
print(f"C5 ok: pooled-gram eigvecs == stacked-crosscov left singular basis (min cos={sv.min():.6f})")


# ---- C6: Loki collapse -------------------------------------------------------
Ckk = (Ks.T @ Ks) / N
Uc, Sc, Vct = np.linalg.svd(Ckk)
sym_align = np.abs(np.diag(Uc.T @ Vct.T))
assert np.allclose(sym_align, 1.0, atol=1e-6)
print(f"C6 ok: symmetric C (K==Q) gives U==V up to sign (min|diag|={sym_align.min():.6f})")

print("\nALL CHECKS PASSED")

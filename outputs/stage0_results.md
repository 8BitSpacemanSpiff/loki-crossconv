# Stage 0 — Instrument validation on synthetic ground truth

Verdict: **ALL PASS — instruments validated; cleared for Stage 1.**

## Summary
```
  [PASS]  (i)   Selector needle-split
  [PASS]  (ii)  Geometry statistic separability
  [PASS]  (iii) Δ metric ranking (oracle<better<worse<random)
  [PASS]  Robustness (8 seeds)
```

## Detail

### PASS  (i)   Selector needle-split
```
Cloud: blob_with_needles  (n=256, needles=8)
  (a) mean normalised pick-order rank of needles [0=first .. 1=last]:
        diversity (FPS)       0.018
        coverage  (fac.loc.)  0.061
        -> diversity ranks needles strictly earlier: PASS
  (b) needle retention vs eviction budget (div should stay HIGH; cov drops under pressure):
        budget          retain b   div keeps   cov keeps
        50% evict          128      8/8        8/8
        1.25x needles       10      8/8        0/8  <== clean split
        1.5x needles        12      8/8        0/8  <== clean split
        1.75x needles       14      8/8        2/8
        2x needles          16      8/8        4/8
        -> clean split exists under aggressive eviction: PASS
```

### PASS  (ii)  Geometry statistic separability
```
  cloud                 participation  spectral_tail  outlier_fract  clusteredness
  gaussian_blobs               4.5716         0.0688         0.0000        10.0041
  blob_with_needles            7.5359         0.0854         0.0312         0.9319
  uniform_shell               15.1414         0.0842         0.0000         0.2998
  gated designed-direction checks:
    [PASS] outlier_fraction: needles >> blobs
    [PASS] outlier_fraction: needles >> shell
    [PASS] participation_ratio: shell > blobs (isotropic vs few modes)
    [PASS] clusteredness: blobs > shell (structure vs none)
  informational (not gated):
    [ok  ] spectral_tail: needles > blobs (isolated directions) [informational]
```

### PASS  (iii) Δ metric ranking (oracle<better<worse<random)
```
Mixed head (n=248, needles=8, retain b=62 @ 75% eviction). Δ raw | normalised [oracle=0, random=1]:
  oracle                    0.0002 |     0.0000
  diversity (FPS)           0.2182 |     0.0271
  coverage  (fac.loc.)      0.4102 |     0.0509
  random (16 draws)         8.0547 |     1.0000
  required order: oracle < diversity < coverage < random  (satisfied)
  diversity keeps needles: 8/8; coverage keeps needles: 8/8
```

### PASS  Robustness across seeds
```
Each criterion re-run on seeds 0..7 (must pass on all):
  [PASS] (i)   selector needle-split: 8/8 seeds
  [PASS] (ii)  geometry separability: 8/8 seeds
  [PASS] (iii) Δ metric ranking: 8/8 seeds
```

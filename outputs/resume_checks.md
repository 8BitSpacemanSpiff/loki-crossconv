# Resume checks (a) logdet conditioning + (b) kcenter/facility overlap

## (a) Logdet conditioning — is logdet=0-wins a bad ridge?

Per valid head: K_pre second-moment cond number & effective rank, then logdet's
normalised Δ (oracle=0, random=1) under a sweep of ridge eps drawn from the spectrum.
A value near 1.0 = at the random floor; < ~0.85 = meaningfully better than random.

   L kv |       cond effrank |     default     lam_min     lam_med    lam_mean   lmax*1e-3 |  keydiff facility
   0  0 |   1.38e+10     1.0 |       0.662       0.677       0.721       3.876       1.352 |    2.228    0.579
   0  1 |   1.50e+10     1.0 |       0.369       1.579       0.088      53.411      36.105 |    0.007    0.610
   0  2 |   4.53e+09     1.0 |       2.402       2.607       2.447      19.861      15.448 |    1.605    2.577
   0  3 |   1.14e+10     1.2 |       3.929       3.508       3.503      35.435       1.506 |    0.006    0.904
   0  5 |   1.23e+09     1.1 |       1.833       1.818       1.793      -5.465       0.659 |    0.027    2.371
   0  6 |   3.31e+09     1.1 |       1.193       1.090       1.239       3.242       2.962 |    0.781    0.746
   0  7 |   1.10e+10     1.0 |       1.241       1.369       1.346       2.466       1.690 |    1.784    0.670
   8  0 |   1.16e+05     1.4 |       1.367       1.367       1.528       1.554       1.475 |    1.198    1.004
   8  1 |   9.88e+04     1.4 |       1.061       1.061       1.102       1.237       1.087 |    0.377    0.958
   8  2 |   5.96e+04     1.3 |       1.268       1.295       1.421       1.617       1.519 |    1.294    0.723
   8  3 |   4.03e+04     1.4 |       1.170       1.174       1.251       1.410       1.173 |    0.965    0.806
   8  4 |   5.97e+04     1.3 |       1.373       1.385       1.476       1.515       1.491 |    0.677    0.931
   8  5 |   4.37e+04     1.3 |       1.196       1.134       1.047       1.138       1.073 |    0.672    0.812
   8  6 |   5.80e+04     1.2 |       1.124       1.124       1.263       1.429       1.292 |    0.497    0.764
   8  7 |   3.12e+05     1.2 |       1.599       1.590       1.774       2.492       1.958 |    2.792    0.516
  16  0 |   2.91e+04     1.3 |       1.315       1.321       1.192       1.350       1.192 |    0.538    0.807
  16  1 |   1.98e+04     1.7 |       1.132       1.155       1.041       1.254       1.036 |    0.967    0.747
  16  2 |   2.41e+04     1.5 |       1.123       1.299       1.087       1.253       0.999 |    0.425    0.790
  16  3 |   3.54e+04     1.8 |       1.432       1.421       1.368       1.293       1.440 |    0.994    0.807
  16  4 |   1.54e+04     1.7 |       1.343       1.343       1.412       1.452       1.442 |    1.088    0.693
  16  5 |   1.75e+04     1.8 |       1.301       1.308       1.139       1.113       1.093 |    0.215    0.932
  16  6 |   1.95e+04     1.7 |       1.096       1.053       1.139       1.189       1.039 |    0.938    0.853
  16  7 |   2.10e+04     1.8 |       1.323       1.333       1.450       1.420       1.446 |    0.663    0.772
  24  0 |   9.81e+03     1.4 |       1.784       1.667       1.676       2.049       1.649 |    0.475    0.561
  24  1 |   6.75e+03     1.6 |       1.486       1.299       1.359       1.592       1.347 |    0.594    0.923
  24  2 |   1.29e+04     1.4 |       1.447       1.153       1.182       1.269       1.225 |    1.069    0.829
  24  3 |   1.38e+04     1.6 |       1.095       1.100       1.093       1.127       1.072 |    0.436    0.982
  24  4 |   4.61e+03     1.4 |       1.125       1.152       1.167       1.156       1.154 |    1.122    0.708
  24  5 |   1.02e+04     1.4 |       1.500       1.335       1.582       1.564       1.582 |    0.567    1.132
  24  6 |   7.03e+03     1.4 |       1.405       1.470       1.476       1.302       1.462 |    0.018    1.147
  24  7 |   4.59e+03     1.3 |       1.159       1.398       1.300       1.526       1.269 |    0.562    0.886
  31  0 |   1.13e+04     1.3 |       1.254       1.245       1.331       1.298       1.331 |    0.654    0.727
  31  1 |   1.84e+04     1.6 |       1.697       1.667       1.665       1.670       1.650 |    1.318    0.953
  31  2 |   1.06e+04     1.4 |       1.257       1.251       1.361       1.481       1.366 |    1.535    0.825
  31  3 |   9.05e+03     1.5 |       1.444       1.327       1.220       1.332       1.203 |    0.389    0.647
  31  4 |   4.44e+04     1.5 |       1.382       1.376       1.340       1.469       1.323 |    1.143    0.590
  31  5 |   1.24e+04     1.3 |       1.261       1.328       1.373       1.580       1.334 |    0.520    0.953
  31  6 |   1.28e+04     1.5 |       1.004       1.023       0.933       1.038       1.020 |    0.630    0.728
  31  7 |   1.18e+04     1.6 |       2.232       1.980       2.169       1.951       2.073 |    0.990    0.515

  mean K_pre cond = 1.55e+09, mean eff-rank = 1.4 / 128
  logdet mean normalised Δ: default eps = 1.395, best-eps-per-head = 1.071  (oracle 0, random 1)
  heads where default logdet is at/above random floor (Δ>0.85): 37/39
  heads pulled OFF the floor (Δ<0.85) by the BEST spectrum eps: 3/39
  best-eps choice tally: {'default': 12, 'lam_med': 4, 'lam_max*1e-3': 10, 'lam_mean': 4, 'lam_min': 9}
  -> VERDICT: logdet stays at the random floor under EVERY eps -> conditioning is NOT the cause; logdet is genuinely dead on real heads (drop it).

## (b) kcenter vs facility — disjoint, subset, or complementary?

Winner labels are an argmin partition, so win-SETS are disjoint by construction.
Real question: on kcenter-win heads, is facility near-best (redundant) or far off
(complementary)? rank: 1=best of the 4 selectors.

  kcenter wins 6 valid heads; facility wins 16.

  kcenter-win heads -> how facility does there:
    L 0 kv2 : kcenter Δ=0.023  facility Δ=2.577 (rank 4)  gap fac-kc=+2.554
    L 0 kv7 : kcenter Δ=0.465  facility Δ=0.670 (rank 2)  gap fac-kc=+0.204
    L 8 kv4 : kcenter Δ=0.644  facility Δ=0.931 (rank 3)  gap fac-kc=+0.287
    L24 kv3 : kcenter Δ=0.410  facility Δ=0.982 (rank 3)  gap fac-kc=+0.573
    L24 kv7 : kcenter Δ=0.389  facility Δ=0.886 (rank 3)  gap fac-kc=+0.497
    L31 kv0 : kcenter Δ=0.625  facility Δ=0.727 (rank 3)  gap fac-kc=+0.103

  facility-win heads -> how kcenter does there:
    L 0 kv0 : facility Δ=0.579  kcenter Δ=1.777 (rank 3)  gap kc-fac=+1.199
    L 0 kv6 : facility Δ=0.746  kcenter Δ=0.787 (rank 3)  gap kc-fac=+0.041
    L 8 kv0 : facility Δ=1.004  kcenter Δ=1.393 (rank 4)  gap kc-fac=+0.389
    L 8 kv2 : facility Δ=0.723  kcenter Δ=1.081 (rank 2)  gap kc-fac=+0.359
    L 8 kv3 : facility Δ=0.806  kcenter Δ=1.339 (rank 4)  gap kc-fac=+0.532
    L 8 kv7 : facility Δ=0.516  kcenter Δ=1.549 (rank 2)  gap kc-fac=+1.033
    L16 kv1 : facility Δ=0.747  kcenter Δ=1.083 (rank 3)  gap kc-fac=+0.336
    L16 kv3 : facility Δ=0.807  kcenter Δ=1.351 (rank 3)  gap kc-fac=+0.543
    L16 kv4 : facility Δ=0.693  kcenter Δ=1.346 (rank 4)  gap kc-fac=+0.653
    L16 kv6 : facility Δ=0.853  kcenter Δ=0.912 (rank 2)  gap kc-fac=+0.059
    L24 kv2 : facility Δ=0.829  kcenter Δ=1.228 (rank 3)  gap kc-fac=+0.399
    L24 kv4 : facility Δ=0.708  kcenter Δ=1.193 (rank 4)  gap kc-fac=+0.485
    L31 kv1 : facility Δ=0.953  kcenter Δ=1.610 (rank 3)  gap kc-fac=+0.657
    L31 kv2 : facility Δ=0.825  kcenter Δ=0.826 (rank 2)  gap kc-fac=+0.001
    L31 kv4 : facility Δ=0.590  kcenter Δ=1.284 (rank 3)  gap kc-fac=+0.694
    L31 kv7 : facility Δ=0.515  kcenter Δ=1.479 (rank 3)  gap kc-fac=+0.964

  across-head corr(kcenter Δ, facility Δ) over 39 heads = -0.421
  kcenter-win heads where facility is FAR off (rank>2 or gap>0.30): 5/6
  -> kcenter is COMPLEMENTARY (wins heads facility cannot) -> keep both in Phase D


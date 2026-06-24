# Phase C — routable-margin probe

## Step 0 — canonical facility check (K equal-mass clusters, b=K)
  bandwidth=10th-pct: facility BEATS logdet/kcenter (7/8 seeds, mean Δ gap vs best-other +0.0462)
  -> facility coverage selector is SOUND; using bandwidth=10th-pct for the real-head probe.

## Step 1 — real-head probe (keep=3%, layers 0/8/16/24/31 x 8 KV, N_cap=1024, avg 2 seq)  [PROBE: reduced averaging -> exact per-head winners are noisy; qualitative tallies are the signal]
   L kv |   logdet  keydiff  kcenter facility |   winner  margin   ld-kc    flag
   0  0 |    0.662    2.228    1.777    0.579 | facility   0.084  -1.115   route
   0  1 |    0.369    0.007    2.834    0.610 |  keydiff   0.602  -2.465   route
   0  2 |    2.402    1.605    0.023    2.577 |  kcenter   0.972  +2.379   route
   0  3 |    3.929    0.006    2.081    0.904 |  keydiff   0.897  +1.849   route
   0  4 |   -0.052   52.401   -1.374   -1.072 |  kcenter   1.020  +1.322 DIFFUSE
   0  5 |    1.833    0.027    0.707    2.371 |  keydiff   2.345  +1.126   route
   0  6 |    1.193    0.781    0.787    0.746 | facility   0.035  +0.406        
   0  7 |    1.241    1.784    0.465    0.670 |  kcenter   0.572  +0.776   route
   8  0 |    1.367    1.198    1.385    0.999 | facility   0.199  -0.017   route
   8  1 |    1.061    0.377    0.613    0.958 |  keydiff   0.581  +0.448   route
   8  2 |    1.268    1.295    1.074    0.722 | facility   0.546  +0.193   route
   8  3 |    1.170    0.965    1.338    0.827 | facility   0.138  -0.169   route
   8  4 |    1.373    0.676    0.644    0.931 |  kcenter   0.255  +0.729   route
   8  5 |    1.196    0.672    1.144    0.813 |  keydiff   0.141  +0.052   route
   8  6 |    1.125    0.497    0.620    0.764 |  keydiff   0.267  +0.505   route
   8  7 |    1.636    2.753    1.548    0.517 | facility   1.119  +0.088   route
  16  0 |    1.314    0.536    0.559    0.808 |  keydiff   0.271  +0.755   route
  16  1 |    1.180    0.967    1.133    0.746 | facility   0.221  +0.047   route
  16  2 |    1.122    0.424    0.579    0.790 |  keydiff   0.365  +0.544   route
  16  3 |    1.432    0.995    1.350    0.808 | facility   0.188  +0.082   route
  16  4 |    1.344    1.087    1.317    0.693 | facility   0.394  +0.027   route
  16  5 |    1.301    0.216    0.477    0.932 |  keydiff   0.717  +0.824   route
  16  6 |    1.097    0.938    0.913    0.853 | facility   0.085  +0.183   route
  16  7 |    1.322    0.663    0.855    0.772 |  keydiff   0.110  +0.467   route
  24  0 |    1.784    0.475    1.138    0.561 |  keydiff   0.086  +0.646   route
  24  1 |    1.486    0.594    1.302    0.923 |  keydiff   0.330  +0.185   route
  24  2 |    1.448    1.068    1.228    0.830 | facility   0.239  +0.220   route
  24  3 |    1.095    0.436    0.410    0.982 |  kcenter   0.546  +0.685   route
  24  4 |    1.125    1.122    1.192    0.708 | facility   0.414  -0.067   route
  24  5 |    1.501    0.567    0.747    1.131 |  keydiff   0.564  +0.754   route
  24  6 |    1.405    0.018    0.043    1.149 |  keydiff   1.131  +1.362   route
  24  7 |    1.159    0.563    0.389    0.886 |  kcenter   0.323  +0.770   route
  31  0 |    1.130    0.655    0.625    0.728 |  kcenter   0.073  +0.505   route
  31  1 |    1.697    1.294    1.609    0.953 | facility   0.342  +0.088   route
  31  2 |    1.241    1.534    0.824    0.830 |  kcenter   0.411  +0.417   route
  31  3 |    1.445    0.390    1.157    0.650 |  keydiff   0.260  +0.287   route
  31  4 |    1.353    1.143    1.284    0.591 | facility   0.553  +0.069   route
  31  5 |    1.260    0.520    1.274    0.953 |  keydiff   0.433  -0.014   route
  31  6 |    1.006    0.630    1.049    0.736 |  keydiff   0.106  -0.043   route
  31  7 |    2.231    0.989    1.429    0.515 | facility   0.473  +0.802   route

## Decisions
  valid heads: 39/40  (excluded as too-diffuse / oracle-not-floor: 1)
  winner tally (valid heads): {'facility': 15, 'keydiff': 17, 'kcenter': 7}
  ROUTABILITY: heads with |margin| > noise floor (0.05): 38/39  (median margin 0.342, max 2.345)
  facility ever wins a valid head: True  -> routable regime exists
  LOGDET vs KCENTER: heads with |Δlogdet-Δkcenter| > 0.05: 34/39  (mean |ld-kc| 0.568)  -> SPLIT (volume vs min-max distinct)
  logdet wins: 0 heads  -> logdet is NEVER best on real heads; the synthetic volume-diversity story does NOT transfer.
  CAVEAT: synthetic Phase A predicted logdet dominates / facility never wins -> the real heads show the OPPOSITE (logdet worst, facility+keydiff win). Origin-centred isotropic synthetic clouds mis-ranked both. Trust the real-head ordering.

# Phase C — routable-margin probe

## Step 0 — canonical facility check (K equal-mass clusters, b=K)
  bandwidth=10th-pct: facility BEATS logdet/kcenter (7/8 seeds, mean Δ gap vs best-other +0.0462)
  -> facility coverage selector is SOUND; using bandwidth=10th-pct for the real-head probe.

## Step 1 — real-head probe (keep=3%, layers 0/8/16/24/31 x 8 KV, N_cap=1024, avg 2 seq)  [PROBE: reduced averaging -> exact per-head winners are noisy; qualitative tallies are the signal]
   L kv |   logdet  keydiff  kcenter facility |   winner  margin   ld-kc    flag
   0  0 |    0.662    2.228    1.777    0.579 | facility   0.084  -1.115   route
   0  1 |    0.369    0.007    2.834    0.610 |  keydiff   0.603  -2.464   route
   0  2 |    2.402    1.605    0.023    2.577 |  kcenter   0.972  +2.378   route
   0  3 |    3.929    0.006    2.081    0.904 |  keydiff   0.897  +1.849   route
   0  4 |   -0.051   52.400   -1.374   -1.072 |  kcenter   1.021  +1.322 DIFFUSE
   0  5 |    1.833    0.027    0.707    2.371 |  keydiff   2.345  +1.126   route
   0  6 |    1.193    0.781    0.787    0.746 | facility   0.035  +0.406        
   0  7 |    1.241    1.784    0.465    0.670 |  kcenter   0.572  +0.776   route
   8  0 |    1.367    1.198    1.393    1.004 | facility   0.194  -0.026   route
   8  1 |    1.061    0.377    0.613    0.958 |  keydiff   0.581  +0.448   route
   8  2 |    1.268    1.294    1.081    0.723 | facility   0.545  +0.186   route
   8  3 |    1.170    0.965    1.339    0.806 | facility   0.159  -0.169   route
   8  4 |    1.373    0.677    0.644    0.931 |  kcenter   0.254  +0.729   route
   8  5 |    1.196    0.672    1.144    0.812 |  keydiff   0.141  +0.052   route
   8  6 |    1.124    0.497    0.617    0.764 |  keydiff   0.267  +0.507   route
   8  7 |    1.599    2.792    1.549    0.516 | facility   1.083  +0.051   route
  16  0 |    1.315    0.538    0.560    0.807 |  keydiff   0.269  +0.755   route
  16  1 |    1.132    0.967    1.083    0.747 | facility   0.220  +0.049   route
  16  2 |    1.123    0.425    0.579    0.790 |  keydiff   0.365  +0.543   route
  16  3 |    1.432    0.994    1.351    0.807 | facility   0.187  +0.081   route
  16  4 |    1.343    1.088    1.346    0.693 | facility   0.394  -0.003   route
  16  5 |    1.301    0.215    0.483    0.932 |  keydiff   0.717  +0.817   route
  16  6 |    1.096    0.938    0.912    0.853 | facility   0.085  +0.184   route
  16  7 |    1.323    0.663    0.855    0.772 |  keydiff   0.109  +0.468   route
  24  0 |    1.784    0.475    1.137    0.561 |  keydiff   0.087  +0.647   route
  24  1 |    1.486    0.594    1.299    0.923 |  keydiff   0.329  +0.187   route
  24  2 |    1.447    1.069    1.228    0.829 | facility   0.240  +0.220   route
  24  3 |    1.095    0.436    0.410    0.982 |  kcenter   0.546  +0.685   route
  24  4 |    1.125    1.122    1.193    0.708 | facility   0.414  -0.068   route
  24  5 |    1.500    0.567    0.747    1.132 |  keydiff   0.564  +0.753   route
  24  6 |    1.405    0.018    0.043    1.147 |  keydiff   1.129  +1.362   route
  24  7 |    1.159    0.562    0.389    0.886 |  kcenter   0.324  +0.770   route
  31  0 |    1.254    0.654    0.625    0.727 |  kcenter   0.073  +0.629   route
  31  1 |    1.697    1.318    1.610    0.953 | facility   0.365  +0.088   route
  31  2 |    1.257    1.535    0.826    0.825 | facility   0.432  +0.431   route
  31  3 |    1.444    0.389    1.157    0.647 |  keydiff   0.258  +0.287   route
  31  4 |    1.382    1.143    1.284    0.590 | facility   0.553  +0.098   route
  31  5 |    1.261    0.520    1.270    0.953 |  keydiff   0.433  -0.009   route
  31  6 |    1.004    0.630    1.047    0.728 |  keydiff   0.098  -0.043   route
  31  7 |    2.232    0.990    1.479    0.515 | facility   0.475  +0.753   route

## Decisions
  valid heads: 39/40  (excluded as too-diffuse / oracle-not-floor: 1)
  winner tally (valid heads): {'facility': 16, 'keydiff': 17, 'kcenter': 6}
  ROUTABILITY: heads with |margin| > noise floor (0.05): 38/39  (median margin 0.365, max 2.345)
  facility ever wins a valid head: True  -> routable regime exists
  LOGDET vs KCENTER: heads with |Δlogdet-Δkcenter| > 0.05: 34/39  (mean |ld-kc| 0.570)  -> SPLIT (volume vs min-max distinct)
  logdet wins: 0 heads  -> logdet is NEVER best on real heads; the synthetic volume-diversity story does NOT transfer.
  CAVEAT: synthetic Phase A predicted logdet dominates / facility never wins -> the real heads show the OPPOSITE (logdet worst, facility+keydiff win). Origin-centred isotropic synthetic clouds mis-ranked both. Trust the real-head ordering.

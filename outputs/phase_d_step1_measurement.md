# Phase D STEP 1 — full-resolution per-head measurement

256 heads (32 layers x 8 KV), FULL cloud (7936 keys), n_seq=8 (held-out split 4/4), budgets ['3%', '10%', '25%', '50%'].
Normalised Δ: oracle=0, random=1 (lower=better). Routing label = argmin{facility,kcenter}.
Δ + oracle on real K_post/V; selectors/geometry on K_pre (centered where noted).

## budget 3%  <<< PRIMARY (aggressive)
  valid heads 255/256 (diffuse-excluded 1)
  winner tally {facility,kcenter}: {'facility': 23, 'kcenter': 232}
  3-way sensitivity {+keydiff_cen}: {'facility': 22, 'keydiff_cen': 7, 'kcenter': 226}
  margin |fac-kc|: min 0.000 p10 0.009 p25 0.180 med 0.590 p75 0.768 p90 0.906 max 8.974
  NEAR-TIE mass: <0.05 -> 43/255, <0.10 -> 49/255 (label noise the AUC must not fit on)
  winner STABILITY across split A vs B: 233/255 stable, 22 flip
  BASELINE: routed min{fac,kc} beats keydiff_unc on 30/255 heads

## budget 10%
  valid heads 253/256 (diffuse-excluded 3)
  winner tally {facility,kcenter}: {'facility': 113, 'kcenter': 140}
  3-way sensitivity {+keydiff_cen}: {'facility': 101, 'keydiff_cen': 24, 'kcenter': 128}
  margin |fac-kc|: min 0.000 p10 0.004 p25 0.011 med 0.036 p75 0.086 p90 0.155 max 0.939
  NEAR-TIE mass: <0.05 -> 153/253, <0.10 -> 203/253 (label noise the AUC must not fit on)
  winner STABILITY across split A vs B: 170/253 stable, 83 flip
  BASELINE: routed min{fac,kc} beats keydiff_unc on 37/253 heads

## budget 25%
  valid heads 251/256 (diffuse-excluded 5)
  winner tally {facility,kcenter}: {'facility': 161, 'kcenter': 90}
  3-way sensitivity {+keydiff_cen}: {'keydiff_cen': 81, 'facility': 101, 'kcenter': 69}
  margin |fac-kc|: min 0.000 p10 0.004 p25 0.014 med 0.047 p75 0.095 p90 0.182 max 0.780
  NEAR-TIE mass: <0.05 -> 131/251, <0.10 -> 191/251 (label noise the AUC must not fit on)
  winner STABILITY across split A vs B: 186/251 stable, 65 flip
  BASELINE: routed min{fac,kc} beats keydiff_unc on 25/251 heads

## budget 50%
  valid heads 252/256 (diffuse-excluded 4)
  winner tally {facility,kcenter}: {'facility': 163, 'kcenter': 89}
  3-way sensitivity {+keydiff_cen}: {'keydiff_cen': 172, 'facility': 47, 'kcenter': 33}
  margin |fac-kc|: min 0.001 p10 0.006 p25 0.028 med 0.129 p75 0.290 p90 0.431 max 5.281
  NEAR-TIE mass: <0.05 -> 85/252, <0.10 -> 111/252 (label noise the AUC must not fit on)
  winner STABILITY across split A vs B: 189/252 stable, 63 flip
  BASELINE: routed min{fac,kc} beats keydiff_unc on 8/252 heads

## winner migration across budgets (heads valid at all budgets)
  250 heads valid at every budget; (win@3%,@10%,@25%,@50%) patterns:
    ('kcenter', 'facility', 'facility', 'facility'): 86
    ('kcenter', 'kcenter', 'kcenter', 'kcenter'): 67
    ('kcenter', 'kcenter', 'facility', 'facility'): 47
    ('facility', 'facility', 'facility', 'facility'): 14
    ('kcenter', 'kcenter', 'kcenter', 'facility'): 12
    ('kcenter', 'kcenter', 'facility', 'kcenter'): 9
    ('kcenter', 'facility', 'kcenter', 'kcenter'): 6
    ('facility', 'kcenter', 'kcenter', 'kcenter'): 2
    ('facility', 'facility', 'kcenter', 'kcenter'): 2
    ('facility', 'facility', 'facility', 'kcenter'): 2
    ('kcenter', 'facility', 'kcenter', 'facility'): 1
    ('facility', 'kcenter', 'facility', 'facility'): 1

## per-head @ 3% (primary)
   L kv |    fac     kc  kd_unc  kd_cen  ld_cen |      win margin        A        B    flag
   0  0 |   0.12   0.14    1.34   10.24   -0.27 | facility   0.01 facility facility        
   0  1 |   0.22   0.23    0.16  122.64   -0.22 | facility   0.01 facility  kcenter    FLIP
   0  2 | 108.84  99.87  -50.56 -1432.53  144.85 |  kcenter   8.97  kcenter  kcenter        
   0  3 |   1.22   1.05    0.03 -123.98    1.12 |  kcenter   0.17  kcenter  kcenter        
   0  4 |   0.61   0.40   -6.88   -7.35    0.62 |  kcenter   0.20  kcenter  kcenter        
   0  5 |   0.47   0.43   -0.03   -2.44    0.42 |  kcenter   0.03  kcenter  kcenter        
   0  6 |   0.85   1.12    2.56    0.61    1.79 | facility   0.27  kcenter facility    FLIP
   0  7 |   1.73   1.89    1.34   22.00    0.02 | facility   0.15 facility facility        
   1  0 |   1.01   0.03    0.03    1.50    0.03 |  kcenter   0.98  kcenter  kcenter        
   1  1 |   1.06   0.08    0.09    3.46    0.08 |  kcenter   0.98  kcenter  kcenter        
   1  2 |   1.51   0.06    0.06    2.71    0.07 |  kcenter   1.45  kcenter  kcenter        
   1  3 |   0.38   0.01    0.02    5.43    0.01 |  kcenter   0.37  kcenter  kcenter        
   1  4 |   0.88   0.01    0.01    1.89    0.01 |  kcenter   0.87  kcenter  kcenter        
   1  5 |   1.08   0.01    0.02    1.55    0.02 |  kcenter   1.07  kcenter  kcenter        
   1  6 |   1.86  -0.05   -0.01    3.44   -0.05 |  kcenter   1.91  kcenter  kcenter DIFFUSE
   1  7 |   0.77   0.16    0.19    1.84    0.17 |  kcenter   0.62  kcenter  kcenter        
   2  0 |   0.90   0.07    0.06    2.26    0.07 |  kcenter   0.83  kcenter  kcenter        
   2  1 |   1.30   0.04    0.03    2.80    0.04 |  kcenter   1.26  kcenter  kcenter        
   2  2 |   1.20   0.01    0.01    3.41    0.01 |  kcenter   1.19  kcenter  kcenter        
   2  3 |   0.10   0.09    0.04    2.08    0.08 |  kcenter   0.01  kcenter  kcenter        
   2  4 |   1.22   0.10    0.05    2.04    0.08 |  kcenter   1.12  kcenter  kcenter        
   2  5 |   1.36   1.64    0.09    2.03    0.07 | facility   0.27 facility facility        
   2  6 |   0.33  -0.00    0.01    3.57   -0.00 |  kcenter   0.34  kcenter  kcenter        
   2  7 |   1.09   0.02    0.01    2.10    0.02 |  kcenter   1.07  kcenter  kcenter        
   3  0 |   0.08   0.10    0.07    2.16    0.06 | facility   0.02 facility facility        
   3  1 |   0.02   0.02    0.02    1.67    0.02 |  kcenter   0.00  kcenter  kcenter        
   3  2 |   0.01   0.01    0.01    1.44    0.01 |  kcenter   0.00  kcenter  kcenter        
   3  3 |   0.01   0.01    0.00    2.04    0.00 |  kcenter   0.00  kcenter  kcenter        
   3  4 |   0.14   0.12    0.10    2.16    0.11 |  kcenter   0.02  kcenter  kcenter        
   3  5 |   0.06   0.06    0.05    2.84    0.06 |  kcenter   0.00  kcenter  kcenter        
   3  6 |   0.01   0.01    0.02    1.50    0.01 |  kcenter   0.00  kcenter  kcenter        
   3  7 |   0.02   0.02    0.01    1.23    0.02 |  kcenter   0.00  kcenter  kcenter        
   4  0 |   0.27   0.09    0.06    1.38    0.07 |  kcenter   0.18  kcenter  kcenter        
   4  1 |   0.06   0.05    0.02    1.96    0.02 |  kcenter   0.02 facility  kcenter    FLIP
   4  2 |   0.05   0.05    0.02    1.30    0.02 | facility   0.00  kcenter facility    FLIP
   4  3 |   0.10   0.09    0.11    2.15    0.09 |  kcenter   0.00 facility  kcenter    FLIP
   4  4 |   0.04   0.04    0.01    2.12    0.01 |  kcenter   0.00  kcenter  kcenter        
   4  5 |   0.16   0.19    0.05    2.13    0.06 | facility   0.03 facility  kcenter    FLIP
   4  6 |   0.30   0.25    0.10    1.61    0.10 |  kcenter   0.05 facility  kcenter    FLIP
   4  7 |   0.60   0.06    0.08    2.31    0.06 |  kcenter   0.54  kcenter  kcenter        
   5  0 |   0.02   0.04    0.02    1.69    0.02 | facility   0.02 facility facility        
   5  1 |   0.13   0.16    0.11    1.66    0.13 | facility   0.02  kcenter facility    FLIP
   5  2 |   0.10   0.11    0.09    1.51    0.09 | facility   0.01 facility facility        
   5  3 |   0.12   0.12    0.06    1.67    0.08 |  kcenter   0.00  kcenter  kcenter        
   5  4 |   0.04   0.03    0.01    1.61    0.03 |  kcenter   0.00  kcenter  kcenter        
   5  5 |   0.04   0.11    0.03    1.50    0.03 | facility   0.07 facility facility        
   5  6 |   1.05   0.09    0.07    1.28    0.09 |  kcenter   0.97  kcenter  kcenter        
   5  7 |   0.72   0.21    0.17    1.29    0.19 |  kcenter   0.52  kcenter  kcenter        
   6  0 |   1.01   0.15    0.13    1.83    0.15 |  kcenter   0.86  kcenter  kcenter        
   6  1 |   0.04   0.04    0.02    1.68    0.02 | facility   0.00 facility facility        
   6  2 |   0.52   0.23    0.16    1.77    0.17 |  kcenter   0.30  kcenter  kcenter        
   6  3 |   0.07   0.07    0.02    1.46    0.04 |  kcenter   0.00  kcenter  kcenter        
   6  4 |   0.14   0.12    0.03    1.41    0.08 |  kcenter   0.02  kcenter  kcenter        
   6  5 |   0.93   0.32    0.12    1.32    0.20 |  kcenter   0.61  kcenter  kcenter        
   6  6 |   0.08   0.08    0.09    1.69    0.09 |  kcenter   0.00  kcenter  kcenter        
   6  7 |   0.05   0.05    0.04    1.65    0.05 |  kcenter   0.00  kcenter  kcenter        
   7  0 |   0.12   0.11    0.06    0.93    0.08 |  kcenter   0.01  kcenter  kcenter        
   7  1 |   0.96   0.30    0.21    1.54    0.27 |  kcenter   0.66  kcenter  kcenter        
   7  2 |   0.20   0.09    0.08    1.57    0.09 |  kcenter   0.11  kcenter  kcenter        
   7  3 |   0.92   0.27    0.22    1.80    0.26 |  kcenter   0.65  kcenter  kcenter        
   7  4 |   0.97   0.34    0.24    1.53    0.21 |  kcenter   0.62  kcenter  kcenter        
   7  5 |   0.98   0.17    0.19    1.79    0.20 |  kcenter   0.81  kcenter  kcenter        
   7  6 |   0.26   0.12    0.04    1.35    0.05 |  kcenter   0.14  kcenter  kcenter        
   7  7 |   0.31   0.68    0.12    1.13    0.13 | facility   0.37 facility facility        
   8  0 |   1.09   0.12    0.11    1.68    0.13 |  kcenter   0.98  kcenter  kcenter        
   8  1 |   0.97   0.27    0.24    1.32    0.27 |  kcenter   0.71  kcenter  kcenter        
   8  2 |   0.17   0.17    0.13    2.01    0.14 | facility   0.00 facility facility        
   8  3 |   0.90   0.27    0.20    1.10    0.29 |  kcenter   0.63  kcenter  kcenter        
   8  4 |   0.34   0.51    0.23    1.46    0.21 | facility   0.17 facility facility        
   8  5 |   0.19   0.14    0.09    1.21    0.10 |  kcenter   0.04 facility  kcenter    FLIP
   8  6 |   0.24   0.34    0.04    1.83    0.04 | facility   0.10 facility  kcenter    FLIP
   8  7 |   0.03   0.03    0.01    1.49    0.01 | facility   0.00  kcenter facility    FLIP
   9  0 |   0.33   0.12    0.13    1.72    0.12 |  kcenter   0.21  kcenter  kcenter        
   9  1 |   0.82   0.31    0.23    1.31    0.31 |  kcenter   0.51  kcenter  kcenter        
   9  2 |   0.67   0.27    0.23    1.66    0.28 |  kcenter   0.40  kcenter  kcenter        
   9  3 |   0.91   0.46    0.29    1.19    0.40 |  kcenter   0.45  kcenter  kcenter        
   9  4 |   0.64   0.28    0.14    0.41    0.29 |  kcenter   0.36  kcenter  kcenter        
   9  5 |   0.99   0.22    0.22    1.45    0.24 |  kcenter   0.77  kcenter  kcenter        
   9  6 |   1.01   0.21    0.22    1.81    0.23 |  kcenter   0.80  kcenter  kcenter        
   9  7 |   0.96   0.33    0.31    1.46    0.31 |  kcenter   0.63  kcenter  kcenter        
  10  0 |   0.91   0.25    0.20    1.22    0.23 |  kcenter   0.65  kcenter  kcenter        
  10  1 |   1.12   0.27    0.08    0.43    0.12 |  kcenter   0.84  kcenter  kcenter        
  10  2 |   0.94   0.22    0.22    1.25    0.24 |  kcenter   0.72  kcenter  kcenter        
  10  3 |   0.12   0.11    0.04    1.79    0.11 |  kcenter   0.01  kcenter  kcenter        
  10  4 |   0.80   0.64    0.15    1.26    0.28 |  kcenter   0.15  kcenter facility    FLIP
  10  5 |   0.91   0.17    0.21    1.33    0.17 |  kcenter   0.74  kcenter  kcenter        
  10  6 |   1.06   0.27    0.29    1.48    0.28 |  kcenter   0.78  kcenter  kcenter        
  10  7 |   0.53   0.58    0.09    1.15    0.10 | facility   0.05 facility facility        
  11  0 |   0.89   0.29    0.24    1.51    0.26 |  kcenter   0.61  kcenter  kcenter        
  11  1 |   0.76   0.29    0.21    1.32    0.27 |  kcenter   0.47  kcenter  kcenter        
  11  2 |   0.65   0.14    0.11    1.35    0.14 |  kcenter   0.51  kcenter  kcenter        
  11  3 |   0.86   0.20    0.20    1.20    0.19 |  kcenter   0.66  kcenter  kcenter        
  11  4 |   0.27   0.26    0.17    1.63    0.19 |  kcenter   0.01  kcenter facility    FLIP
  11  5 |   0.88   0.88    0.65    1.57    0.73 |  kcenter   0.01 facility  kcenter    FLIP
  11  6 |   0.22   0.16    0.08    0.94    0.10 |  kcenter   0.07  kcenter facility    FLIP
  11  7 |   0.95   0.31    0.26    1.42    0.31 |  kcenter   0.64  kcenter  kcenter        
  12  0 |   0.07   0.07    0.09    1.50    0.09 |  kcenter   0.00 facility  kcenter    FLIP
  12  1 |   1.02   0.28    0.13    1.49    0.12 |  kcenter   0.73  kcenter  kcenter        
  12  2 |   0.92   0.36    0.35    1.43    0.37 |  kcenter   0.56  kcenter  kcenter        
  12  3 |   1.03   0.35    0.22    1.60    0.31 |  kcenter   0.67  kcenter  kcenter        
  12  4 |   0.83   0.52    0.43    1.72    0.51 |  kcenter   0.31  kcenter  kcenter        
  12  5 |   0.96   0.39    0.30    1.40    0.38 |  kcenter   0.57  kcenter  kcenter        
  12  6 |   0.90   0.48    0.43    1.42    0.50 |  kcenter   0.42  kcenter  kcenter        
  12  7 |   1.08   0.28    0.23    1.57    0.28 |  kcenter   0.80  kcenter  kcenter        
  13  0 |   0.41   0.30    0.22    1.43    0.30 |  kcenter   0.11  kcenter  kcenter        
  13  1 |   0.92   0.38    0.37    1.76    0.36 |  kcenter   0.55  kcenter  kcenter        
  13  2 |   0.56   0.26    0.15    1.06    0.24 |  kcenter   0.30  kcenter  kcenter        
  13  3 |   1.17   0.44    0.12    1.46    0.14 |  kcenter   0.73 facility  kcenter    FLIP
  13  4 |   0.93   0.32    0.22    1.35    0.25 |  kcenter   0.61  kcenter  kcenter        
  13  5 |   0.89   0.27    0.22    1.25    0.27 |  kcenter   0.62  kcenter  kcenter        
  13  6 |   1.12   0.46    0.32    1.35    0.36 |  kcenter   0.66  kcenter  kcenter        
  13  7 |   0.96   0.14    0.11    1.48    0.13 |  kcenter   0.82  kcenter  kcenter        
  14  0 |   0.87   0.36    0.38    1.62    0.41 |  kcenter   0.51  kcenter  kcenter        
  14  1 |   0.89   0.28    0.30    1.07    0.28 |  kcenter   0.61  kcenter  kcenter        
  14  2 |   0.89   0.33    0.30    1.50    0.31 |  kcenter   0.56  kcenter  kcenter        
  14  3 |   0.83   0.67    0.26    1.01    0.36 |  kcenter   0.16  kcenter  kcenter        
  14  4 |   0.93   0.21    0.23    1.76    0.22 |  kcenter   0.71  kcenter  kcenter        
  14  5 |   0.89   0.43    0.45    1.35    0.44 |  kcenter   0.46  kcenter  kcenter        
  14  6 |   1.03   0.46    0.47    1.76    0.48 |  kcenter   0.57  kcenter  kcenter        
  14  7 |   0.97   0.24    0.23    1.61    0.24 |  kcenter   0.72  kcenter  kcenter        
  15  0 |   0.90   0.45    0.40    1.60    0.43 |  kcenter   0.45  kcenter  kcenter        
  15  1 |   0.88   0.37    0.29    1.17    0.40 |  kcenter   0.51  kcenter  kcenter        
  15  2 |   0.91   0.72    0.65    1.80    0.65 |  kcenter   0.19  kcenter  kcenter        
  15  3 |   0.28   0.11    0.08    1.13    0.09 |  kcenter   0.17  kcenter  kcenter        
  15  4 |   0.89   0.53    0.45    1.62    0.46 |  kcenter   0.36  kcenter  kcenter        
  15  5 |   0.99   0.37    0.20    1.88    0.21 |  kcenter   0.62  kcenter  kcenter        
  15  6 |   0.91   0.18    0.17    1.36    0.18 |  kcenter   0.73  kcenter  kcenter        
  15  7 |   0.84   0.57    0.49    1.34    0.54 |  kcenter   0.27  kcenter  kcenter        
  16  0 |   0.98   0.38    0.23    1.42    0.38 |  kcenter   0.60  kcenter  kcenter        
  16  1 |   0.94   0.43    0.41    1.37    0.44 |  kcenter   0.51  kcenter  kcenter        
  16  2 |   0.89   0.19    0.11    1.22    0.15 |  kcenter   0.69  kcenter  kcenter        
  16  3 |   0.89   0.44    0.45    1.84    0.46 |  kcenter   0.45  kcenter  kcenter        
  16  4 |   0.97   0.21    0.18    1.56    0.22 |  kcenter   0.76  kcenter  kcenter        
  16  5 |   0.87   0.35    0.28    1.23    0.31 |  kcenter   0.52  kcenter  kcenter        
  16  6 |   0.89   0.34    0.29    1.00    0.34 |  kcenter   0.55  kcenter  kcenter        
  16  7 |   0.84   0.35    0.31    1.51    0.32 |  kcenter   0.49  kcenter  kcenter        
  17  0 |   1.15   0.31    0.22    1.47    0.31 |  kcenter   0.84  kcenter  kcenter        
  17  1 |   0.58   0.18    0.10    0.84    0.12 |  kcenter   0.39  kcenter  kcenter        
  17  2 |   0.13   0.12    0.10    1.45    0.11 |  kcenter   0.01  kcenter facility    FLIP
  17  3 |   1.09   0.43    0.12    1.33    0.14 |  kcenter   0.66  kcenter  kcenter        
  17  4 |   1.07   0.22    0.17    1.61    0.20 |  kcenter   0.85  kcenter  kcenter        
  17  5 |   0.60   0.28    0.09    1.32    0.15 |  kcenter   0.32  kcenter  kcenter        
  17  6 |   0.90   0.40    0.38    1.50    0.37 |  kcenter   0.51  kcenter  kcenter        
  17  7 |   0.77   0.17    0.11    1.44    0.11 |  kcenter   0.61  kcenter  kcenter        
  18  0 |   0.99   0.22    0.20    1.36    0.27 |  kcenter   0.77  kcenter  kcenter        
  18  1 |   0.94   0.51    0.44    0.83    0.60 |  kcenter   0.42  kcenter  kcenter        
  18  2 |   0.97   0.25    0.22    1.41    0.26 |  kcenter   0.72  kcenter  kcenter        
  18  3 |   1.01   0.24    0.19    0.63    0.26 |  kcenter   0.76  kcenter  kcenter        
  18  4 |   0.90   0.37    0.28    1.81    0.34 |  kcenter   0.53  kcenter  kcenter        
  18  5 |   0.92   0.18    0.17    1.19    0.18 |  kcenter   0.73  kcenter  kcenter        
  18  6 |   0.91   0.18    0.12    1.28    0.16 |  kcenter   0.73  kcenter  kcenter        
  18  7 |   0.87   0.48    0.37    1.77    0.38 |  kcenter   0.39  kcenter  kcenter        
  19  0 |   0.92   0.15    0.08    1.56    0.10 |  kcenter   0.77  kcenter  kcenter        
  19  1 |   1.08   0.32    0.20    0.57    0.33 |  kcenter   0.75  kcenter  kcenter        
  19  2 |   1.04   0.38    0.31    1.59    0.37 |  kcenter   0.66  kcenter  kcenter        
  19  3 |   1.03   0.40    0.29    1.61    0.35 |  kcenter   0.63  kcenter  kcenter        
  19  4 |   0.98   0.36    0.33    1.48    0.34 |  kcenter   0.62  kcenter  kcenter        
  19  5 |   0.33   0.19    0.07    1.16    0.12 |  kcenter   0.14  kcenter  kcenter        
  19  6 |   0.91   0.07    0.02    1.48    0.05 |  kcenter   0.84  kcenter  kcenter        
  19  7 |   1.12   0.21    0.14    1.27    0.23 |  kcenter   0.91  kcenter  kcenter        
  20  0 |   1.22   0.26    0.18    1.29    0.23 |  kcenter   0.96  kcenter  kcenter        
  20  1 |   1.00   0.08    0.06    0.95    0.08 |  kcenter   0.92  kcenter  kcenter        
  20  2 |   0.98   0.33    0.25    1.21    0.28 |  kcenter   0.66  kcenter  kcenter        
  20  3 |   1.01   0.16    0.11    0.44    0.12 |  kcenter   0.85  kcenter  kcenter        
  20  4 |   0.79   0.21    0.17    1.28    0.20 |  kcenter   0.58  kcenter  kcenter        
  20  5 |   1.02   0.27    0.29    2.27    0.41 |  kcenter   0.75  kcenter  kcenter        
  20  6 |   0.90   0.29    0.21    1.35    0.29 |  kcenter   0.61  kcenter  kcenter        
  20  7 |   1.00   0.16    0.14    1.25    0.16 |  kcenter   0.84  kcenter  kcenter        
  21  0 |   0.94   0.24    0.19    1.34    0.24 |  kcenter   0.70  kcenter  kcenter        
  21  1 |   1.05   0.24    0.13    0.98    0.23 |  kcenter   0.82  kcenter  kcenter        
  21  2 |   1.09   0.19    0.13    0.49    0.17 |  kcenter   0.90  kcenter  kcenter        
  21  3 |   0.75   0.09    0.06    1.66    0.07 |  kcenter   0.66  kcenter  kcenter        
  21  4 |   1.02   0.50    0.16    1.11    0.23 |  kcenter   0.52  kcenter  kcenter        
  21  5 |   0.93   0.20    0.12    1.26    0.21 |  kcenter   0.73  kcenter  kcenter        
  21  6 |   0.09   0.18    0.05    1.61    0.07 | facility   0.09 facility facility        
  21  7 |   0.06   0.05    0.02    1.31    0.02 |  kcenter   0.00  kcenter  kcenter        
  22  0 |   0.91   0.22    0.11    0.32    0.15 |  kcenter   0.69  kcenter  kcenter        
  22  1 |   1.08   0.20    0.14    1.60    0.20 |  kcenter   0.88  kcenter  kcenter        
  22  2 |   1.09   0.19    0.15    1.50    0.17 |  kcenter   0.91  kcenter  kcenter        
  22  3 |   0.09   0.10    0.02    1.50    0.03 | facility   0.01 facility  kcenter    FLIP
  22  4 |   1.02   0.25    0.18    0.66    0.23 |  kcenter   0.77  kcenter  kcenter        
  22  5 |   1.11   0.48    0.07    1.34    0.12 |  kcenter   0.63  kcenter  kcenter        
  22  6 |   0.92   0.09    0.02    0.86    0.09 |  kcenter   0.82  kcenter  kcenter        
  22  7 |   0.95   0.16    0.15    1.45    0.17 |  kcenter   0.80  kcenter  kcenter        
  23  0 |   0.26   0.18    0.04    2.32    0.05 |  kcenter   0.08 facility  kcenter    FLIP
  23  1 |   0.93   0.22    0.08    0.19    0.21 |  kcenter   0.71  kcenter  kcenter        
  23  2 |   1.00   0.18    0.08    0.23    0.17 |  kcenter   0.83  kcenter  kcenter        
  23  3 |   0.79   0.16    0.12    1.07    0.17 |  kcenter   0.63  kcenter  kcenter        
  23  4 |   0.72   0.07    0.04    2.03    0.05 |  kcenter   0.65  kcenter  kcenter        
  23  5 |   0.57   0.11    0.02    1.76    0.03 |  kcenter   0.46  kcenter  kcenter        
  23  6 |   0.63   0.15    0.06    1.81    0.15 |  kcenter   0.48  kcenter  kcenter        
  23  7 |   0.93   0.21    0.04    0.98    0.07 |  kcenter   0.72  kcenter  kcenter        
  24  0 |   0.95   0.11    0.05    2.49    0.07 |  kcenter   0.85  kcenter  kcenter        
  24  1 |   1.06   0.23    0.16    1.03    0.24 |  kcenter   0.83  kcenter  kcenter        
  24  2 |   0.85   0.08    0.05    0.68    0.09 |  kcenter   0.77  kcenter  kcenter        
  24  3 |   0.85   0.27    0.17    1.03    0.26 |  kcenter   0.58  kcenter  kcenter        
  24  4 |   0.97   0.19    0.10    0.65    0.18 |  kcenter   0.78  kcenter  kcenter        
  24  5 |   0.93   0.17    0.07    0.50    0.17 |  kcenter   0.75  kcenter  kcenter        
  24  6 |   1.23   0.15    0.05    1.09    0.08 |  kcenter   1.08  kcenter  kcenter        
  24  7 |   0.21   0.06    0.01    1.27    0.06 |  kcenter   0.15  kcenter  kcenter        
  25  0 |   0.06   0.08    0.01    1.71    0.04 | facility   0.02 facility facility        
  25  1 |   1.20   0.10    0.03    1.87    0.11 |  kcenter   1.09  kcenter  kcenter        
  25  2 |   0.87   0.22    0.15    1.29    0.20 |  kcenter   0.66  kcenter  kcenter        
  25  3 |   1.07   0.12    0.09    1.46    0.12 |  kcenter   0.95  kcenter  kcenter        
  25  4 |   1.09   0.21    0.17    1.14    0.24 |  kcenter   0.88  kcenter  kcenter        
  25  5 |   0.57   0.11    0.03    0.26    0.11 |  kcenter   0.46  kcenter  kcenter        
  25  6 |   1.02   0.14    0.05    0.86    0.13 |  kcenter   0.88  kcenter  kcenter        
  25  7 |   0.94   0.15    0.11    1.48    0.15 |  kcenter   0.79  kcenter  kcenter        
  26  0 |   0.25   0.14    0.02    0.01    0.07 |  kcenter   0.11 facility  kcenter    FLIP
  26  1 |   0.95   0.18    0.14    1.08    0.23 |  kcenter   0.77  kcenter  kcenter        
  26  2 |   1.02   0.11    0.07    0.24    0.11 |  kcenter   0.90  kcenter  kcenter        
  26  3 |   1.15   0.23    0.11    0.63    0.18 |  kcenter   0.92  kcenter  kcenter        
  26  4 |   0.92   0.25    0.17    1.39    0.28 |  kcenter   0.67  kcenter  kcenter        
  26  5 |   0.86   0.14    0.10    1.56    0.13 |  kcenter   0.72  kcenter  kcenter        
  26  6 |   1.08   0.17    0.09    0.59    0.19 |  kcenter   0.91  kcenter  kcenter        
  26  7 |   0.69   0.14    0.08    1.92    0.11 |  kcenter   0.55  kcenter  kcenter        
  27  0 |   0.18   0.16    0.02    0.19    0.06 |  kcenter   0.03  kcenter  kcenter        
  27  1 |   1.40   0.10    0.02    0.90    0.05 |  kcenter   1.29  kcenter  kcenter        
  27  2 |   0.65   0.08    0.01    0.88    0.03 |  kcenter   0.57  kcenter  kcenter        
  27  3 |   0.77   0.35    0.13    0.76    0.37 |  kcenter   0.42  kcenter  kcenter        
  27  4 |   1.07   0.13    0.04    1.52    0.09 |  kcenter   0.94  kcenter  kcenter        
  27  5 |   0.31   0.14    0.05    0.29    0.15 |  kcenter   0.17  kcenter  kcenter        
  27  6 |   1.13   0.25    0.12    1.78    0.24 |  kcenter   0.89  kcenter  kcenter        
  27  7 |   0.95   0.12    0.05    0.45    0.10 |  kcenter   0.83  kcenter  kcenter        
  28  0 |   0.96   0.21    0.13    1.61    0.18 |  kcenter   0.76  kcenter  kcenter        
  28  1 |   0.36   0.26    0.11    2.02    0.20 |  kcenter   0.09  kcenter  kcenter        
  28  2 |   0.75   0.15    0.07    1.54    0.11 |  kcenter   0.59  kcenter  kcenter        
  28  3 |   1.03   0.36    0.34    1.58    0.39 |  kcenter   0.67  kcenter  kcenter        
  28  4 |   1.23   0.24    0.16    1.41    0.29 |  kcenter   0.99  kcenter  kcenter        
  28  5 |   0.89   0.27    0.09    0.94    0.17 |  kcenter   0.62  kcenter  kcenter        
  28  6 |   1.26   0.41    0.28    1.68    0.41 |  kcenter   0.85  kcenter  kcenter        
  28  7 |   1.14   0.20    0.08    0.77    0.12 |  kcenter   0.94  kcenter  kcenter        
  29  0 |   1.26   0.30    0.20    0.53    0.30 |  kcenter   0.96  kcenter  kcenter        
  29  1 |   0.04   0.04    0.01    1.02    0.03 |  kcenter   0.00  kcenter  kcenter        
  29  2 |   1.15   0.42    0.20    1.33    0.37 |  kcenter   0.72  kcenter  kcenter        
  29  3 |   1.01   0.46    0.31    1.25    0.50 |  kcenter   0.55  kcenter  kcenter        
  29  4 |   0.88   0.31    0.15    1.40    0.27 |  kcenter   0.58  kcenter  kcenter        
  29  5 |   1.19   0.22    0.08    1.16    0.22 |  kcenter   0.97  kcenter  kcenter        
  29  6 |   1.12   0.23    0.03    0.28    0.13 |  kcenter   0.89  kcenter  kcenter        
  29  7 |   0.50   0.19    0.08    1.19    0.12 |  kcenter   0.31  kcenter  kcenter        
  30  0 |   0.91   0.37    0.22    1.33    0.36 |  kcenter   0.53  kcenter  kcenter        
  30  1 |   1.00   0.35    0.22    1.75    0.35 |  kcenter   0.65  kcenter  kcenter        
  30  2 |   0.89   0.34    0.33    1.54    0.37 |  kcenter   0.55  kcenter  kcenter        
  30  3 |   0.94   0.37    0.39    2.19    0.42 |  kcenter   0.56  kcenter  kcenter        
  30  4 |   0.97   0.47    0.35    0.94    0.43 |  kcenter   0.50  kcenter  kcenter        
  30  5 |   0.65   0.20    0.09    2.24    0.17 |  kcenter   0.45  kcenter facility    FLIP
  30  6 |   0.94   0.35    0.11    1.10    0.30 |  kcenter   0.58  kcenter  kcenter        
  30  7 |   1.08   0.41    0.14    1.50    0.24 |  kcenter   0.67  kcenter  kcenter        
  31  0 |   0.88   1.01    1.12    1.70    1.03 | facility   0.12 facility facility        
  31  1 |   0.83   0.32    0.30    1.52    0.31 |  kcenter   0.51  kcenter  kcenter        
  31  2 |   1.03   0.26    0.17    2.08    0.25 |  kcenter   0.77  kcenter  kcenter        
  31  3 |   0.90   0.26    0.17    1.30    0.26 |  kcenter   0.63  kcenter  kcenter        
  31  4 |   1.02   0.43    0.29    1.25    0.31 |  kcenter   0.59  kcenter  kcenter        
  31  5 |   0.88   0.34    0.23    2.38    0.34 |  kcenter   0.54  kcenter  kcenter        
  31  6 |   0.92   0.49    0.32    1.14    0.48 |  kcenter   0.43  kcenter  kcenter        
  31  7 |   1.09   0.24    0.18    1.41    0.26 |  kcenter   0.85  kcenter  kcenter        

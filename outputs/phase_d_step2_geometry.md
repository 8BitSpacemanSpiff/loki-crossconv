# Phase D STEP 2 — geometry predictors g_h (centered K_pre)

Per head (avg over sequences): PR (participation ratio), spectral_tail (1 - lambda_max/trace), outlier_fraction (>med+3MAD dist-to-centroid), clusteredness (2-means variance reduction). NOT fit yet -- raw predictors.

## winner=facility  (23 heads)
  PR                mean 19.207  [min 1.431 med 20.494 max 45.594]
  spectral_tail     mean 0.772  [min 0.170 med 0.845 max 0.950]
  outlier_fraction  mean 0.026  [min 0.000 med 0.017 max 0.151]
  clusteredness     mean 0.167  [min 0.034 med 0.116 max 0.543]

## winner=kcenter  (232 heads)
  PR                mean 27.180  [min 1.528 med 28.064 max 48.819]
  spectral_tail     mean 0.874  [min 0.206 med 0.890 max 0.941]
  outlier_fraction  mean 0.012  [min 0.000 med 0.008 max 0.263]
  clusteredness     mean 0.093  [min 0.043 med 0.078 max 0.755]

## class separation (mean g_h | facility-win vs kcenter-win)
  g_h                 facility    kcenter   |diff|
  PR                    19.207     27.180    7.973
  spectral_tail          0.772      0.874    0.102
  outlier_fraction       0.026      0.012    0.014
  clusteredness          0.167      0.093    0.074

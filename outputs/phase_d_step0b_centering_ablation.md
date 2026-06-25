# Phase D STEP 0b — centering ablation on the selectors

40 heads (layers 0/8/16/24/31 x 8 KV), keep=3%, n_cap=1024, 2-seq avg. valid=39, excluded-diffuse=1 (L0/kv4).

## Translation-invariance check (selection set A == B, all 40 heads)
  facility identical A==B: True   kcenter identical A==B: True
  keydiff identical A==B: 0/40   logdet identical A==B: 0/40
  -> Euclidean coverage selectors are offset-INVARIANT by construction; only the inner-product selectors (keydiff cosine, logdet leverage) move under de-meaning.

## 3-way winner tally {keydiff, facility, kcenter} (valid heads)
  (A) uncentered selectors: {'facility': 16, 'keydiff': 17, 'kcenter': 6}
  (B) centered   selectors: {'facility': 24, 'kcenter': 13, 'keydiff': 2}
  per-head winner agreement A->B: 24/39 stable, 15 flip(s)
    L 0 kv1: keydiff -> facility  (keydiff 0.01->73.66, facility 0.61, kcenter 2.83)
    L 0 kv3: keydiff -> facility  (keydiff 0.01->60.48, facility 0.90, kcenter 2.08)
    L 8 kv1: keydiff -> kcenter  (keydiff 0.38->1.18, facility 0.96, kcenter 0.61)
    L 8 kv5: keydiff -> facility  (keydiff 0.67->1.12, facility 0.81, kcenter 1.14)
    L 8 kv6: keydiff -> kcenter  (keydiff 0.50->1.77, facility 0.76, kcenter 0.62)
    L16 kv0: keydiff -> kcenter  (keydiff 0.54->0.70, facility 0.81, kcenter 0.56)
    L16 kv2: keydiff -> kcenter  (keydiff 0.42->1.12, facility 0.79, kcenter 0.58)
    L16 kv5: keydiff -> kcenter  (keydiff 0.22->1.08, facility 0.93, kcenter 0.48)
    L16 kv7: keydiff -> facility  (keydiff 0.66->1.39, facility 0.77, kcenter 0.86)
    L24 kv0: keydiff -> facility  (keydiff 0.47->2.16, facility 0.56, kcenter 1.14)
    L24 kv5: keydiff -> kcenter  (keydiff 0.57->1.41, facility 1.13, kcenter 0.75)
    L24 kv6: keydiff -> kcenter  (keydiff 0.02->0.67, facility 1.15, kcenter 0.04)
    L31 kv3: keydiff -> facility  (keydiff 0.39->0.71, facility 0.65, kcenter 1.16)
    L31 kv5: keydiff -> facility  (keydiff 0.52->1.87, facility 0.95, kcenter 1.27)
    L31 kv6: keydiff -> facility  (keydiff 0.63->1.00, facility 0.73, kcenter 1.05)

## Margin distribution (|best - 2nd| in normalised-Δ units)
  (A) min 0.001 p25 0.098 med 0.204 p75 0.414 max 1.582   near-tie(<0.05): 7/39
  (B) min 0.001 p25 0.204 med 0.369 p75 0.576 max 5.440   near-tie(<0.05): 2/39

## keydiff: normalised-Δ shift when centered (the only menu selector that moves)
  |Δnorm_A - Δnorm_B|: min 0.004 med 0.691 max 73.655
  centered keydiff BETTER on 4/39 heads, WORSE on 34/39 (|change|>0.02)

## logdet centered vs uncentered (was it scoring on the sink axis?)
  mean normalised Δ: uncentered 1.395, centered 1.076 (oracle 0, random 1)
  heads below random floor (Δ<0.85): uncentered 2/39, centered 14/39
  heads logdet would WIN outright vs menu: uncentered 0/39, centered 5/39
  -> centering RECOVERS logdet on some heads: it was scoring on the sink axis

## per-head (valid + diffuse flagged)
   L kv |   kd_A   kd_B    fac     kc   ld_A   ld_B |     winA     winB    mA    mB    flag
   0  0 |   2.23   1.43   0.58   1.78   0.66   1.68 | facility facility  1.20  0.85        
   0  1 |   0.01  73.66   0.61   2.83   0.37   3.59 |  keydiff facility  0.60  2.22    FLIP
   0  2 |   1.61  19.08   2.58   0.02   2.40   0.34 |  kcenter  kcenter  1.58  2.55        
   0  3 |   0.01  60.48   0.90   2.08   3.93   1.86 |  keydiff facility  0.90  1.18    FLIP
   0  4 |  52.40  52.99  -1.07  -1.37  -0.05  -0.19 |  kcenter  kcenter  0.30  0.30 DIFFUSE
   0  5 |   0.03  -4.73   2.37   0.71   1.83   1.41 |  keydiff  keydiff  0.68  5.44        
   0  6 |   0.78   1.93   0.75   0.79   1.19   0.59 | facility facility  0.03  0.04        
   0  7 |   1.78   3.47   0.67   0.47   1.24   0.86 |  kcenter  kcenter  0.20  0.20        
   8  0 |   1.20   1.77   1.00   1.39   1.37   1.53 | facility facility  0.19  0.39        
   8  1 |   0.38   1.18   0.96   0.61   1.06   0.74 |  keydiff  kcenter  0.24  0.34    FLIP
   8  2 |   1.29   1.72   0.72   1.08   1.27   1.22 | facility facility  0.36  0.36        
   8  3 |   0.97   0.87   0.81   1.34   1.17   1.29 | facility facility  0.16  0.07        
   8  4 |   0.68   1.41   0.93   0.64   1.37   0.67 |  kcenter  kcenter  0.03  0.29        
   8  5 |   0.67   1.12   0.81   1.14   1.20   1.15 |  keydiff facility  0.14  0.31    FLIP
   8  6 |   0.50   1.77   0.76   0.62   1.12   0.48 |  keydiff  kcenter  0.12  0.15    FLIP
   8  7 |   2.79   3.59   0.52   1.55   1.60   1.68 | facility facility  1.03  1.03        
  16  0 |   0.54   0.70   0.81   0.56   1.32   0.70 |  keydiff  kcenter  0.02  0.14    FLIP
  16  1 |   0.97   1.05   0.75   1.08   1.13   1.03 | facility facility  0.22  0.31        
  16  2 |   0.42   1.12   0.79   0.58   1.12   0.70 |  keydiff  kcenter  0.15  0.21    FLIP
  16  3 |   0.99   1.22   0.81   1.35   1.43   1.35 | facility facility  0.19  0.42        
  16  4 |   1.09   1.17   0.69   1.35   1.34   1.42 | facility facility  0.39  0.48        
  16  5 |   0.22   1.08   0.93   0.48   1.30   0.21 |  keydiff  kcenter  0.27  0.45    FLIP
  16  6 |   0.94   0.90   0.85   0.91   1.10   1.04 | facility facility  0.06  0.05        
  16  7 |   0.66   1.39   0.77   0.86   1.32   0.93 |  keydiff facility  0.11  0.08    FLIP
  24  0 |   0.47   2.16   0.56   1.14   1.78   1.13 |  keydiff facility  0.09  0.58    FLIP
  24  1 |   0.59   0.66   0.92   1.30   1.49   0.60 |  keydiff  keydiff  0.33  0.27        
  24  2 |   1.07   1.84   0.83   1.23   1.45   1.53 | facility facility  0.24  0.40        
  24  3 |   0.44   0.89   0.98   0.41   1.09   0.37 |  kcenter  kcenter  0.03  0.48        
  24  4 |   1.12   1.28   0.71   1.19   1.13   1.01 | facility facility  0.41  0.48        
  24  5 |   0.57   1.41   1.13   0.75   1.50   0.77 |  keydiff  kcenter  0.18  0.38    FLIP
  24  6 |   0.02   0.67   1.15   0.04   1.40   0.05 |  keydiff  kcenter  0.03  0.63    FLIP
  24  7 |   0.56   1.49   0.89   0.39   1.16   0.41 |  kcenter  kcenter  0.17  0.50        
  31  0 |   0.65   1.03   0.73   0.62   1.25   0.73 |  kcenter  kcenter  0.03  0.10        
  31  1 |   1.32   1.32   0.95   1.61   1.70   1.84 | facility facility  0.37  0.37        
  31  2 |   1.54   2.17   0.82   0.83   1.26   0.90 | facility facility  0.00  0.00        
  31  3 |   0.39   0.71   0.65   1.16   1.44   1.05 |  keydiff facility  0.26  0.06    FLIP
  31  4 |   1.14   1.21   0.59   1.28   1.38   1.34 | facility facility  0.55  0.62        
  31  5 |   0.52   1.87   0.95   1.27   1.26   1.23 |  keydiff facility  0.43  0.32    FLIP
  31  6 |   0.63   1.00   0.73   1.05   1.00   1.04 |  keydiff facility  0.10  0.28    FLIP
  31  7 |   0.99   1.78   0.52   1.48   2.23   1.52 | facility facility  0.47  0.96        

## Findings (decision deferred to user -- a deploy-vs-content fork)
  1. Coverage selectors (facility, kcenter) are offset-INVARIANT (identical sets A==B); their Phase C wins are REAL, not an uncentered artifact.
  2. Winners shift substantially A->B (15/39 flips) -- ALL driven by the two inner-product selectors. keydiff collapses (17->2 wins; worse on 34/39 heads), logdet partially recovers (0->5 outright wins, 2->14 heads off the floor). Both were dominated by the shared sink axis uncentered.
  3. On de-meaned CONTENT geometry, coverage dominates (facility 24, kcenter 13, keydiff 2) and margins are cleaner (near-tie 2/39 vs 7/39).

  FORK for STEP 1 (user decides):
   (i) CONTENT view -- run all selectors centered. Coverage dominates; keydiff/logdet are weak; the routing question is facility-vs-kcenter. Phase C's keydiff wins are superseded.
   (ii) DEPLOY view -- KeyDiff ships on RAW keys (the sink is present at inference), so its uncentered strength (17 wins) is its true performance; centering it is a strawman. Keep keydiff uncentered as the deployable baseline; facility/kcenter are invariant either way.
  The two are not exclusive: the principled diagnostic is centered, but the baseline to BEAT is uncentered KeyDiff. STEP 1 setup depends on which framing the paper takes.

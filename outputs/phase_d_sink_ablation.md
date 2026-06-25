# Phase D sink ablation — is KeyDiff's win real or sink-riding?

All 256 heads, n_seq=8, full cloud. Protect positions [0, 1, 2, 3] (always-keep) equally for keydiff_unc/facility/kcenter. Raw attention-output Δ (winner/beats are normalization-invariant). Δ on real K_post/V.

## budget 3%
  winner {facility,kcenter}  UNPROTECTED: {'facility': 27, 'kcenter': 229}
  winner {facility,kcenter}  SINK-PROT  : {'facility': 68, 'kcenter': 188}
  BEATS keydiff_unc (routed min<keydiff)  UNPROTECTED: 35/256
  BEATS keydiff_unc (routed min<keydiff)  SINK-PROT  : 64/256

## budget 25%
  winner {facility,kcenter}  UNPROTECTED: {'facility': 165, 'kcenter': 91}
  winner {facility,kcenter}  SINK-PROT  : {'facility': 93, 'kcenter': 163}
  BEATS keydiff_unc (routed min<keydiff)  UNPROTECTED: 30/256
  BEATS keydiff_unc (routed min<keydiff)  SINK-PROT  : 36/256

## Mechanism — heads where keydiff_unc WINS unprotected @ 3% (221/256)

  mean sink_mass (eval-query attention on pos 0-3): 0.539
  sinks RETAINED (of 4) @ 3%:  keydiff 3.02  facility 0.27  kcenter 1.37
  fraction of keydiff's RETAINED attention mass on the sinks: 0.683

  (for contrast, ALL heads:)
  mean sink_mass all heads: 0.531; keydiff sinks-kept 2.96, facility 0.32, kcenter 1.44

## Verdict
  @ 3%: coverage beats keydiff on 35/256 (13%) unprotected -> 64/256 (25%) sink-protected.
  DOMINANCE SURVIVES (sink-augmented): sink-protection roughly DOUBLES coverage's wins (35->64/256, +29) -- KeyDiff demonstrably rides the sinks facility evicts -- but KeyDiff still beats the coverage menu on 192/256 (75%) heads with sinks controlled. The Phase D baseline was PARTLY unfair (fix: sink-protect all selectors), yet distinctiveness still wins on the majority. Reframe toward distinctiveness; coverage is NOT competitive even after the sink fix.

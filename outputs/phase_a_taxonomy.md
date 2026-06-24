# Phase A — corrected selector taxonomy

Genuine diversity poles (logdet = k-DPP MAP, keydiff = KeyDiff-proper) built and
scored against the coverage pole (facility) and the k-center coverage reference (FPS).

## Decisive test & routing viability
```
Peaked-head win table (valid oracle); winner per cell = lowest normΔ:
    needle  aggressive -> winner: logdet
    needle  moderate   -> winner: kcenter
    bulk    aggressive -> winner: logdet
    bulk    moderate   -> winner: kcenter

Decisive moderate-budget test (bulk head): each diversity pole vs facility
    logdet  vs facility: -0.4185  -> beats facility (diversity better even at moderate budget)
    keydiff vs facility: +7.9661  -> worse than facility
    kcenter vs facility: -0.4526  -> beats facility (diversity better even at moderate budget)

Cells where COVERAGE (facility) clearly beats best DIVERSITY (logdet): NONE
Distinct cell-winners across peaked heads: ['kcenter', 'logdet']
```

## Recommendation
```
LIKELY ROUTING NO-GO signal (synthetic): the coverage pole (facility) never beats the best diversity pole (logdet) on any peaked synthetic head/budget; logdet is near-oracle everywhere. If this holds on real heads there is nothing to route between diversity and coverage. This is a FINDING to confirm cheaply on real heads (Phase C) before any heavy GPU -- not a failure.
Best diversity pole on the needle cloud (its home turf): 'logdet' (normΔ +0.0011, needle_keep 6.2/8); facility drops needles (1.3/8).
keydiff (KeyDiff-proper) is catastrophic on every synthetic cloud, but the clouds are origin-centred (cosine-to-anchor degenerate); treat this as a synthetic artifact, NOT a verdict -- keydiff is the mandatory Stage 3 baseline and needs a real-data score.
PROPOSED Stage 1 menu (carry exactly 3): diversity-pole='logdet', baseline='keydiff', coverage-pole='facility'. Drop kcenter from the GPU sweep (it is the min-max-coverage reference and tracks logdet closely).
```

## Detail (12 seeds each)

### cloud=needle  budget=aggressive (keep=5%)  attn-entropy~0.33
    selector    normΔ[0=oracle,1=random]   needle_keep   gap-vs-facility
    logdet       +0.0011                 6.2/8       -0.6900  <- tracks oracle
    keydiff      +0.8746                 5.4/8       +0.1835
    kcenter      +0.0082                 6.2/8       -0.6829
    facility     +0.6911                 1.3/8       +0.0000
    best diversity pole here: logdet

### cloud=needle  budget=moderate (keep=50%)  attn-entropy~0.33
    selector    normΔ[0=oracle,1=random]   needle_keep   gap-vs-facility
    logdet       +0.0264                 8.0/8       -0.0273
    keydiff      +1.0546                 7.3/8       +1.0009
    kcenter      +0.0252                 8.0/8       -0.0286  <- tracks oracle
    facility     +0.0537                 8.0/8       +0.0000
    best diversity pole here: kcenter

### cloud=skewed  budget=aggressive (keep=5%)  attn-entropy~3.64
    selector    normΔ[0=oracle,1=random]   needle_keep   gap-vs-facility
    logdet       -0.1378                   -         -0.1886  <- tracks oracle
    keydiff      +2.2692                   -         +2.2184
    kcenter      -0.0227                   -         -0.0735
    facility     +0.0508                   -         +0.0000
    best diversity pole here: logdet

### cloud=skewed  budget=moderate (keep=50%)  attn-entropy~3.64
    selector    normΔ[0=oracle,1=random]   needle_keep   gap-vs-facility
    logdet       -0.4598                   -         -0.0793  <- tracks oracle
    keydiff     +13.3065                   -        +13.6870
    kcenter      -0.4101                   -         -0.0296
    facility     -0.3805                   -         +0.0000
    best diversity pole here: logdet

### cloud=bulk  budget=aggressive (keep=5%)  attn-entropy~0.45
    selector    normΔ[0=oracle,1=random]   needle_keep   gap-vs-facility
    logdet       +0.3274                   -         -0.2896  <- tracks oracle
    keydiff      +6.9754                   -         +6.3584
    kcenter      +0.5203                   -         -0.0967
    facility     +0.6170                   -         +0.0000
    best diversity pole here: logdet

### cloud=bulk  budget=moderate (keep=50%)  attn-entropy~0.45
    selector    normΔ[0=oracle,1=random]   needle_keep   gap-vs-facility
    logdet       +0.3693                   -         -0.4185
    keydiff      +8.7539                   -         +7.9661
    kcenter      +0.3352                   -         -0.4526  <- tracks oracle
    facility     +0.7879                   -         +0.0000
    best diversity pole here: kcenter


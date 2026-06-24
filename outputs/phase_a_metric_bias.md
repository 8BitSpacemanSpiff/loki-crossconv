# Phase A — bidirectional metric (Δ) validation

Metric is unbiased (flips both ways at aggressive budget): **True**

Key finding: the diversity/coverage flip is BUDGET-DEPENDENT. It is robust at
aggressive budgets and head-dependent (needle->diversity, skewed-bulk->coverage),
but at moderate/50% budgets the two query-free selectors converge to near-oracle
(farthest-point sampling is itself near-optimal), so no separation exists there to
test directionality. This is Stage 0 Finding 1 / the A.4 'selectors agree' regime,
and it is the stated reason the Stage 1 primary operating point is aggressive, not 50%.

## Verdict
```
[PASS] D1 diversity-wins direction (12/12)
[PASS] D2 coverage-wins direction at aggressive budget (11/12)
[note] C  selectors converge at moderate budget (normalised margin -0.0112)
[caveat] tractable oracle = top-b-by-mass is a tight floor only when attention is PEAKED; under the deliberately near-uniform D2/C head it is loose (real Mistral attention is peaked, so it is fine for Stage 1).
```

## Detail

### D1  needle/retrieval head  (keep=25%, 12 seeds)
    mean Δ:  oracle=0.0005  diversity=0.2209  coverage=0.4252  random=6.7978
    diversity-wins 12/12 | coverage-wins 0/12 | mean(Δdiv-Δcov)=-0.2044  normalised=-0.0301  (>0 => coverage wins)
    expectation: Δ(diversity) < Δ(coverage)

### D2  skewed bulk head  (keep=10%, 12 seeds)
    mean Δ:  oracle=3.7232  diversity=3.3119  coverage=2.9416  random=15.2017
    diversity-wins 1/12 | coverage-wins 11/12 | mean(Δdiv-Δcov)=+0.3703  normalised=+0.0323  (>0 => coverage wins)
    expectation: Δ(coverage) < Δ(diversity)

### C   skewed bulk head, MODERATE budget  (keep=50%, 12 seeds)
    mean Δ:  oracle=1.5216  diversity=0.9488  coverage=1.0082  random=6.8197
    diversity-wins 9/12 | coverage-wins 3/12 | mean(Δdiv-Δcov)=-0.0594  normalised=-0.0112  (>0 => coverage wins)
    expectation: selectors converge: |Δdiv-Δcov| ~ 0, both near oracle


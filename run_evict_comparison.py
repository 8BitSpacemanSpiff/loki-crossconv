#!/usr/bin/env python3
"""
Compare eviction methods at a MATCHED cache budget. Shells out to evaluate_tasks.py
once per method, parses PPL + the *_mass_kept diagnostics, prints a table + JSON.

Methods:
  evict     CrossCov R_q-weighted score-energy   (ours; needs basis + Rq calibration)
  keynorm   ||k||^2                               (control: Rq = I limit; no calibration)
  keydiff   cosine-to-anchor diversity           (baseline arXiv 2504.15364; no calibration)
  stream    sinks + recent only                  (floor; emulated as keynorm with evict_ratio=1
                                                   minus protection -> see --modes)

Budget matching: pass --keep-tokens N to fix an absolute KV budget for ALL methods
(this is KeyDiff's protocol), or --evict-ratio for a fractional budget. Protection
(--sink-tokens, --recent-window) is identical across methods.

Llama-3.1-8B-Instruct example (after one calibration pass producing the basis + Rq):
  python run_evict_comparison.py \
      --model-id meta-llama/Llama-3.1-8B-Instruct --model-type llama \
      --dataset wikitext-test --sequence-length 8192 \
      --top-r 32 --keep-tokens 2048 \
      --rotary-type prerotary --transform-dataset c4 \
      --sink-tokens 16 --recent-window 64 \
      --out /mnt/user-data/outputs/evict_llama31.json
"""
import argparse
import json
import re
import subprocess
import sys

PPL_RE = re.compile(r"tensor\(([-+0-9.eE]+)(?:[,)]|$)|^([-+0-9.]+)\s*$", re.M)
DIAG_RE = re.compile(r"^(\w+):\s+overall=([-+0-9.eE]+)\s+layers=(\d+)\s+samples=(\d+)", re.M)

# methods that need the offline CrossCov basis + Rq
NEEDS_CALIB = {"evict"}


def base_cmd(args):
    return [
        sys.executable, "evaluate_tasks.py",
        "--model-id", args.model_id,
        "--model-type", args.model_type,
        "--sequence-length", str(args.sequence_length),
        "--dataset", args.dataset,
        "--use-pca-topk", "--use-crosscov",
        "--top-r", str(args.top_r),
        "--rotary-type", args.rotary_type,
        "--transform-dataset", args.transform_dataset,
        "--sink-tokens", str(args.sink_tokens),
        "--recent-window", str(args.recent_window),
        "--quiet-diagnostics", "--log-mass-recall",
    ]


def mode_cmd(args, mode):
    cmd = base_cmd(args) + ["--crosscov-mode", mode]
    if args.keep_tokens > 0:
        cmd += ["--keep-tokens", str(args.keep_tokens)]
    else:
        cmd += ["--evict-ratio", str(args.evict_ratio)]
    if args.extra:
        cmd += args.extra
    return cmd


def parse_output(text):
    ppl = None
    for m in PPL_RE.finditer(text):
        val = m.group(1) or m.group(2)
        if val:
            try:
                ppl = float(val)
            except ValueError:
                pass
    diags = {m.group(1): float(m.group(2)) for m in DIAG_RE.finditer(text)}
    return ppl, diags


def run(cmd):
    print("  $ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("  [stderr tail]\n" + "\n".join(p.stderr.splitlines()[-15:]))
    return p.stdout + "\n" + p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--model-type", default="llama")
    ap.add_argument("--dataset", default="wikitext-test")
    ap.add_argument("--sequence-length", type=int, default=8192)
    ap.add_argument("--top-r", type=float, default=32)
    ap.add_argument("--evict-ratio", type=float, default=0.5)
    ap.add_argument("--keep-tokens", type=int, default=0, help="absolute budget; matches KeyDiff")
    ap.add_argument("--rotary-type", default="prerotary")
    ap.add_argument("--transform-dataset", default="c4")
    ap.add_argument("--sink-tokens", type=int, default=16)
    ap.add_argument("--recent-window", type=int, default=64)
    ap.add_argument("--modes", default="evict,keynorm,keydiff")
    ap.add_argument("--out", default="evict_comparison.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = {}
    for mode in modes:
        cmd = mode_cmd(args, mode)
        tag = mode + (" (needs calib)" if mode in NEEDS_CALIB else " (no calib)")
        if args.dry_run:
            print(f"[{tag}]\n  $ " + " ".join(cmd) + "\n")
            continue
        print(f"[{tag}] running...", flush=True)
        out = run(cmd)
        ppl, diags = parse_output(out)
        results[mode] = {"ppl": ppl, "diagnostics": diags}
        print(f"[{mode}] ppl={ppl} diags={diags}\n", flush=True)

    if args.dry_run:
        return

    # the mass-kept metric has a method-specific key (evict_mass_kept, keydiff_mass_kept,
    # keynorm_mass_kept); surface them under one column for the table.
    def mass_of(mode):
        d = results[mode]["diagnostics"]
        for k in (f"{mode}_mass_kept", "evict_mass_kept", "keydiff_mass_kept", "keynorm_mass_kept"):
            if k in d:
                return d[k]
        return None

    budget = f"{args.keep_tokens} tokens" if args.keep_tokens > 0 else f"evict_ratio={args.evict_ratio}"
    col = max(10, *(len(m) for m in modes))
    print(f"\nBudget: {budget} | protect: sink={args.sink_tokens} recent={args.recent_window}\n")
    header = "metric".ljust(16) + "".join(m.rjust(col + 2) for m in modes)
    print(header); print("-" * len(header))
    print("mass_kept".ljust(16) + "".join(
        (f"{mass_of(m):.4f}" if mass_of(m) is not None else "-").rjust(col + 2) for m in modes))
    print("ppl".ljust(16) + "".join(
        (f"{results[m]['ppl']:.4f}" if results[m]['ppl'] is not None else "n/a").rjust(col + 2) for m in modes))

    meta = {"budget": budget, "top_r": args.top_r, "model_id": args.model_id,
            "sequence_length": args.sequence_length,
            "protect": {"sink": args.sink_tokens, "recent": args.recent_window},
            "results": results}
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

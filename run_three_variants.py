#!/usr/bin/env python3
"""
Run the three CrossCov variants at a MATCHED operating point and dump a side-by-side
table. Pure stdlib: it shells out to evaluate_tasks.py (which does the torch/GPU work)
three times, parses PPL + the --quiet-diagnostics summary from each run, and writes
both a printed table and a JSON.

Variants:
  select    sparse attention, full cache (the existing Loki-style path)
  compress  rank-r cache + selective reconstruction (accuracy cost of compression)
  evict     static query-weighted score-energy eviction (needs --emit-rq offline)

Matched point: all three share --top-r and the keep fraction. For select/compress the
keep fraction is --top-k (<=1). For evict the budget is evict_ratio = 1 - keep, with
sinks+recent additionally protected (so effective keep can exceed the budget on short
sequences; that asymmetry is expected and is reported).

Example:
  python run_three_variants.py \
      --model-id mistralai/Mistral-7B-v0.2 --model-type mistral \
      --dataset wikitext-test --sequence-length 4096 \
      --top-r 8 --keep 0.125 \
      --rotary-type prerotary --transform-dataset c4 \
      --out /mnt/user-data/outputs/three_variants.json

Add --dry-run to print the three commands without running them.
"""
import argparse
import json
import re
import subprocess
import sys


PPL_RE = re.compile(r"tensor\(([-+0-9.eE]+)(?:[,)]|$)|^([-+0-9.]+)\s*$", re.M)
DIAG_RE = re.compile(r"^(\w+):\s+overall=([-+0-9.eE]+)\s+layers=(\d+)\s+samples=(\d+)", re.M)


def base_cmd(args):
    cmd = [
        sys.executable, "evaluate_tasks.py",
        "--model-id", args.model_id,
        "--model-type", args.model_type,
        "--sequence-length", str(args.sequence_length),
        "--dataset", args.dataset,
        "--use-pca-topk",            # required: selects the pca_topk modifier
        "--use-crosscov",            # required: switches to the CrossCov branch
        "--top-r", str(args.top_r),
        "--rotary-type", args.rotary_type,
        "--transform-dataset", args.transform_dataset,
        "--quiet-diagnostics",
        "--log-mass-recall",
        "--log-output-error",
    ]
    return cmd


def mode_cmd(args, mode):
    cmd = base_cmd(args) + ["--crosscov-mode", mode]
    if mode in ("select", "compress"):
        cmd += ["--top-k", str(args.keep)]
    else:  # evict
        evict_ratio = round(1.0 - args.keep, 6) if args.keep <= 1 else None
        if evict_ratio is None:
            raise SystemExit("evict matching requires --keep as a fraction <= 1")
        # top-k still needed by the forward's topk computation; harmless for evict.
        cmd += ["--top-k", str(args.keep),
                "--evict-ratio", str(evict_ratio),
                "--sink-tokens", str(args.sink_tokens),
                "--recent-window", str(args.recent_window)]
    if args.extra:
        cmd += args.extra
    return cmd


def parse_output(text):
    ppl = None
    for m in PPL_RE.finditer(text):
        val = m.group(1) or m.group(2)
        if val is not None:
            try:
                ppl = float(val)
            except ValueError:
                pass
    diags = {}
    for m in DIAG_RE.finditer(text):
        diags[m.group(1)] = {"overall": float(m.group(2)),
                             "layers": int(m.group(3)),
                             "samples": int(m.group(4))}
    return ppl, diags


def run(cmd):
    print("  $ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("  [stderr tail]\n" + "\n".join(proc.stderr.splitlines()[-15:]))
    return proc.stdout + "\n" + proc.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--model-type", default="mistral")
    ap.add_argument("--dataset", default="wikitext-test")
    ap.add_argument("--sequence-length", type=int, default=4096)
    ap.add_argument("--top-r", type=float, required=True, help="latent rank r (matched across modes)")
    ap.add_argument("--keep", type=float, required=True, help="keep fraction (<=1), matched across modes")
    ap.add_argument("--rotary-type", default="prerotary")
    ap.add_argument("--transform-dataset", default="c4")
    ap.add_argument("--sink-tokens", type=int, default=16)
    ap.add_argument("--recent-window", type=int, default=64)
    ap.add_argument("--modes", default="select,compress,evict")
    ap.add_argument("--out", default="three_variants.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra args forwarded verbatim to evaluate_tasks.py (must be last)")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = {}

    for mode in modes:
        cmd = mode_cmd(args, mode)
        if args.dry_run:
            print(f"[{mode}]\n  $ " + " ".join(cmd) + "\n")
            continue
        print(f"[{mode}] running...", flush=True)
        out = run(cmd)
        ppl, diags = parse_output(out)
        results[mode] = {"ppl": ppl, "diagnostics": diags}
        print(f"[{mode}] ppl={ppl} diags={ {k: v['overall'] for k, v in diags.items()} }\n", flush=True)

    if args.dry_run:
        return

    # side-by-side table
    diag_keys = sorted({k for r in results.values() for k in r["diagnostics"]})
    col = max(12, *(len(m) for m in modes))
    header = "metric".ljust(20) + "".join(m.rjust(col + 2) for m in modes)
    print("\n" + header)
    print("-" * len(header))
    ppl_row = "ppl".ljust(20) + "".join(
        (f"{results[m]['ppl']:.4f}" if results[m]['ppl'] is not None else "n/a").rjust(col + 2) for m in modes)
    print(ppl_row)
    for dk in diag_keys:
        row = dk.ljust(20)
        for m in modes:
            v = results[m]["diagnostics"].get(dk, {}).get("overall")
            row += (f"{v:.4f}" if v is not None else "-").rjust(col + 2)
        print(row)

    meta = {"operating_point": {"top_r": args.top_r, "keep": args.keep,
                               "evict_ratio": round(1.0 - args.keep, 6),
                               "sink_tokens": args.sink_tokens, "recent_window": args.recent_window},
            "model_id": args.model_id, "dataset": args.dataset,
            "sequence_length": args.sequence_length, "results": results}
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

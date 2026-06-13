#!/usr/bin/env python3
"""
Patch a local Llama-3.1 model config for this repo's pinned transformers==4.40.2.

Transformers 4.40.2 does not understand Llama-3.1's extended rope_scaling schema:
{"rope_type": "llama3", ...}. For the 8K WikiText/C4 experiments in this repo we
only need the original 8K context, so this helper backs up config.json, removes the
unsupported rope_scaling entry, and caps max_position_embeddings at the original
8K value.

Use only for <=8K runs. For >8K runs, use a newer transformers-compatible branch
instead of this shim.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="Local model directory, e.g. /home/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--restore", action="store_true", help="Restore config.json from config.json.llama31_rope_backup")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    config_path = model_dir / "config.json"
    backup_path = model_dir / "config.json.llama31_rope_backup"

    if args.restore:
        if not backup_path.exists():
            raise SystemExit(f"backup not found: {backup_path}")
        config_path.write_text(backup_path.read_text())
        print(f"restored {config_path} from {backup_path}")
        return

    cfg = json.loads(config_path.read_text())
    rope = cfg.get("rope_scaling")
    if rope and rope.get("rope_type") == "llama3":
        if not backup_path.exists():
            backup_path.write_text(config_path.read_text())
            print(f"backed up original config to {backup_path}")
        original_max = rope.get("original_max_position_embeddings", 8192)
        cfg.pop("rope_scaling", None)
        cfg["max_position_embeddings"] = int(original_max)
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"patched {config_path} for <= {original_max} token runs")
    else:
        print(f"no llama3 rope_scaling patch needed for {config_path}")


if __name__ == "__main__":
    main()

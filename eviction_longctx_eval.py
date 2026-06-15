import argparse
import json
import re
import string
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from methods.common.configure_model import get_eviction_args
from methods.eviction.modify_mistral import make_mistral_attention_eviction, reset_evict


def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


TASKS = ["hotpotqa", "2wikimqa", "musique", "multifieldqa_en"]
PROMPT = (
    "Answer the question based on the passages below. Only give the answer.\n\n"
    "{context}\n\nQuestion: {input}\nAnswer:"
)


def run(args):
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to("cuda")
    model.eval()
    if args.use_evict:
        make_mistral_attention_eviction(args)

    results = {}
    for task in args.tasks:
        ds = load_dataset("THUDM/LongBench", task, split="test")
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        scores = []
        for ex in ds:
            prompt = PROMPT.format(context=ex["context"], input=ex["input"])
            ids = tok(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_len,
            ).input_ids.cuda()
            reset_evict()
            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_new_tokens=args.max_gen,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            pred = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
            scores.append(max(f1_score(pred, g) for g in ex["answers"]))
        results[task] = sum(scores) / len(scores)
        print(f"{task:16s} F1={results[task]:.3f}  (n={len(scores)})")
    avg = sum(results.values()) / len(results)
    print(f"\n{args.evict_method:14s} budget={args.evict_budget}  avg F1={avg:.3f}")
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"results": results, "avg": avg, "args": vars(args)}, f, indent=2)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--model-type", default="mistral")
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--max-len", type=int, default=7500)
    p.add_argument("--max-gen", type=int, default=64)
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--output-json", default="")
    p = get_eviction_args(p)
    run(p.parse_args())

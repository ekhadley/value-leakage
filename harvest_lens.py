#!./.venv/bin/python
"""
Harvest J-lens readouts for one rollout, for view_tokens.py.

Runs the model over the rollout's exact token sequence (tokenize_rollout), applies the
J-lens at resid_pre of every source layer, and saves the top-k token ids and probs per
(layer, position) to <run_dir>/lens/<condition>-<i>.pt.

    python harvest_lens.py runs/qwen3.6-35b-a3b_20260829_133101 below_good 67
"""

import argparse
import os

from utils import *


def main():
    p = argparse.ArgumentParser(description="Harvest J-lens top-k readouts for a rollout")
    p.add_argument("run_dir", help="Run directory, e.g. runs/qwen3.6-35b-a3b_20260829_133101")
    p.add_argument("condition", help="Condition JSON name, e.g. baseline, above_good, below_good")
    p.add_argument("i", type=int, help="Rollout index within the condition")
    p.add_argument("--k", type=int, default=20, help="Top-k tokens to save per (layer, position)")
    p.add_argument("--chunk", type=int, default=512, help="Positions per unembed chunk")
    p.add_argument("--model-path", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--lens-path", default="qwen3.6-35b-a3b/j-lens/lens.pt")
    args = p.parse_args()

    t.set_grad_enabled(False)
    print(f"{gray}booting {args.model_path}...{endc}")
    model = TransformerBridge.boot_transformers(args.model_path, dtype=t.bfloat16, device_map="auto")
    model.eval()
    lens = load_jlens(args.lens_path)

    rollout, meta = load_rollout(args.run_dir, args.condition, args.i)
    toks = tokenize_rollout(rollout, meta["prompt"], model.tokenizer).to(model.device)
    print(f"{gray}forward over {len(toks)} tokens...{endc}")
    _, cache = model.run_with_cache(toks, names_filter=lambda n: "resid_pre" in n)
    t.cuda.empty_cache()

    layers = lens["source_layers"]
    top_ids = t.empty(len(layers), len(toks), args.k, dtype=t.int32)
    top_probs = t.empty(len(layers), len(toks), args.k, dtype=t.float16)
    for li, layer in enumerate(tqdm(layers, desc="lens layers", ascii=" >=")):
        acts = cache[f"blocks.{layer}.hook_resid_pre"].squeeze(0)
        J = lens["J"][layer].to(acts.device, t.bfloat16)
        for start in range(0, len(toks), args.chunk):
            h = acts[start : start + args.chunk] @ J.T
            probs = model.unembed(model.ln_final(h)).float().softmax(-1)
            top = probs.topk(args.k)
            top_ids[li, start : start + h.shape[0]] = top.indices.to("cpu", t.int32)
            top_probs[li, start : start + h.shape[0]] = top.values.to("cpu", t.float16)
        t.cuda.empty_cache()

    os.makedirs(f"{args.run_dir}/lens", exist_ok=True)
    path = f"{args.run_dir}/lens/{args.condition}-{args.i}.pt"
    t.save({"layers": layers, "k": args.k, "ids": top_ids, "probs": top_probs, "toks": toks.cpu()}, path)
    print(f"{green}saved {cyan}{tuple(top_ids.shape)}{green} lens readouts to {cyan}{path}{endc}")


if __name__ == "__main__":
    main()

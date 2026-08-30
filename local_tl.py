#%%
from utils import *
from transformer_lens.model_bridge import TransformerBridge

from value_leakage.sample import BASELINE, ABOVE_GOOD, BELOW_GOOD

t.set_grad_enabled(False)
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
t.random.manual_seed(random_seed)

#%%

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

model = TransformerBridge.boot_transformers(
    MODEL_ID,
    dtype=t.bfloat16,
    device_map="auto",
)
tokenizer = model.tokenizer
model.eval()

#%%

test_streaming_generation = False
if test_streaming_generation:
    new_toks = 256

    
    t.cuda.empty_cache()
    conversation = [
        {"role": "user", "content": BASELINE},
    ]
    conv_toks = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=False,
        enable_thinking=True,
        reasoning_effort="medium",
    ).to(model.device)

    print(tokenizer.decode(conv_toks)[0])
    for tok in stream_toks_bridge(model, conv_toks, new_toks=new_toks):
        print(tokenizer.decode(tok), end="", flush=True)
    t.cuda.empty_cache()

#%%

get_rollout_acts_and_logits = False
if get_rollout_acts_and_logits:
    # rollout, meta = load_rollout("runs/qwen3.6-35b-a3b_20260829_133101", "baseline", 10)
    # rollout, meta = load_rollout("runs/qwen3.6-35b-a3b_20260829_133101", "above_good", 10)
    rollout, meta = load_rollout("runs/qwen3.6-35b-a3b_20260829_133101", "below_good", 67)

    rollout_tokens = tokenize_rollout(rollout, meta["prompt"], tokenizer).to(model.device)
    print(tokenizer.decode(rollout_tokens))

    logits, cache = model.run_with_cache(rollout_tokens, names_filter=lambda n: "resid" in n)

    t.cuda.empty_cache()

#%%

lens = load_jlens("qwen3.6-35b-a3b/j-lens/lens.pt", device=model.device)
print(lens.keys())
print(lens["provenance"])

#%%

def jlens_transport(acts: Tensor, lens: dict, layer: int) -> Tensor:
    lens_layer = lens["J"][layer]
    return lens_layer @ acts

def get_lens_logits(h: Tensor, layer: int, model: TransformerBridge, lens: dict) -> Tensor:
    return model.unembed(model.ln_final(jlens_transport(h, lens, layer)))

def top_toks_table(logits: Tensor, tokenizer, k: int = 10, show_negative: bool = False, show_probs: bool = True, title: str | None = None, return_top=False):
    logits = logits.flatten()
    probs = logits.softmax(-1)
    top = logits.topk(k)
    top_strs = [tokenizer.decode([tok]) for tok in top.indices.tolist()]
    top_vals = top.values.tolist()
    headers = ["Tok", "Value"] + (["Prob"] if show_probs else [])
    cols = [[repr(s) for s in top_strs], top_vals] + ([probs[top.indices].tolist()] if show_probs else [])
    if show_negative:
        bot = logits.topk(k, largest=False)
        bot_strs = [tokenizer.decode([tok]) for tok in bot.indices.tolist()]
        bot_vals = bot.values.tolist()
        headers = [f"Top {h}" for h in headers] + [f"Bot {h}" for h in headers]
        cols += [[repr(s) for s in bot_strs], bot_vals] + ([probs[bot.indices].tolist()] if show_probs else [])
    data = [(i, *(col[i] for col in cols)) for i in range(k)]
    table_str = tabulate(data, headers=["Idx"] + headers, tablefmt="rounded_outline")
    if title is not None:
        lines = table_str.splitlines()
        inner = len(lines[0]) - 2
        print(f"╭{'─' * inner}╮")
        print(f"│{bold}{title.center(inner)}{endc}│")
        print(f"├{'─' * inner}┤")
        print("\n".join(lines[1:]))
    else:
        print(table_str)

    if return_top:
        if show_negative:
            return (top_strs, top_vals, bot_strs, bot_vals)
        else:
            return (top_strs, top_vals)

conv = [
    {"role": "user", "content": "What's the currency used in the country shaped like a boot?"},
    # {"role": "user", "content": BASELINE},
]
conv_toks = tokenizer.apply_chat_template(
    conv,
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
    return_dict=False,
    enable_thinking=False,
    reasoning_effort="medium",
).to(model.device)

for i, stok in enumerate(to_str_toks(conv_toks, tokenizer)):
    print(f"[{i}] {stok}")

logits, cache = model.run_with_cache(conv_toks, names_filter=lambda n: "resid" in n)
t.cuda.empty_cache()

def jlens_transport(acts: Tensor, lens: dict, layer: int) -> Tensor:
    lens_layer = lens["J"][layer].to(t.bfloat16)
    return lens_layer @ acts

def get_lens_logits(h: Tensor, layer: int, model: TransformerBridge, lens: dict) -> Tensor:
    return model.unembed(model.ln_final(jlens_transport(h, lens, layer)))

layer = 26
seq_pos = 15
targ_acts = cache[f"blocks.{layer}.hook_resid_pre"].squeeze()[seq_pos]
print(pink, targ_acts.shape, endc)
lens_logits = get_lens_logits(targ_acts, layer, model, lens)
top_toks_table(lens_logits, tokenizer, k=15)

t.cuda.empty_cache()

#%%
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

get_rollout_acts_and_logits = True
if get_rollout_acts_and_logits:
    # rollout, meta = load_rollout("runs/qwen3.6-35b-a3b_20260829_133101", "baseline", 10)
    # rollout, meta = load_rollout("runs/qwen3.6-35b-a3b_20260829_133101", "above_good", 10)
    rollout, meta = load_rollout("runs/qwen3.6-35b-a3b_20260829_133101", "below_good", 67)

    rollout_tokens = tokenize_rollout(rollout, meta["prompt"], tokenizer).to(model.device)
    print(tokenizer.decode(rollout_tokens))

    logits, cache = model.run_with_cache(rollout_tokens, names_filter=lambda n: "resid" in n)

    t.cuda.empty_cache()

#%%

def jlens_transport(acts: Tensor, lens: dict, layer: int) -> Tensor:
    lens_layer = lens["J"][layer]
    return lens_layer @ acts

def get_lens_logits(h: Tensor, layer: int, model: TransformerBridge, lens: dict) -> Tensor:
    return model.unembed(model.ln_final(jlens_transport(h, lens, layer)))

lens = load_jlens("qwen3.6-35b-a3b/j-lens/lens.pt", device=model.device)
print(lens.keys())
print(lens["provenance"])


#%%
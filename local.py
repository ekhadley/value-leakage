#%%
from utils import *

from value_leakage.sample import BASELINE, ABOVE_GOOD, BELOW_GOOD

t.set_grad_enabled(False)

#%%

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=t.bfloat16,
    device_map="auto",
)
model.eval()

#%%

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

def stream_toks(inputs, new_toks: int = 512):
    toks = inputs
    past = None
    for _ in range(new_toks):
        out = model(toks, past_key_values=past)
        past = out.past_key_values
        probs = t.softmax(out.logits[0, -1], dim=-1)
        toks = t.multinomial(probs, num_samples=1).unsqueeze(0)
        if toks.item() == tokenizer.eos_token_id:
            break
        yield toks.item()

print(tokenizer.decode(conv_toks)[0])
for tok in stream_toks(conv_toks, new_toks=4096):
    print(tokenizer.decode(tok), end="", flush=True)
t.cuda.empty_cache()

#%%

def rollout_tokens(data: dict, i: int) -> t.Tensor:
    """Full token sequence for row i of a saved condition JSON: templated prompt + think block + answer."""
    row = data["rows"][i]
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": data["prompt"]}], add_generation_prompt=True, return_tensors="pt")
    completion = f"<think>\n{row['reasoning']}\n</think>\n\n{row['content']}"
    completion_ids = tokenizer(completion, return_tensors="pt", return_dict=False, tokenize=True)
    return t.cat([prefix, completion_ids], dim=1)

baseline = json.load(open("runs/qwen3.6-35b-a3b_20260829_133101/baseline.json"))
# baseline = json.load(open("runs/qwen3.6-35b-a3b_20260829_133101/above_good.json"))
tokens = rollout_tokens(baseline, 0)
print(f"{cyan}{tokens.shape=}{endc}")
print(tokenizer.decode(tokens[0]))

#%%

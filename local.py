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
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": data["prompt"]}],
        add_generation_prompt = True,
        return_tensors = "pt",
        tokenize = True,
        return_dict = False
    )
    completion = f"<think>\n{row['reasoning']}\n</think>\n\n{row['content']}"
    print(completion)
    completion_ids = tokenizer.encode(completion, return_tensors="pt")
    print(completion_ids)
    return t.cat([prefix, completion_ids], dim=1)

baseline = json.load(open("runs/qwen3.6-35b-a3b_20260829_133101/baseline.json"))
# baseline = json.load(open("runs/qwen3.6-35b-a3b_20260829_133101/above_good.json"))
tokens = rollout_tokens(baseline, 0)
print(f"{cyan}{tokens.shape=}{endc}")
print(tokenizer.decode(tokens[0]))

#%%

#%%

example_reasoning = "Maybe 30 million spots. Hmm, wait, that seems low. Let's say 400 billion."
example_content = "My final estimate is 400000000000."
fake = {"prompt": BASELINE, "rows": [{"reasoning": example_reasoning, "content": example_content}]}
hand = rollout_tokens(fake, 0)[0]

conv = [
    {"role": "user", "content": BASELINE},
    {"role": "assistant", "content": f"<think>\n{example_reasoning}\n</think>\n\n{example_content}"},
]
ref = tokenizer.apply_chat_template(conv, return_tensors="pt", return_dict=False)[0]

n = min(len(hand), len(ref))
mismatches = (hand[:n] != ref[:n]).nonzero()
if len(mismatches) == 0 and len(hand) == len(ref):
    print(f"{green}exact match ({len(hand)} tokens){endc}")
elif len(mismatches) == 0:
    longer, name = (hand, "hand") if len(hand) > len(ref) else (ref, "ref")
    print(f"{yellow}common prefix matches, but lengths differ: hand={len(hand)} ref={len(ref)}{endc}")
    print(f"{yellow}extra {name} tokens: {tokenizer.decode(longer[n:])!r}{endc}")
else:
    i = mismatches[0].item()
    print(f"{red}first mismatch at token {i}/{n}{endc}")
    print(f"{gray}shared context before it: {tokenizer.decode(hand[max(0, i - 10):i])!r}{endc}")
    print(f"hand: {[tokenizer.decode(tok) for tok in hand[i:i + 8]]}")
    print(f"ref:  {[tokenizer.decode(tok) for tok in ref[i:i + 8]]}")

#%%
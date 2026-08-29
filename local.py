#!./.venv/bin/python
#%%
from utils import *

from value_leakage.sample import BASELINE

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

conversation = [
    {"role": "user", "content": BASELINE},
]
inputs = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt").to(model.device)

out = model.generate(inputs, max_new_tokens=8192, do_sample=True, temperature=1.0)
completion = tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
reasoning, _, answer = completion.rpartition("</think>")

print(reasoning)
print(answer)

#%%

def rollout_tokens(data: dict, i: int) -> t.Tensor:
    """Full token sequence for row i of a saved condition JSON: templated prompt + think block + answer."""
    row = data["rows"][i]
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": data["prompt"]}], add_generation_prompt=True, return_tensors="pt")
    completion = f"<think>\n{row['reasoning']}\n</think>\n\n{row['content']}"
    completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids
    return t.cat([prefix, completion_ids], dim=1)

baseline = json.load(open("runs/qwen3.6-35b-a3b_20260829_133101/baseline.json"))
tokens = rollout_tokens(baseline, 0)
print(f"{cyan}{tokens.shape=}{endc}")
print(tokenizer.decode(tokens[0]))

#%%

row = baseline["rows"][0]
print(f"{bold}{yellow}{baseline['model']} | {baseline['condition']} | rollout {row['i']}{endc}")
print(f"{gray}{row['reasoning']}{endc}")
print(f"{lime}{row['content']}{endc}")

#%%

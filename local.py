#!./.venv/bin/python
#%%
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM

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

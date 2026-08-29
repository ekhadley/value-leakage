#%%
from utils import *

from value_leakage.sample import BASELINE, ABOVE_GOOD, BELOW_GOOD

t.set_grad_enabled(False)
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
t.random.manual_seed(random_seed)

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

test_streaming_generation = True
if test_streaming_generation:
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
    for tok in stream_toks(conv_toks, tokenizer, new_toks=4096):
        print(tokenizer.decode(tok), end="", flush=True)
    t.cuda.empty_cache()

#%%

rollout, meta = load_rollout("runs/qwen3.6-35b-a3b_20260829_133101", "above_good", 10)
tokens = tokenize_rollout(rollout, meta["prompt"], tokenizer)
print(f"{cyan}{tokens.shape=}{endc}")
print(tokenizer.decode(tokens))

#%%
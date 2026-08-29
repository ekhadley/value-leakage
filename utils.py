import json
import requests
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
import IPython

import numpy as np

import torch as t
from torch import Tensor

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from transformer_lens.model_bridge import TransformerBridge

IPYTHON = IPython.get_ipython()
if IPYTHON is not None:
    IPYTHON.run_line_magic('load_ext', 'autoreload')
    IPYTHON.run_line_magic('autoreload', '2')

purple = '\x1b[38;2;255;0;255m'
blue = '\x1b[38;2;0;0;255m'
brown = '\x1b[38;2;128;128;0m'
cyan = '\x1b[38;2;0;255;255m'
lime = '\x1b[38;2;0;255;0m'
yellow = '\x1b[38;2;255;255;0m'
red = '\x1b[38;2;255;0;0m'
pink = '\x1b[38;2;255;51;204m'
orange = '\x1b[38;2;255;51;0m'
green = '\x1b[38;2;5;170;20m'
gray = '\x1b[38;2;127;127;127m'
magenta = '\x1b[38;2;128;0;128m'
white = '\x1b[38;2;255;255;255m'
bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

load_dotenv()

def print_batch(results: list[dict]) -> None:
    for r in results:
        msg = r["choices"][0]["message"]
        print(f"{bold}{yellow}{r['model']}{endc}")
        if msg.get("reasoning"):
            print(f"{gray}{msg['reasoning']}{endc}")
        print(f"{lime}{msg['content']}{endc}")
        print()

def query(model: str, prompt: str) -> dict:
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning": {"enabled": True},
        },
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


def run_batch(model: str, prompt: str, n_samples: int, bar: tqdm) -> list[dict]:
    with ThreadPoolExecutor(max_workers=n_samples) as pool:
        futures = [pool.submit(query, model, prompt) for _ in range(n_samples)]
        for done, _ in enumerate(as_completed(futures), 1):
            bar.set_description(f"{model} {done}/{n_samples} done")
    return [f.result() for f in futures]


def save_batch(results: list[dict], name: str = None, timestamp: bool = True, metadata: dict = None) -> str:
    os.makedirs("results", exist_ok=True)
    name = (name or results[0]["model"]).replace("/", "_")
    if timestamp:
        name += time.strftime("_%Y%m%d_%H%M%S")
    path = f"results/{name}.json"
    with open(path, "w") as f:
        json.dump({"metadata": metadata or {}, "results": results}, f, indent=2)
    return path


def load_batch_results(path: str) -> tuple[list[dict], dict]:
    with open(path) as f:
        data = json.load(f)
    return data["results"], data["metadata"]

def load_rollout(run_dir: str, condition: str, i: int) -> tuple[dict, dict]:
    """Row i of a condition JSON, plus the run metadata and that rollout's judged estimate trajectory."""
    data = json.load(open(f"{run_dir}/{condition}.json"))
    rollout = data["rows"][i]
    meta = {k: v for k, v in data.items() if k != "rows"}
    meta["trajectory"] = json.load(open(f"{run_dir}/trajectories.json"))[condition][i]
    return rollout, meta

def stream_toks_bridge(model: TransformerBridge, inputs, new_toks: int = 512):
    toks = inputs
    past = None
    for _ in range(new_toks):
        logits, past = model(toks, return_type="logits_and_cache", past_key_values=past)
        probs = t.softmax(logits[0, -1], dim=-1)
        toks = t.multinomial(probs, num_samples=1).unsqueeze(0)
        if toks.item() == model.tokenizer.eos_token_id:
            break
        yield toks.item()

def tokenize_rollout(rollout: dict, prompt: str, tokenizer: AutoTokenizer, tokenize = True) -> t.Tensor|str:
    """Full token sequence for a rollout: templated prompt + think block + answer."""
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt = True,
        return_tensors = "pt",
        tokenize = True,
        return_dict = False
    )
    completion = f"{rollout['reasoning']}\n</think>\n\n{rollout['content']}"
    completion_ids = tokenizer.encode(completion, return_tensors="pt")
    full_toks = t.cat([prefix, completion_ids], dim=1)[0]
    if not tokenize:
        return tokenizer.decode(full_toks)
    return full_toks

def load_jlens(path: str, device: str = "cpu") -> dict:
    """Download a single lens .pt file from the workspace-lenses repo (cached) and load it."""
    local_path = hf_hub_download(repo_id="camilablank/workspace-lenses", filename=path)
    return t.load(local_path, map_location=device, weights_only=False)

def stream_toks(inputs, tokenizer: AutoTokenizer, new_toks: int = 512):
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
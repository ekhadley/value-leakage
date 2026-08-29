import json
import requests
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

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
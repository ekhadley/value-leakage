#!./.venv/bin/python
"""
Interactive token-level viewer for a rollout.

Serves the exact token sequence produced by tokenize_rollout (templated prompt +
think block + answer) as one flat, scannable stream, split by turn dividers with
a clickable turn sidebar. Hover any token for its position index, vocab id, and
exact string; click to pin it in the header. Press 'b' to toggle token
boundaries, or type an index in the jump box. Activation overlays come later.

    python view_tokens.py runs/qwen3.6-35b-a3b_20260829_133101 below_good 67
"""

import argparse
import html
import json

from flask import Flask
from transformers import AutoTokenizer

from utils import load_rollout, tokenize_rollout

PAGE = """<!doctype html>
<title>__TITLE__</title>
<style>
    body { margin: 0; height: 100vh; display: flex; flex-direction: column; background: #282828; color: #ebdbb2; font-family: monospace; }
    #header { flex: none; display: flex; align-items: center; gap: 16px; background: #1d2021; padding: 8px 16px; font-size: 13px; color: #a89984; border-bottom: 1px solid #3c3836; }
    #jump { width: 70px; background: #282828; color: #ebdbb2; border: 1px solid #504945; border-radius: 3px; padding: 2px 6px; font-family: monospace; }
    #pin { color: #fe8019; white-space: pre; }
    #main { flex: 1; display: flex; overflow: hidden; }
    #side { flex: none; width: 170px; overflow-y: auto; background: #1d2021; border-right: 1px solid #3c3836; padding: 8px 0; font-size: 12px; }
    #side div { padding: 3px 12px; cursor: pointer; color: #928374; }
    #side div:hover { background: #3c3836; color: #ebdbb2; }
    #stream { flex: 1; overflow-y: auto; padding: 16px 24px; white-space: pre-wrap; overflow-wrap: break-word; font-size: 14px; line-height: 1.7; cursor: default; user-select: none; }
    .tok { border-radius: 2px; }
    .even { background: #2e2c2b; }
    .noalt .even { background: transparent; }
    .special { color: #d3869b; }
    .tok:hover { background: #504945; }
    .pinned { box-shadow: 0 0 0 1px #fe8019; }
    .flash { animation: flash 1.2s; }
    @keyframes flash { from { background: #fabd2f; color: #282828; } }
    .divider { margin: 10px 0 4px; font-size: 12px; color: #928374; user-select: none; }
    .system { color: #928374; } .user { color: #83a598; } .assistant { color: #b8bb26; }
    #tip { position: fixed; display: none; background: #1d2021; color: #ebdbb2; border: 1px solid #504945; border-radius: 4px; padding: 4px 8px; font-size: 12px; font-family: monospace; pointer-events: none; z-index: 10; white-space: pre; }
</style>
<div id="header">
    <span>__TITLE__ &mdash; __NTOK__ tokens</span>
    <input id="jump" placeholder="pos...">
    <span id="pin"></span>
</div>
<div id="main"><div id="side"></div><div id="stream"></div></div>
<div id="tip"></div>
<script>
    const data = __DATA__;
    const specialIds = new Set(data.special_ids);
    const visible = s => s.replace(/ /g, "·").replace(/\\n/g, "⏎").replace(/\\t/g, "⇥");
    const info = pos => `pos ${pos}  id ${data.ids[pos]}  '${visible(data.texts[pos])}'`;

    const stream = document.getElementById("stream");
    const side = document.getElementById("side");
    const frag = document.createDocumentFragment();
    const spans = [];
    let turnIdx = 0;
    data.ids.forEach((id, pos) => {
        if (turnIdx < data.turns.length && data.turns[turnIdx].start === pos) {
            const role = data.turns[turnIdx].role;
            const div = document.createElement("div");
            div.className = "divider " + role;
            div.id = "turn-" + turnIdx;
            div.textContent = `── turn ${turnIdx} · ${role} ──`;
            frag.appendChild(div);
            const nav = document.createElement("div");
            nav.innerHTML = `${turnIdx} <span class="${role}">${role}</span>`;
            nav.onclick = () => document.getElementById(div.id).scrollIntoView({ block: "start", behavior: "smooth" });
            side.appendChild(nav);
            turnIdx++;
        }
        const span = document.createElement("span");
        span.className = "tok" + (specialIds.has(id) ? " special" : (pos % 2 ? "" : " even"));
        span.textContent = data.texts[pos];
        span.dataset.pos = pos;
        spans.push(span);
        frag.appendChild(span);
    });
    stream.appendChild(frag);

    const tip = document.getElementById("tip");
    stream.addEventListener("mousemove", e => {
        const span = e.target.closest(".tok");
        if (!span) { tip.style.display = "none"; return; }
        tip.textContent = info(+span.dataset.pos);
        tip.style.display = "block";
        tip.style.left = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8) + "px";
        tip.style.top = (e.clientY + 18) + "px";
    });
    stream.addEventListener("mouseleave", () => tip.style.display = "none");

    const pin = document.getElementById("pin");
    let pinnedEl = null;
    stream.addEventListener("click", e => {
        const span = e.target.closest(".tok");
        if (pinnedEl) pinnedEl.classList.remove("pinned");
        pinnedEl = span;
        pin.textContent = span ? info(+span.dataset.pos) : "";
        if (span) span.classList.add("pinned");
    });

    document.getElementById("jump").addEventListener("keydown", e => {
        if (e.key !== "Enter") return;
        const span = spans[+e.target.value];
        if (!span) return;
        span.scrollIntoView({ block: "center" });
        span.classList.remove("flash");
        void span.offsetWidth;
        span.classList.add("flash");
    });

    document.addEventListener("keydown", e => {
        if (e.key === "b" && e.target.tagName !== "INPUT") stream.classList.toggle("noalt");
    });
</script>
"""


def build_turns(ids: list[int], texts: list[str], im_start_id: int) -> list[dict]:
    """Split the stream at <|im_start|> tokens; the role is the token that follows."""
    return [{"start": pos, "role": texts[pos + 1].strip()} for pos, tid in enumerate(ids) if tid == im_start_id]


def main():
    p = argparse.ArgumentParser(description="Serve a token-level view of a rollout")
    p.add_argument("run_dir", help="Run directory, e.g. runs/qwen3.6-35b-a3b_20260829_133101")
    p.add_argument("condition", help="Condition JSON name, e.g. baseline, above_good, below_good")
    p.add_argument("i", type=int, help="Rollout index within the condition")
    p.add_argument("--model-path", default="Qwen/Qwen3.6-35B-A3B", help="Tokenizer to decode with")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    rollout, meta = load_rollout(args.run_dir, args.condition, args.i)
    ids = tokenize_rollout(rollout, meta["prompt"], tokenizer).tolist()

    texts = [tokenizer.decode([i]) for i in ids]
    added = {tok.content: tid for tid, tok in tokenizer.added_tokens_decoder.items()}
    turns = build_turns(ids, texts, added["<|im_start|>"])
    data = {"ids": ids, "texts": texts, "special_ids": sorted(set(ids) & set(tokenizer.added_tokens_decoder.keys())), "turns": turns}
    title = f"{args.run_dir} {args.condition}[{args.i}]"
    page = PAGE.replace("__TITLE__", html.escape(title)).replace("__NTOK__", str(len(ids))).replace("__DATA__", json.dumps(data).replace("</", "<\\/"))

    app = Flask(__name__)
    app.add_url_rule("/", "index", lambda: page)
    print(f"serving {len(ids)} tokens, {len(turns)} turns at http://localhost:{args.port}")
    app.run(port=args.port)


if __name__ == "__main__":
    main()

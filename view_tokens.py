#!./.venv/bin/python
"""
Interactive token-level browser for a run directory.

Serves every rollout in the run: tabs switch between the conditions (baseline,
below_good, above_good), and the index box / arrow buttons pick the rollout
within the condition (left/right arrow keys work too). Each rollout is the
exact token sequence produced by tokenize_rollout (templated prompt + think
block + answer) as one flat, scannable stream, split by turn dividers with a
clickable turn sidebar. Hover any token for its position index, vocab id, and
exact string; click to pin it in the header. Press 'b' to toggle token
boundaries, or type an index in the jump box. Activation overlays come later.

    python view_tokens.py runs/qwen3.6-35b-a3b_20260829_133101
"""

import argparse
import html
import json
import os

from flask import Flask
from transformers import AutoTokenizer

from utils import load_rollout, tokenize_rollout

PAGE = """<!doctype html>
<title>__TITLE__</title>
<style>
    body { margin: 0; height: 100vh; display: flex; flex-direction: column; background: #282828; color: #ebdbb2; font-family: monospace; }
    #header { flex: none; display: flex; align-items: center; gap: 12px; background: #1d2021; padding: 8px 16px; font-size: 13px; color: #a89984; border-bottom: 1px solid #3c3836; }
    .tab { background: #282828; color: #928374; border: 1px solid #504945; border-radius: 3px; padding: 2px 8px; font-family: monospace; cursor: pointer; }
    .tab.active { color: #fabd2f; border-color: #fabd2f; }
    #prev, #next { background: #282828; color: #a89984; border: 1px solid #504945; border-radius: 3px; font-family: monospace; cursor: pointer; }
    #idx { width: 50px; background: #282828; color: #ebdbb2; border: 1px solid #504945; border-radius: 3px; padding: 2px 6px; font-family: monospace; }
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
    <span>__TITLE__</span>
    <span id="tabs"></span>
    <button id="prev">&#9664;</button>
    <input id="idx" type="number" min="0" value="0">
    <button id="next">&#9654;</button>
    <span id="count"></span>
    <span id="ntok"></span>
    <input id="jump" placeholder="pos...">
    <span id="pin"></span>
</div>
<div id="main"><div id="side"></div><div id="stream"></div></div>
<div id="tip"></div>
<script>
    const conds = __CONDS__;
    let cond = Object.keys(conds)[0], idx = 0, data = null, spans = [], pinnedEl = null;
    if (location.hash) { const [c, i] = location.hash.slice(1).split("/"); if (c in conds) { cond = c; idx = +i || 0; } }

    const stream = document.getElementById("stream");
    const side = document.getElementById("side");
    const pin = document.getElementById("pin");
    const visible = s => s.replace(/ /g, "·").replace(/\\n/g, "⏎").replace(/\\t/g, "⇥");
    const info = pos => `pos ${pos}  id ${data.ids[pos]}  '${visible(data.texts[pos])}'`;

    const tabs = document.getElementById("tabs");
    Object.keys(conds).forEach(c => {
        const b = document.createElement("button");
        b.className = "tab";
        b.textContent = c;
        b.onclick = () => { cond = c; load(); };
        tabs.appendChild(b);
    });

    function render() {
        stream.textContent = side.textContent = pin.textContent = "";
        pinnedEl = null;
        spans = [];
        const specialIds = new Set(data.special_ids);
        const frag = document.createDocumentFragment();
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
        stream.scrollTop = 0;
        document.getElementById("ntok").textContent = data.ids.length + " tokens";
    }

    function load() {
        idx = Math.max(0, Math.min(idx, conds[cond] - 1));
        location.hash = `${cond}/${idx}`;
        document.getElementById("idx").value = idx;
        document.getElementById("count").textContent = "/ " + conds[cond];
        document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b.textContent === cond));
        document.getElementById("ntok").textContent = "loading...";
        fetch(`/data/${cond}/${idx}`).then(r => r.json()).then(d => { data = d; render(); });
    }

    document.getElementById("prev").onclick = () => { idx--; load(); };
    document.getElementById("next").onclick = () => { idx++; load(); };
    document.getElementById("idx").addEventListener("keydown", e => {
        if (e.key !== "Enter") return;
        idx = +e.target.value;
        load();
    });

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
        if (e.target.tagName === "INPUT") return;
        if (e.key === "b") stream.classList.toggle("noalt");
        if (e.key === "ArrowLeft") { idx--; load(); }
        if (e.key === "ArrowRight") { idx++; load(); }
    });

    load();
</script>
"""


def build_turns(ids: list[int], texts: list[str], im_start_id: int) -> list[dict]:
    """Split the stream at <|im_start|> tokens; the role is the token that follows."""
    return [{"start": pos, "role": texts[pos + 1].strip()} for pos, tid in enumerate(ids) if tid == im_start_id]


def main():
    p = argparse.ArgumentParser(description="Serve a token-level browser for every rollout in a run")
    p.add_argument("run_dir", help="Run directory, e.g. runs/qwen3.6-35b-a3b_20260829_133101")
    p.add_argument("--model-path", default="Qwen/Qwen3.6-35B-A3B", help="Tokenizer to decode with")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    added = {tok.content: tid for tid, tok in tokenizer.added_tokens_decoder.items()}
    special_ids = set(tokenizer.added_tokens_decoder.keys())

    conds = {c: len(json.load(open(f"{args.run_dir}/{c}.json"))["rows"]) for c in ("baseline", "below_good", "above_good") if os.path.exists(f"{args.run_dir}/{c}.json")}
    page = PAGE.replace("__TITLE__", html.escape(args.run_dir)).replace("__CONDS__", json.dumps(conds))

    cache = {}
    def data(condition: str, i: int) -> str:
        if (condition, i) not in cache:
            rollout, meta = load_rollout(args.run_dir, condition, i)
            ids = tokenize_rollout(rollout, meta["prompt"], tokenizer).tolist()
            texts = [tokenizer.decode([tid]) for tid in ids]
            payload = {"ids": ids, "texts": texts, "special_ids": sorted(set(ids) & special_ids), "turns": build_turns(ids, texts, added["<|im_start|>"])}
            cache[(condition, i)] = json.dumps(payload)
        return cache[(condition, i)]

    app = Flask(__name__)
    app.add_url_rule("/", "index", lambda: page)
    app.add_url_rule("/data/<condition>/<int:i>", "data", data)
    print(f"serving {sum(conds.values())} rollouts across {list(conds)} at http://localhost:{args.port}")
    app.run(port=args.port)


if __name__ == "__main__":
    main()

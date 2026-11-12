#!/usr/bin/env python3
"""Decode battery: the same work per arm, measured by the engine. Use it to compare two settings of one lane
(worker counts, a power profile, a draft setting) with one fresh engine start per arm.

Part 1 (short): five short probes (prose, code, JSON, sampled prose, Spanish; thinking off, median of n).
Part 2 (agent): a ~34K-token agent context (system prompt, tool schemas, code, docs, Spanish, JSON) and six
different tasks at the lane's thinking default and its own sampling, twice. Short repeated prompts keep the
expert cache at 94-99% hits and barely load the CPU pool; mixed agent work sits near 89%, which is where a
worker-count or power change shows.

Usage: python3 decode_battery.py <label> [--passes N] [--no-short] [--no-agent] [--no-warm]
Writes battery-<label>.json next to this file and prints one summary block.
"""
import argparse
import json
import pathlib
import statistics as st

from lane_common import STRATA, Sampler, chat, get, model_id, power_profile, throttle_count

HERE = pathlib.Path(__file__).resolve().parent
PROSE = "Write a 150-word description of ocean waves at sunrise."
SHORT = [  # key, prompt, max_tokens, temperature, runs
    ("prose150", PROSE, 300, 0.0, 4),
    ("code600", "Write a Python module: a thread-safe LRU cache class with get/put/eviction, plus unit tests. "
                "Code only.", 900, 0.0, 2),
    ("json400", "Emit a JSON array of 40 inventory objects, each with sku, name, qty, price and warehouse fields, "
                "plausible and distinct. JSON only.", 600, 0.0, 2),
    ("sampled150", PROSE, 300, 0.7, 2),
    ("spanish150", "Escribe una descripcion de 150 palabras sobre las olas del mar al amanecer.", 300, 0.0, 2),
]

SYSTEM = (
    "You are an autonomous coding agent working in a Linux terminal on the user's machine. You plan before you act, "
    "you read files before changing them, you keep changes minimal and you verify your work by running it. "
    "When you answer, be precise and concrete: name the files, functions and flags you mean. Prefer plain statements "
    "over hedging. When a task asks for code, return complete, runnable code. When a task asks for an explanation, "
    "ground every claim in the material you were given. You have tools for reading and writing files, searching, "
    "patching and running shell commands; call a tool only when the task needs information you do not already have. "
    "For the tasks in this session the project context is already loaded below, so answer directly."
)

def _fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}

TOOLS = [
    _fn("read_file", "Read a text file with line numbers.", {"path": {"type": "string"}, "offset": {"type": "integer"},
                                                             "limit": {"type": "integer"}}, ["path"]),
    _fn("write_file", "Write a file, replacing its content.", {"path": {"type": "string"},
                                                              "content": {"type": "string"}}, ["path", "content"]),
    _fn("patch", "Replace one unique string in a file.", {"path": {"type": "string"}, "old_string": {"type": "string"},
                                                         "new_string": {"type": "string"}}, ["path", "old_string", "new_string"]),
    _fn("search_files", "Search file contents with a regular expression.", {"pattern": {"type": "string"},
                                                                           "path": {"type": "string"}}, ["pattern"]),
    _fn("terminal", "Run a shell command and return its output.", {"command": {"type": "string"},
                                                                  "timeout": {"type": "integer"}}, ["command"]),
]

FILES = [("serve/server.py", 0, 60000), ("tools/needle_bench.py", 0, 20000), ("docs/HOW_IT_WORKS.md", 0, 25000),
         ("README.es.md", 0, 12000), ("docs/data/capabilities.json", 0, 15000), ("docs/TROUBLESHOOTING.md", 0, 30000)]

RESULT = ("Tool result (terminal, exit 0):\n"
          "$ git status --short && python -m pytest -q serve/test_vram.py | tail -n 3\n"
          " M serve/server.py\n?? notes/plan.md\n"
          "........................................................................ [ 72%]\n"
          "............................                                             [100%]\n"
          "100 passed in 4.81s\n\n")

TASKS = [  # key, instruction, max_tokens
    ("code", "Write a new Python module `ratelimit.py` for this project: an asyncio token-bucket rate limiter class "
             "with a per-key registry and a context manager, plus pytest tests. Return the full file contents.", 1000),
    ("explain", "Explain how this server decides whether the picture reader runs on the GPU or on the CPU, and what "
                "it tells the user in each case. Cite the function and class names from server.py shown above.", 900),
    ("spanish", "Resume en español, en unos 250 palabras, cómo funciona el motor y por qué mantiene los expertos "
                "en la memoria de la tarjeta gráfica, según los documentos anteriores.", 800),
    ("json", "Extract the command-line flags and environment variables mentioned in the troubleshooting document "
             "above into a JSON array of objects with the keys name, kind (flag or env) and purpose. JSON only.", 1000),
    ("plan", "Plan how to add a `/v1/health/deep` endpoint to this server that reports engine state, free VRAM and "
             "the conversation cache counters. List the files and functions to touch, then write the handler.", 1000),
    ("edit", "Rewrite the `child_env` function from server.py with full type hints and a docstring that documents "
             "every environment variable it sets. Return only the complete function.", 900),
]


def context():
    parts = ["Project context follows. Read it; tasks come after.\n"]
    for rel, a, b in FILES:
        p = STRATA / rel
        if p.exists():
            parts.append(f"\n### FILE: {rel}\n```\n{p.read_text(errors='replace')[a:b]}\n```\n")
    return "".join(parts)


def summarize(rows):
    out = sum(r["out"] or 0 for r in rows)
    ms = sum(r["decode_ms"] or 0 for r in rows)
    drafts = sum(r["drafts"] or 0 for r in rows)
    acc = sum(r["accepted"] or 0 for r in rows)
    hits = [r["hit"] for r in rows if r.get("hit") is not None]
    return {"n": len(rows), "tokens": out, "tps": round(1000 * out / ms, 1) if ms else None,
            "accept": round(acc / drafts, 3) if drafts else None,
            "hit": round(st.mean(hits), 3) if hits else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--no-short", action="store_true")
    ap.add_argument("--no-agent", action="store_true")
    ap.add_argument("--no-warm", action="store_true")
    a = ap.parse_args()
    model = model_id()
    eng = get("/metrics")["engine"]
    res = {"label": a.label, "model": model, "power_profile": power_profile(),
           "engine": {k: eng.get(k) for k in ("version", "pool_workers", "expert_slots", "vram_free_mib",
                                              "conversation_cache_slots", "conversation_cache_min_tokens")}}
    thr0 = throttle_count()
    print(f"battery {a.label}: {model}  workers {eng.get('pool_workers')}  profile {res['power_profile']}", flush=True)

    if not a.no_warm:
        for _ in range(3):
            chat(model, [{"role": "user", "content": "Describe in 200 words how a suspension bridge carries load."}],
                 300, effort="none", temperature=0)

    if not a.no_short:
        res["short"] = {}
        for key, prompt, mx, temp, n in SHORT:
            rows = [chat(model, [{"role": "user", "content": prompt}], mx, effort="none", temperature=temp)[0]
                    for _ in range(n)]
            s = summarize(rows)
            s["median_tps"] = round(st.median(r["tps"] for r in rows), 1)
            s["all_tps"] = [r["tps"] for r in rows]
            res["short"][key] = s
            print(f"  short {key:11s} n={n} median {s['median_tps']:6.1f} t/s  accept {s['accept']}  hit {s['hit']}", flush=True)

    if not a.no_agent:
        base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": context()},
                {"role": "assistant", "content": "Understood. The project context is loaded; send the first task."}]
        sampler = Sampler()
        sampler.start()
        rows = []
        for p in range(a.passes):
            for key, text, mx in TASKS:
                msgs = base + [{"role": "user", "content": RESULT + text}]
                r, _, _ = chat(model, msgs, mx, effort="medium", tools=TOOLS)
                r.update({"task": key, "pass": p + 1})
                rows.append(r)
                print(f"  agent p{p+1} {key:8s} prompt {r['prompt_tokens']:>6,} read {r['read']:>6,} in {r['prompt_ms']/1000:5.2f} s"
                      f" | {r['out']:>5} tok @ {r['tps']:6.1f} t/s  accept {r['accepted']}/{r['drafts']}  hit {r['hit']}"
                      f"  {r['finish']}", flush=True)
        sysm = sampler.stop()
        cold = rows[0]
        warm_reads = [r for r in rows[1:] if (r["read"] or 0) < 2000]
        s = summarize(rows)
        s["by_task"] = {k: summarize([r for r in rows if r["task"] == k]) for k, _, _ in TASKS}
        s["cold_context"] = {"tokens_read": cold["read"], "prompt_s": round(cold["prompt_ms"] / 1000, 2),
                             "prefill_tps": round(1000 * cold["read"] / cold["prompt_ms"], 1) if cold["prompt_ms"] else None}
        s["turn_read_ms_median"] = round(st.median(r["prompt_ms"] for r in warm_reads), 0) if warm_reads else None
        s["turn_read_tokens_median"] = int(st.median(r["read"] for r in warm_reads)) if warm_reads else None
        s["system"] = sysm
        s["rows"] = rows
        res["agent"] = s
        print(f"  AGENT {a.label}: {s['tokens']:,} tokens at {s['tps']} t/s  accept {s['accept']}  hit {s['hit']}  "
              f"turn read {s['turn_read_ms_median']} ms ({s['turn_read_tokens_median']} tok)  "
              f"cold {s['cold_context']['tokens_read']:,} tok at {s['cold_context']['prefill_tps']} t/s", flush=True)
        print(f"  system: {sysm}", flush=True)

    thr1 = throttle_count()
    res["thermal_throttle_events"] = (thr1 - thr0) if thr0 is not None and thr1 is not None else None
    print(f"  thermal throttle events during the battery: {res['thermal_throttle_events']}", flush=True)
    (HERE / f"battery-{a.label}.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()

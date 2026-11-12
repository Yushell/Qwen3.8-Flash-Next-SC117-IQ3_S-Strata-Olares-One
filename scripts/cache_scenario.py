#!/usr/bin/env python3
"""Does a long conversation survive side requests and other long conversations?

Three long conversations (A, B, C: about 45K tokens each, built from the engine's own docs) and five short
side requests, in the order agent traffic produces them. For each turn the engine's record says how many
prompt tokens it reused and how many it had to read again. Run it on a config before and after a cache change.

Usage: python3 cache_scenario.py <label>      writes scenario-<label>.json next to this file
"""
import json
import pathlib
import sys

from lane_common import STRATA, chat, get, model_id

HERE = pathlib.Path(__file__).resolve().parent
DOCS = STRATA / "docs"


def doc(names, limit):
    text = ""
    for n in names:
        p = DOCS / n
        if p.exists():
            text += f"\n\n# {n}\n\n" + p.read_text(errors="replace")
    return text[:limit]


class Conv:
    def __init__(self, name, model, names, limit=165000):
        self.name, self.model = name, model
        self.first = "Reference document follows.\n" + doc(names, limit) + "\n\n"
        self.messages = [{"role": "system", "content": f"You are a careful technical assistant. Session {name}."}]

    def ask(self, question):
        content = (self.first if len(self.messages) == 1 else "") + "Question: " + question + " Answer in one sentence."
        self.messages.append({"role": "user", "content": content})
        rec, text, _ = chat(self.model, self.messages, 48, effort="none", temperature=0)
        self.messages.append({"role": "assistant", "content": text})
        return rec


SIDES = [
    ("title", "Generate a concise title of at most six words for a chat about tuning the expert cache of a local "
              "language model server on a laptop graphics card. Reply with the title only."),
    ("approve", "A coding agent wants to run this shell command: `rm -rf /tmp/build-cache-7731 && make -j8`. "
                "Reply APPROVE or DENY and nothing else."),
    ("summary", "Summarize in two sentences: " + "The harbour crane lifted forty containers before the tide turned, "
                "and the night shift logged each one by hand because the scanner was down. " * 6),
    ("title2", "Generate a concise title of at most six words for a chat about a Spanish invoice parser that fails "
               "on accented characters. Reply with the title only."),
    ("classify", "Classify the sentiment of this review as POSITIVE, NEGATIVE or MIXED, one word only: the keyboard "
                 "is superb, the battery is mediocre, and the fan never stops."),
]


def main():
    label = sys.argv[1]
    model = model_id()
    eng = get("/metrics")["engine"]
    cfg = {k: eng.get(k) for k in ("conversation_cache_slots", "conversation_cache_mib", "conversation_cache_min_tokens")}
    A = Conv("A", model, ["DETAILS.md"])
    B = Conv("B", model, ["BATCHING.md", "AMD_HIP.md", "MULTI_GPU.md"])
    C = Conv("C", model, ["TERMINAL.md", "INTEL.md", "A770_SPEED_SPEC.md"])
    steps = []

    def turn(conv, q, key=False):
        r = conv.ask(q)
        steps.append({"step": f"{conv.name} turn {len(conv.messages) // 2}", "key": key, **r})
        show(steps[-1])

    def side(i):
        name, prompt = SIDES[i]
        r, _, _ = chat(model, [{"role": "user", "content": prompt}], 40, effort="none", temperature=0)
        steps.append({"step": f"side:{name}", "key": False, **r})
        show(steps[-1])

    def show(s):
        frac = s["reused"] / s["prompt_tokens"] if s["prompt_tokens"] else 0
        flag = ""
        if s["key"]:
            flag = "  <== KEPT" if frac >= 0.9 else "  <== RE-READ"
        print(f"{s['step']:14s} prompt {s['prompt_tokens']:>7,} = {s['reused']:>7,} reused + {s['read']:>7,} read "
              f"in {s['prompt_ms']/1000:6.2f} s | first token {s['first_token_s']:6.2f} s | parked {s['cache']['parked']}"
              f"/{s['cache']['slots']} evictions {s['cache']['evictions']}{flag}", flush=True)

    print(f"scenario {label}: {model}  cache {cfg}")
    print("-- phase 1: one long conversation, then three short side requests")
    turn(A, "what is the conversation cache?")
    turn(A, "what does --kv-resident do?")
    side(0); side(1); side(2)
    turn(A, "what is a session file?", key=True)
    print("-- phase 2: two more long conversations and two side requests")
    turn(B, "what does --batch do?")
    turn(C, "what is the terminal monitor for?")
    turn(A, "what is the expert cache?", key=True)
    side(3); side(4)
    turn(B, "what does --batch-mtp need?", key=True)
    turn(C, "which Intel cards are covered?", key=True)
    turn(A, "what does --vram-reserve-mib do?", key=True)

    keys = [s for s in steps if s["key"]]
    kept = sum(1 for s in keys if s["reused"] / s["prompt_tokens"] >= 0.9)
    wait = sum(s["prompt_ms"] for s in keys) / 1000
    print(f"RESULT {label}: {kept} of {len(keys)} returning turns kept their conversation; "
          f"prompt time on those turns {wait:.1f} s total, worst {max(s['prompt_ms'] for s in keys)/1000:.1f} s")
    out = {"label": label, "model": model, "cache_config": cfg, "steps": steps,
           "returning_turns": len(keys), "kept": kept, "returning_prompt_s": round(wait, 2),
           "conversation_cache_after": get("/metrics")["conversation_cache"]}
    (HERE / f"scenario-{label}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

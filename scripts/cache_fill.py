#!/usr/bin/env python3
"""Fill the conversation cache close to its budget and prove the lane stays inside its memory cap.

Four different conversations of about 100K tokens each (the engine's docs, cut into four pieces), read once,
then each one visited again. Reports reuse on the second visit, the cache's bytes against its budget, and the
unit's cgroup (anon, shmem, file) against MemoryMax.

Usage: python3 cache_fill.py <label> [chars_per_conversation]      (UNIT=<systemd user unit>, default flashnext-serve.service)
"""
import json
import os
import pathlib
import subprocess
import sys

from lane_common import STRATA, chat, get, model_id

HERE = pathlib.Path(__file__).resolve().parent


def cgroup():
    cg = subprocess.run(["systemctl", "--user", "show", os.environ.get("UNIT", "flashnext-serve.service"), "-p", "ControlGroup", "--value"],
                        capture_output=True, text=True).stdout.strip()
    base = pathlib.Path("/sys/fs/cgroup" + cg)
    stat = dict(l.split() for l in (base / "memory.stat").read_text().splitlines())
    gib = lambda k: round(int(stat[k]) / 2 ** 30, 2)
    return {"current_gib": round(int((base / "memory.current").read_text()) / 2 ** 30, 2),
            "max_gib": round(int((base / "memory.max").read_text()) / 2 ** 30, 2),
            "anon_gib": gib("anon"), "shmem_gib": gib("shmem"), "file_gib": gib("file"),
            "inactive_file_gib": gib("inactive_file"),
            "oom_kills": (base / "memory.events").read_text().split("oom_kill ")[1].split()[0]}


def main():
    label = sys.argv[1]
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 330000
    model = model_id()
    names = sorted(p for p in (STRATA / "docs").glob("*.md"))
    blob = "".join(f"\n\n# {p.name}\n\n" + p.read_text(errors="replace") for p in names)
    convs = []
    for i in range(4):
        piece = blob[i * per:(i + 1) * per]
        convs.append([{"role": "system", "content": f"You are a careful technical assistant. Fill session {i + 1}."},
                      {"role": "user", "content": "Reference material follows.\n" + piece +
                       "\n\nQuestion: name one command-line flag mentioned above. Answer in one sentence."}])
    print(f"fill {label}: 4 conversations of {per:,} chars; before: {cgroup()}", flush=True)
    rows = []
    for visit in (1, 2):
        for i, msgs in enumerate(convs):
            if visit == 2:
                msgs.append({"role": "user", "content": "Question: name a second one. Answer in one sentence."})
            r, text, _ = chat(model, msgs, 48, effort="none", temperature=0)
            msgs.append({"role": "assistant", "content": text})
            cc = get("/metrics")["conversation_cache"]
            row = {"visit": visit, "conv": i + 1, **{k: r[k] for k in ("prompt_tokens", "reused", "read", "prompt_ms")},
                   "parked": cc["parked"], "cache_gib": round(cc["bytes"] / 2 ** 30, 2), "evictions": cc["evictions"]}
            rows.append(row)
            print(f"  visit {visit} conv {i+1}: prompt {r['prompt_tokens']:>7,} = {r['reused']:>7,} reused + {r['read']:>7,} read "
                  f"in {r['prompt_ms']/1000:6.2f} s | parked {cc['parked']}/{cc['slots']} cache {row['cache_gib']} GiB "
                  f"of {cc['budget_mib']/1024:.0f} evictions {cc['evictions']}", flush=True)
    cg = cgroup()
    kept = sum(1 for r in rows if r["visit"] == 2 and r["reused"] / r["prompt_tokens"] >= 0.9)
    print(f"RESULT {label}: {kept} of 4 kept on the second visit; cache peak {max(r['cache_gib'] for r in rows)} GiB; cgroup {cg}")
    (HERE / f"fill-{label}.json").write_text(json.dumps({"label": label, "rows": rows, "cgroup_after": cg,
                                                         "health": get("/health")}, indent=1) + "\n")


if __name__ == "__main__":
    main()

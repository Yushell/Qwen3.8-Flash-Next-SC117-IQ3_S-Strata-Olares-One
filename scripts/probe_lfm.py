#!/usr/bin/env python3
"""LFM2.5-24B-A2B smoke + speed + capability battery against a live llama-server."""
import json
import os
import time
import urllib.request

URL = os.environ.get("LMB_API", "http://127.0.0.1:8001/v1/chat/completions")
MODEL = os.environ.get("LMB_MODEL", "LFM2.5-24B-A2B-APEX-I-Balanced")


def call(payload, timeout=900):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t0


def report(name, d, dt):
    ch = d["choices"][0]
    msg = ch["message"]
    tim = d.get("timings") or {}
    u = d.get("usage") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    tc = msg.get("tool_calls") or []
    print(f"--- {name} ---")
    print(f"  finish={ch.get('finish_reason')}  wall={dt:.1f}s")
    print(f"  usage: prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')}")
    if tim:
        print(f"  decode {tim.get('predicted_per_second', 0):.1f} t/s ({tim.get('predicted_n')} tok)"
              f" | prefill {tim.get('prompt_per_second', 0):.0f} t/s ({tim.get('prompt_n')} tok)")
    print(f"  reasoning={len(reasoning)} chars  content={len(content)} chars")
    if tc:
        print(f"  TOOL_CALLS: {json.dumps(tc)[:500]}")
    if content:
        print(f"  content[:280]={content[:280]!r}")
    elif reasoning:
        print(f"  (no content) reasoning[:280]={reasoning[:280]!r}")
    print()


def p(msg, max_tokens, temp=0.1, **kw):
    base = {"model": MODEL, "messages": [{"role": "user", "content": msg}],
            "temperature": temp, "max_tokens": max_tokens}
    base.update(kw)
    return base


tests = [
    ("short-chat", p("In two sentences, explain what a hash table is and when to use one.", 1024)),
    ("code", p("Write a Python function merge_sorted(a, b) that merges two sorted lists. Return only the code.", 2048)),
    ("greedy-17x23", p("What is 17*23? Reply with just the number.", 1500, temp=0)),
    ("tool-call", p("What is the weather in Guadalajara right now?", 1024,
                    tools=[{"type": "function", "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
                        "parameters": {"type": "object",
                                       "properties": {"location": {"type": "string", "description": "City name"}},
                                       "required": ["location"]}}}])),
]

for name, payload in tests:
    try:
        d, dt = call(payload)
        report(name, d, dt)
    except Exception as e:
        print(f"--- {name} --- ERROR: {type(e).__name__}: {e}\n")

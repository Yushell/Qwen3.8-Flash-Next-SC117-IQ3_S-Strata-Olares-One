#!/usr/bin/env python3
"""Where a Strata lane's time really goes, read from the engine's own log. No request is sent.

Usage:
  cp /opt/strata/server-arm1-sc117.log engine.log         # the config's "log" file, not the unit's server.log
  python3 strata_traffic_report.py engine.log [SINCE=YYYY-MM-DD] [--trace 'YYYY-MM-DD HH:MM:SS' [FROM TO]]

Every number comes from the per-request summary lines the engine prints
(`strata serve: prompt N tokens = R reused + F read in T ms ..., G generated in T ms ...`), never wall clock.
The report answers the questions a decode benchmark cannot:
  * prompt reading versus generation as a share of engine time
  * the cost of a turn by how many fresh tokens it had to read
  * decode rate by context depth, with expert hit rate and draft acceptance
  * conversations that were pushed out of the conversation cache and read again from token 0,
    and what sat between the two turns (short side requests or another deep conversation)
  * cold reads caused by an engine start
`--trace` prints one engine start request by request with its conversation-cache events.
"""
import collections
import re
import statistics as st
import sys

RE_START = re.compile(r"^\[strata\] (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) engine started")
RE_REQ = re.compile(
    r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
    r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+), (\d+) checkpoints")
RE_HIT = re.compile(r"decode expert cache hit rate: ([\d.]+)% \((\d+) hits / (\d+) lookups\)")
RE_SKIP = re.compile(r"route tail skip: (\d+) missed experts skipped")
RE_KV = re.compile(r"KV streaming: ([\d.]+)% of \d+ block reads hit VRAM")
RE_PARK = re.compile(r"conversation cache: parked (\d+) tokens in ([\d.]+) ms; parked=(\d+).*?evictions=(\d+)")
RE_CC = re.compile(r"conversation cache: (.*)")


def pct(v, q):
    v = sorted(v)
    return v[int(q * (len(v) - 1))]


def load(path):
    epochs, cur = [], None
    for line in open(path, errors="replace"):
        m = RE_START.match(line)
        if m:
            cur = {"start": m.group(1), "reqs": [], "skip": 0, "evict": 0, "events": []}
            epochs.append(cur)
            continue
        if cur is None:
            continue
        m = RE_REQ.search(line)
        if m:
            g = m.groups()
            r = {"p": int(g[0]), "r": int(g[1]), "f": int(g[2]), "ms": int(g[3]), "g": int(g[5]),
                 "gms": int(g[6]), "tps": float(g[7]), "acc": int(g[8]), "drafts": int(g[9])}
            cur["reqs"].append(r)
            cur["events"].append(("req", r))
            continue
        m = RE_CC.search(line)
        if m:
            cur["events"].append(("cc", m.group(1).strip()))
            p = RE_PARK.search(line)
            if p:
                cur["evict"] = max(cur["evict"], int(p.group(4)))
                cur.setdefault("parks", []).append((int(p.group(1)), float(p.group(2))))
            continue
        if not cur["reqs"]:
            continue
        last = cur["reqs"][-1]
        m = RE_HIT.search(line)
        if m:
            last["hit"], last["hits"], last["lookups"] = float(m.group(1)), int(m.group(2)), int(m.group(3))
            continue
        m = RE_SKIP.search(line)
        if m:
            cur["skip"] = int(m.group(1))
            continue
        m = RE_KV.search(line)
        if m:
            last["kv"] = float(m.group(1))
    return epochs


def trace(epochs, stamp, lo, hi):
    ep = [e for e in epochs if e["start"].startswith(stamp)]
    if not ep:
        sys.exit(f"no engine start matching {stamp!r}")
    n = 0
    for kind, v in ep[0]["events"]:
        if kind == "req":
            n += 1
            if lo <= n <= hi:
                cold = "  <== COLD" if v["f"] > 8000 else ""
                print(f"#{n:<4d} prompt {v['p']:>7,} = {v['r']:>7,} reused + {v['f']:>7,} read {v['ms']/1000:6.2f} s"
                      f" | gen {v['g']:>5} @ {v['tps']:.1f} t/s{cold}")
        elif lo <= n <= hi:
            t = re.sub(r"(snapshot_bytes|reused_kv_bytes)=\d+ ?", "", v)
            print("        cc: " + t[:150])


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    path = args[0]
    epochs = load(path)
    if "--trace" in args:
        i = args.index("--trace")
        lo = int(args[i + 2]) if len(args) > i + 2 else 1
        hi = int(args[i + 3]) if len(args) > i + 3 else 10 ** 9
        trace(epochs, args[i + 1], lo, hi)
        return
    since = args[1] if len(args) > 1 else "0000"
    sel = [e for e in epochs if e["start"] >= since]
    reqs = [r for e in sel for r in e["reqs"]]
    if not reqs:
        sys.exit("no requests in range")
    print(f"{path}: {len(sel)} engine starts since {since}, {len(reqs)} requests")

    read_ms, gen_ms = sum(r["ms"] for r in reqs), sum(r["gms"] for r in reqs)
    fresh, reused, gen = sum(r["f"] for r in reqs), sum(r["r"] for r in reqs), sum(r["g"] for r in reqs)
    print("\n== engine time ==")
    print(f"prompt reading {read_ms/1000:8.0f} s ({100*read_ms/(read_ms+gen_ms):.0f}%)  "
          f"{fresh:,} fresh tokens, {100*reused/(fresh+reused):.1f}% of prompt tokens reused")
    print(f"generation     {gen_ms/1000:8.0f} s ({100*gen_ms/(read_ms+gen_ms):.0f}%)  {gen:,} tokens, {1000*gen/gen_ms:.1f} t/s overall")

    print("\n== cost of a turn by fresh tokens read ==")
    print(f"{'fresh tokens':14s} {'n':>5s} {'total s':>8s} {'median ms':>10s} {'p90 ms':>8s}")
    for lo, hi, name in [(1, 64, "1-64"), (65, 600, "65-600"), (601, 4000, "601-4K"),
                         (4001, 20000, "4K-20K"), (20001, 10 ** 9, ">20K")]:
        v = [r["ms"] for r in reqs if lo <= r["f"] <= hi]
        if v:
            print(f"{name:14s} {len(v):5d} {sum(v)/1000:8.0f} {st.median(v):10.0f} {pct(v, .9):8.0f}")

    print("\n== decode by context depth (>= 200 tokens generated) ==")
    print(f"{'context':18s} {'n':>4s} {'median t/s':>10s} {'p10':>6s} {'p90':>6s} {'hit %':>6s} {'accept':>7s} {'KV in VRAM %':>13s}")
    for lo, hi in [(0, 8000), (8000, 32768), (32768, 65536), (65536, 131072), (131072, 10 ** 9)]:
        v = [r for r in reqs if lo <= r["p"] < hi and r["g"] >= 200]
        if not v:
            continue
        t = [r["tps"] for r in v]
        hit = [r["hit"] for r in v if "hit" in r] or [0]
        acc = [r["acc"] / r["drafts"] for r in v if r["drafts"]] or [0]
        kv = [r["kv"] for r in v if "kv" in r] or [0]
        print(f"{lo:>7,}-{min(hi, 262144):<9,} {len(v):4d} {st.median(t):10.1f} {pct(t, .1):6.1f} {pct(t, .9):6.1f} "
              f"{st.median(hit):6.1f} {st.median(acc):7.3f} {st.median(kv):13.2f}")

    # cold reads from token 0: split by cause. A size seen cold more than once is a benchmark probe.
    sizes = collections.Counter(r["p"] for r in reqs if r["r"] == 0 and r["f"] > 8000)
    probes = [s for s, c in sizes.items() if c >= 2]

    def is_probe(p):
        return any(abs(p - s) <= 12 for s in probes)

    evicted, restart, probe_n, probe_ms, fresh_n, fresh_ms = [], [], 0, 0, 0, 0
    for e in sel:
        R, seen_deep = e["reqs"], False
        for k, q in enumerate(R):
            if q["r"] == 0 and q["f"] > 8000:
                if is_probe(q["p"]):
                    probe_n, probe_ms = probe_n + 1, probe_ms + q["ms"]
                else:
                    j = next((i for i in range(k - 1, -1, -1)
                              if R[i]["p"] > 8000 and not is_probe(R[i]["p"])
                              and q["p"] - 25000 <= R[i]["p"] + R[i]["g"] <= q["p"] + 300), None)
                    if j is not None:
                        between = R[j + 1:k]
                        short = sum(1 for x in between if x["p"] + x["g"] < 2048)
                        evicted.append((e["start"], k + 1, q, short, len(between) - short))
                    elif not seen_deep:
                        restart.append((e["start"], k + 1, q))
                    else:
                        fresh_n, fresh_ms = fresh_n + 1, fresh_ms + q["ms"]
            if q["p"] > 8000:
                seen_deep = True
    print("\n== prompts over 8K tokens read from token 0 ==")
    print(f"benchmark probes (a size seen cold twice or more): n={probe_n}, {probe_ms/1000:.0f} s")
    print(f"first deep prompt after an engine start:           n={len(restart)}, {sum(q['ms'] for _, _, q in restart)/1000:.0f} s")
    print(f"new deep prompt, nothing like it earlier:          n={fresh_n}, {fresh_ms/1000:.0f} s")
    print(f"a conversation seen earlier this start, read again: n={len(evicted)}, "
          f"{sum(x[2]['ms'] for x in evicted)/1000:.0f} s" +
          (f", median {st.median(x[2]['ms'] for x in evicted)/1000:.1f} s, max {max(x[2]['ms'] for x in evicted)/1000:.1f} s" if evicted else ""))
    only_short = [x for x in evicted if x[4] == 0 and x[3] >= 2]
    print(f"   pushed out by short (<2,048-token) requests alone: n={len(only_short)}, {sum(x[2]['ms'] for x in only_short)/1000:.0f} s")
    for s, k, q, sh, dp in evicted:
        print(f"   {s} req#{k:<4d} {q['p']:>7,} tokens, {q['ms']/1000:5.1f} s   between the two turns: {sh} short, {dp} deep")

    print("\n== per engine start ==")
    for e in sel:
        R = e["reqs"]
        if not R:
            continue
        look, hits = sum(r.get("lookups", 0) for r in R), sum(r.get("hits", 0) for r in R)
        parks = e.get("parks", [])
        print(f"{e['start']}  reqs {len(R):4d}  gen {sum(r['g'] for r in R):>8,}  hit {100*hits/max(look, 1):5.1f}%  "
              f"tail-skipped {100*e['skip']/max(look, 1):4.1f}% of lookups  evictions {e['evict']:3d}  "
              f"parks {len(parks):3d} ({sum(ms for _, ms in parks)/1000:.1f} s)")


if __name__ == "__main__":
    main()

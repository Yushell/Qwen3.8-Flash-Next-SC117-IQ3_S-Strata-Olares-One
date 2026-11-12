# Qwen3.8-Flash-Next · SC117 IQ3_S · Strata on Olares One

A serving recipe for **Qwen3.8-Flash-Next** (125B MoE, 3B active) on a single **Olares One** (RTX 5090 Laptop, 24 GB VRAM, 91 GB RAM): the **SC117 abliterated GSQ-RCO IQ3_S** pack served by the **Strata** engine at **262,144 context**, with int8 KV streaming, MTP speculative decode, vision, and an OpenAI-compatible API on `:8001`. It runs as a **systemd user unit with a supervisor** that owns readiness, load shape and a GPU ceiling.

Everything here was measured on this hardware class in October and November 2026. The engine beats llama.cpp on the same shards by 2.6x decode and 4-6x prefill; this repo is the config, the unit, the supervisor and the numbers.

**The weights are not hosted here** (79 GB). This repo is the serving kit and the measured record.

## Credits and attributions

Credit where it is due, all links:

- **The model**: [Qwen](https://huggingface.co/Qwen) publishes Qwen3.8-Flash-Next (125B MoE, ~3B active).
- **The abliterated GGUF pack**: [SC117](https://huggingface.co/SC117) publishes [Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF) (Apache-2.0), the IQ3_S shards, the Strata runtime pack and the BF16 mmproj this lane serves.
- **The engine**: [Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT), a C++/CUDA Flash-Next serving engine on [ggml](https://github.com/ggml-org/ggml), with the three-tier expert split (VRAM hot set, RAM all experts, SSD PLE), MTP drafts, KV streaming and a vision child. The GSQ-RCO quant format is the pack line's.
- **The ggml / llama.cpp lineage**: [ggml-org/ggml](https://github.com/ggml-org/ggml) and the [llama.cpp](https://github.com/ggml-org/llama.cpp) contributors, whose kernels and file formats Strata builds on.
- **The serve-layer sanitizer patch**: ours, from the lab's 2026-09-30 special-token smuggling work (see `patches/`).

The full attribution list is at the end of this README.

## Results at a glance

| Metric | Result |
|---|---|
| Served id | `qwen3.8-flash-next-sc117-iq3_s` (one request at a time, `total_slots: 1`) |
| Engine | Strata **0.1.59.1** (source build, v0.2.13.1 line + dev commits), sm_120, CUDA 13.0 |
| Decode, short prompts | **121-128 t/s** median (109.7-131.4 across runs); ~112-117 t/s after a 25K-token fill |
| Decode, agent work | **135-139 t/s** across six mixed tasks at a 34K-token context with tools and thinking on (`scripts/decode_battery.py`) |
| Prefill | **3,523-3,569 t/s** on a fresh 28.8K prompt |
| Long context | 28,843-token needle exact; 250K-token needle exact in a 51-86 s read (the 2026-10-14 battery) |
| Returning to a parked conversation | **0.2-0.6 s** for a 38K-53K-token conversation after side requests and two other long conversations; the same turns cost an 11-16 s re-read with a 2-slot cache |
| Conversations held at once | four of 91K-107K tokens (398K tokens) in 7.97 GiB of host RAM, each back in 0.5-1.1 s |
| Tool calls | `get_weather` exact through the OpenAI path; `tool_call_recovery` on by default since 0.2.8 |
| Vision | pass (BF16 mmproj, separate GPU vision child) |
| Load | ~30 s from restart to loaded (2026-11-10: 11:25:11 to 11:25:41); engine VRAM 22,786 MiB; process-tree RSS 53 GiB |
| Context | 262,144 tokens, int8 KV with 32,768 cells resident |
| Weights | SC117 IQ3_S: 55.06 GB + 28.80 GB GGUF shards, 907 MB BF16 mmproj, ~1.2 GB Strata runtime pack (79 GB total) |
| Host RAM while loaded | ~57 of 91 GB used with an empty conversation cache (~33 GB available); parked conversations add up to 10 GiB |

## The quantization choice

Three quantizations of the same weights were served through the same engine at 262,144 context with identical flags and measured with one battery (2026-10-14). The lane was kept on IQ3_S; it later moved to the SC117 **abliterated** IQ3_S pack (same bpw class) on 2026-10-27, which is what it serves now.

| Arm | Quantization | top-1 vs Q8_0 | chat decode | agent decode | 32K prefill | needle 32K/128K/250K | RAM free |
|---|---|---:|---:|---:|---:|---|---:|
| **IQ3_S (kept)** | GSQ-RCO IQ3_S, 3.33 bpw, 84 GB | 87.69% | **121.2 t/s** | 125.3 t/s | 3,139 t/s | 9/9 | 30 GB |
| IQ3_XXS | GSQ-RCO IQ3_XXS, 3.00 bpw, 76 GB | 84.46% | 119.9 t/s | **131.2 t/s** | 3,191 t/s | 9/9 | 31 GB |
| NVFP4 | NVFP4 GPTQ, 72 GB | **92.15%** | 71.8 t/s | 81.7 t/s | 2,955 t/s | 9/9 | 11 GB |

- The quality column is the publisher's `tools/quant_quality.py`: teacher-forced top-1 agreement against a Q8_0 run of the same fixed text set.
- IQ3_XXS is the fastest at agent decode; IQ3_S is the fidelity pick; NVFP4 is the quality ceiling at 40% of the decode and only 11 GB RAM left, so it is the documented switch, not the default.
- All three arms passed the same contract: clean OpenAI `tool_calls`, a real `merge_sorted`, exact greedy `17 * 23 = 391`, and 250K needle recall.

## Why Strata (the engine question)

| Engine | Verdict | Evidence |
|---|---|---|
| **Strata** | the engine | 80.7 t/s vs 29.1 for the same shards on llama.cpp (2026-09-29) |
| llama.cpp mainline | out | 29.1 t/s hybrid, RAM-bandwidth-bound |
| TensorFold | out | the Flash-Next recipe targets a 128 GB single device |
| vLLM | out | the NVFP4 checkpoint is 126-133 GB resident, no CPU expert tiering |

The engine splits the model three ways: the busiest experts stay resident in VRAM (ranked by a shipped routing profile, ~15.4 GiB), all experts live in RAM and the CPU computes a miss concurrently with the GPU, and the PLE/N-gram table lives on SSD. MTP drafts, int8 KV streaming and the vision child are built in.

## Serve it

Prerequisites: a 24 GB sm_120 card with ~90 GB of system RAM, CUDA 13.0, a Python venv for Strata, and Strata installed at `/opt/strata` (see its README; `./update.sh` builds the engine and never starts the `:8080` web chat).

1. **Stage the weights** (79 GB):

   ```bash
   hf download SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF \
     --local-dir /var/lib/strata-data/models/SC117 \
     --include "IQ3_S/*" "strata/*" "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
   ```

   The runtime pack belongs at `/var/lib/strata-data/packs/sc117_iq3_s/`. Its manifest is `config/conversions.json`; the files are `dense.bin`, `index.txt`, `native_experts.txt`, `tokenizer/`, built by Strata's `tools/iq_pack.py` from the two shards.

2. **Put the config** at `/opt/strata/strata-arm1-sc117-iq3s.json` (`config/`), adjust the paths, and copy the env example to `~/.config/flashnext-serve.env`.

3. **Install the unit and supervisor** (`unit/`):

   ```bash
   install -m 755 unit/flashnext-serve.sh ~/.local/bin/flashnext-serve.sh
   install -m 644 unit/flashnext-serve.service ~/.config/systemd/user/flashnext-serve.service
   systemctl --user daemon-reload
   systemctl --user start flashnext-serve
   journalctl --user -u flashnext-serve -f      # or: tail ~/.local/state/flashnext-serve/{supervisor,server}.log
   ```

4. **Gate it**:

   ```bash
   curl -s http://127.0.0.1:8001/health          # {"status":"ok","loaded":true,"max_context":262144}
   curl -s http://127.0.0.1:8001/v1/models       # qwen3.8-flash-next-sc117-iq3_s
   ./scripts/run_arm_battery.sh smoke qwen3.8-flash-next-sc117-iq3_s   # probes + needles + engine bench
   ```

The supervisor waits for `/health` `loaded: true` **and** a live engine child (a stale server can hold the port), checks the engine's own VRAM (14,000 MiB floor) and the **process tree's** RSS (35 GiB floor; measuring the Python wrapper alone passes on garbage), then applies `REASONING_EFFORT` as the lane default and enforces a whole-GPU ceiling (23,700 MiB, two strikes).

## The conversation cache (the setting that matters for agents)

An agent session is one long conversation that keeps coming back, interleaved with short side requests (a title, a command approval, a picture, a summary) and sometimes with other long conversations. The engine keeps one conversation in the GPU and parks the others in host RAM. If a parked conversation is evicted, its next turn re-reads the whole prompt: about 30 s per 100K tokens on this card.

With `--conversation-cache-slots 2` and every request parked, two short side requests in a row are enough to evict a long conversation. The config here uses **4 slots, 10240 MiB, and `--conversation-cache-min-tokens 2048`**, so short requests never take a slot.

Measured with `scripts/cache_scenario.py` (three conversations of 38K-53K tokens and five side requests, in the order agent traffic produces them; every figure is the engine's own per-request record):

| Returning turn | 2 slots, park everything | 4 slots, park 2,048+ tokens |
|---|---|---|
| A after three side requests | 0 reused, 52,777 read, 14.74 s | 52,752 reused, 0.20 s |
| A after B and C | 52,824 reused, 0.25 s | 52,824 reused, 0.61 s |
| B after two side requests | 0 reused, 50,727 read, 14.43 s | 50,697 reused, 0.22 s |
| C after that | 0 reused, 38,217 read, 10.97 s | 38,192 reused, 0.38 s |
| A after that | 0 reused, 52,914 read, 15.57 s | 52,885 reused, 0.42 s |
| **Prompt time on the five turns** | **56.0 s** | **1.8 s** |

- **Capacity** (`scripts/cache_fill.py`): four different conversations of 91K-107K tokens were held at once and each came back in 0.5-1.1 s (27.8-32.7 s cold). The cache peaked at 7.97 GiB of its 10; the unit's cgroup went from 49.1 to 57.3 GiB of anonymous memory under the 73.2 GiB `MemoryMax`, with no OOM kill.
- **A parked conversation costs** about 0.24 GB plus roughly 19 KB per token of host RAM (a 228K-token conversation is 4.6 GB). Size `--conversation-cache-mib` against the unit's `MemoryMax` minus the engine's own footprint.
- **Find out whether your traffic has the problem** with `scripts/strata_traffic_report.py <engine log>`: it reads the engine's per-request lines and lists every conversation that was evicted and read again from token 0, with what sat between the two turns. On the reference machine the 2-slot config showed 27 such re-reads in two weeks (median 26 s, worst 45 s), and prompt reading took 36% of the engine's time.
- A restart empties the cache, so each open conversation pays one cold read afterwards.

## Calibration and knobs

The engine's calibrator (`tools/calibrate.py`, run with the engine venv) measured these on the reference machine in 486 s:

- **`--pool-workers 23`** beat 15, 12, 7 and 6 (IQ3_S 115.7 to 121.2 t/s against 107.9 at 10 workers). Re-measured on 0.1.59.1 with `scripts/decode_battery.py`, one fresh engine start per arm (table below): 15 workers, which the engine's own start-up warning recommends for this CPU, is 2-7% slower on every probe, and `--pool-affinity p-cores` runs at about half speed.
- **`--pcie-frac 0`** beat the default 0.25 (126.4 vs 124.3 t/s median of 3); every higher share lost, so misses are computed on the CPU pool instead of copied.
- **`--spec-min-p 0.7`** confirmed best (131.4 vs 129.1 at 0.3 and 125.7 at 0.5).
- The adaptive expert tier stayed at the engine default (`1/160/0.97`, +1.5%, under the 3% keep threshold).
- **`STRATA_ROUTE_TAIL_SKIP=7` is on**: it is the engine's default for IQ packs since 0.1.42 (it is off by default only for NVFP4 experts, where the publisher measured it moving answers by about as much as the quantization does). On this recipe it skips 5-7% of expert lookups, about half of the cache misses. The publisher's cost on 12-16 GB cards is KL 0.003-0.006 against the previous behaviour with HumanEval unchanged; `STRATA_ROUTE_TAIL_SKIP=0` in the config's `env` block turns it off (speed cost not measured here).
- **The power profile does not matter**: Balanced and Performance measured the same (149.1 against 149.8 t/s mean over the five short probes), because under generation all 24 cores hold about 4.1 GHz and 59-63 W in both.

| Arm (0.1.59.1) | prose | code | JSON | sampled prose | Spanish | agent work | GPU |
|---|---:|---:|---:|---:|---:|---:|---|
| **23 workers** | 133.2 | 179.6 | 194.4 | 114.8 | 123.9 | **139.3** | 157 W, 96% |
| 15 workers | 125.8 | 170.4 | 190.6 | 107.2 | 116.0 | 133.7 | 154 W, 98% |
| P-cores only (7) | 71.1 | 75.4 | 93.3 | 50.2 | 64.8 | 59.8 | 119 W, 93% |

Tokens per second from the engine's own record; short probes are the median of 2-4 runs with thinking off, agent work is about 6,000 tokens per arm at an expert hit rate near 0.89.

Other served settings: `--max-context 262144`, `--kv int8 --kv-resident 32768`, `--vram-reserve-mib 1000`, `--conversation-cache-mib 10240 --conversation-cache-slots 4 --conversation-cache-min-tokens 2048`, `--expert-cache auto --prefill auto --spec 4`, MTP with `--mtp-draft-vocab`, vision on. Sampling and thinking are per request; the lane default effort is `medium` (`/settings` accepts `none|low|medium|high`; the request surface accepts the full vocabulary).

## Notes and gotchas

- **No API key is set** (`STRATA_API_KEY=none`). The server warns once, and anyone who can reach `:8001` can use the model. Keep the port on a trusted LAN or set a key in the env file and restart; every client then sends `Authorization: Bearer <key>`.
- **`TMPDIR` must not be an agent scratch dir.** The supervisor forces `/tmp`; a long-lived server inheriting a pruned scratch dir 503s every request (this is why the env file exists).
- **One server instance per GPU.** Each instance spawns its own engine and vision pair, and 24 GB fits one. The `:8080` web chat cannot run beside the `:8001` lane.
- **`MemoryMax=75000M` on the unit.** The engine keeps ~44-46 GiB of experts in host RAM plus ~3 GiB of pinned KV, and self-limits; the cap sits above the lane's own footprint and below a runaway. The GPU ceiling is the supervisor's, not systemd's.
- **Thinking tokens count toward `max_tokens`.** A small cap returns `finish=length` with empty `content`; give long-form requests hundreds to thousands, or send `reasoning_effort: none`.
- **A request must fit `prompt + max_tokens <= 262144`.** The server 4xxs instead of truncating, so a client's `max_tokens` reserves prompt headroom (32,768 gives prompts up to 229,376).
- **Servicing the engine:** re-apply the sanitizer patch after every `git pull` (`patches/patch_strata_sanitize.py`; it rewrites `serve/server.py`), then run the serve tests and restart the unit. The update flow and the pin to check (`engine/BUILD.json`) are in the Strata repo's docs.
- **If the display freezes or goes black once loaded**, raise `--vram-reserve-mib` to 1500 in the config (the card also drives a display). The shipped lane runs 1000 and stayed stable through every battery.
- **Several clients or agent side requests**: the config parks up to 4 conversations in 10240 MiB and only those of 2,048+ tokens. With 2 slots and no minimum, two short side requests evict a long conversation and its next turn re-reads the whole prompt (see the conversation cache section).
- **Restarting through a launcher script**: if your launcher rewrites `REASONING_EFFORT` in the env file on start, pass the level you want every time; an empty value leaves the engine's saved `/settings` default in charge.
- **Not affiliated** with any upstream authors; see Credits for the full attribution and licenses.

## What is not measured here

- The house real-work gate has not been run on this lane; these numbers are capacity and speed.
- The quality column is the publisher's measurement, not re-measured here (a Q8_0 reference run of the 188 GB model is needed).
- Vision is configured and passes a probe, but no image battery was run.
- Concurrency is one request at a time by design; parallel slots cost 10-25% per request on this card.
- The conversation-cache numbers come from scripted conversations and one two-week log, not from a long-run comparison of the two settings under the same traffic.

## Credits

Every piece this work builds on, with links:

- **Qwen3.8-Flash-Next** (model, design, training): [Qwen](https://huggingface.co/Qwen) ([model family](https://huggingface.co/Qwen)).
- **The abliterated GGUF pack, the Strata runtime pack and the BF16 mmproj**: [SC117](https://huggingface.co/SC117) ([repo](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF), Apache-2.0).
- **Strata** (the serving engine, MIT): [Niko1221/Strata](https://github.com/Niko1221/Strata) by [Niko1221](https://github.com/Niko1221).
- **ggml and llama.cpp** (the kernels and file formats Strata builds on, MIT): [ggml-org](https://github.com/ggml-org) ([ggml](https://github.com/ggml-org/ggml), [llama.cpp](https://github.com/ggml-org/llama.cpp)).
- **The GSQ-RCO quant format**: the pack line's, as published by SC117 with the Strata ecosystem.

**Licenses:** this repo's scripts, config, unit and notes are Apache-2.0 (see `LICENSE`). The weights and mmproj are Apache-2.0 at their source; Strata is MIT; ggml/llama.cpp are MIT. No weights are redistributed here.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

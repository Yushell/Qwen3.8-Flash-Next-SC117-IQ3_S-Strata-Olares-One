# Qwen3.8-Flash-Next · SC117 IQ3_S · Strata on Olares One

A production lane for **Qwen3.8-Flash-Next** (125B MoE, 3B active) on a single **Olares One** (RTX 5090 Laptop, 24 GB VRAM, 91 GB RAM): the **SC117 abliterated GSQ-RCO IQ3_S** pack served by the **Strata** engine at **262,144 context**, with int8 KV streaming, MTP speculative decode, vision, and an OpenAI-compatible API on `:8001`. It runs as a **systemd user unit with a supervisor** that owns readiness, load shape and a GPU ceiling.

Everything here was measured on the box in October and November 2026. The engine beats llama.cpp on the same shards by 2.6x decode and 4-6x prefill; this repo is the config, the unit, the supervisor and the numbers.

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
| Engine | Strata **0.1.59** (source build, v0.2.13.1 line + dev commits), sm_120, CUDA 13.0 |
| Decode, short prompts | **121-128 t/s** median (109.7-131.4 across runs); ~112-117 t/s after a 25K-token fill |
| Prefill | **3,523-3,569 t/s** on a fresh 28.8K prompt |
| Long context | 28,843-token needle exact; 250K-token needle exact in a 51-86 s read (the 2026-10-14 battery) |
| Tool calls | `get_weather` exact through the OpenAI path; `tool_call_recovery` on by default since 0.2.8 |
| Vision | pass (BF16 mmproj, separate GPU vision child) |
| Load | ~30 s from restart to loaded (2026-11-10: 11:25:11 to 11:25:41); engine VRAM 22,786 MiB; process-tree RSS 53 GiB |
| Context | 262,144 tokens, int8 KV with 32,768 cells resident |
| Weights | SC117 IQ3_S: 55.06 GB + 28.80 GB GGUF shards, 907 MB BF16 mmproj, ~1.2 GB Strata runtime pack (79 GB total) |
| Host RAM while loaded | ~57 of 91 GB used; ~33 GB available |

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

## Calibration and knobs

The engine's calibrator (`tools/calibrate.py`, run with the engine venv) measured these on this box in 486 s:

- **`--pool-workers 23`** beat 15, 12, 7 and 6 (IQ3_S 115.7 to 121.2 t/s against 107.9 at 10 workers).
- **`--pcie-frac 0`** beat the default 0.25 (126.4 vs 124.3 t/s median of 3); every higher share lost, so misses are computed on the CPU pool instead of copied.
- **`--spec-min-p 0.7`** confirmed best (131.4 vs 129.1 at 0.3 and 125.7 at 0.5).
- The adaptive expert tier stayed at the engine default (`1/160/0.97`, +1.5%, under the 3% keep threshold).
- **`STRATA_ROUTE_TAIL_SKIP=7`** is the one big speed lever left and is **off on purpose**: the publisher measured it moving answers by about as much as its quantization does.

Other served settings: `--max-context 262144`, `--kv int8 --kv-resident 32768`, `--vram-reserve-mib 1000`, `--conversation-cache-mib 6144 --conversation-cache-slots 2`, `--expert-cache auto --prefill auto --spec 4`, MTP with `--mtp-draft-vocab`, vision on. Sampling and thinking are per request; the lane default effort is `medium` (`/settings` accepts `none|low|medium|high`; the request surface accepts the full vocabulary).

## Notes and gotchas

- **No API key is set** (`STRATA_API_KEY=none`). The server warns once, and anyone who can reach `:8001` can use the model. Keep the port on a trusted LAN or set a key in the env file and restart; every client then sends `Authorization: Bearer <key>`.
- **`TMPDIR` must not be an agent scratch dir.** The supervisor forces `/tmp`; a long-lived server inheriting a pruned scratch dir 503s every request (this is why the env file exists).
- **One server instance per GPU.** Each instance spawns its own engine and vision pair, and 24 GB fits one. The `:8080` web chat cannot run beside the `:8001` lane.
- **`MemoryMax=75000M` on the unit.** The engine keeps ~44-46 GiB of experts in host RAM plus ~3 GiB of pinned KV, and self-limits; the cap sits above the lane's own footprint and below a runaway. The GPU ceiling is the supervisor's, not systemd's.
- **Thinking tokens count toward `max_tokens`.** A small cap returns `finish=length` with empty `content`; give long-form requests hundreds to thousands, or send `reasoning_effort: none`.
- **A request must fit `prompt + max_tokens <= 262144`.** The server 4xxs instead of truncating, so a client's `max_tokens` reserves prompt headroom (32,768 gives prompts up to 229,376).
- **Servicing the engine:** re-apply the sanitizer patch after every `git pull` (`patches/patch_strata_sanitize.py`; it rewrites `serve/server.py`), then run the serve tests and restart the unit. The update flow and the pin to check (`engine/BUILD.json`) are in the Strata repo's docs.
- **If the display freezes or goes black once loaded**, raise `--vram-reserve-mib` to 1500 in the config (the card also drives a display). The shipped lane runs 1000 and stayed stable through every battery.
- **Several clients at once**: `--conversation-cache-mib 8192` keeps each conversation; the shipped lane runs 6144 with 2 slots.
- **Not affiliated** with any upstream authors; see Credits for the full attribution and licenses.

## What is not measured here

- The house real-work gate has not been run on this lane; these numbers are capacity and speed.
- The quality column is the publisher's measurement, not re-measured here (a Q8_0 reference run of the 188 GB model is needed).
- Vision is configured and passes a probe, but no image battery was run.
- Concurrency is one request at a time by design; parallel slots cost 10-25% per request on this card.

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

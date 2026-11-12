# Changelog

## v1.1.0 · 2026-11-12

- **Conversation cache retuned for agent traffic**: `--conversation-cache-slots 4`, `--conversation-cache-mib 10240`,
  `--conversation-cache-min-tokens 2048` (was 2 slots, 6144 MiB, every request parked). In the scripted scenario the
  returning turns that kept their conversation went from 1 of 5 to 5 of 5 and their prompt time from 56.0 s to 1.8 s;
  four 100K-token conversations fit at once in 7.97 GiB.
- **Pool workers re-measured on Strata 0.1.59.1**: 23 stays. 15 workers (the engine's start-up recommendation) is 2-7%
  slower on every probe and `--pool-affinity p-cores` is about half speed.
- **Power profile measured**: Balanced and Performance are the same under generation.
- **Corrected**: `STRATA_ROUTE_TAIL_SKIP=7` is on for this recipe (the engine's default for IQ packs); v1.0.0 said it
  was off.
- **New reproducers** in `scripts/`: `cache_scenario.py`, `cache_fill.py`, `decode_battery.py`, `lane_common.py` and
  `strata_traffic_report.py` (where an engine log says the time goes, and every evict-and-reread event).
- Engine 0.1.59 to 0.1.59.1 (source build); decode and prefill unchanged (124.6 t/s median, 3,544-3,554 t/s on a fresh
  28.8K prompt). All of the above measured 2026-11-11.

## v1.0.0 · 2026-11-10

- Initial release: serving recipe for Qwen3.8-Flash-Next on an Olares One, served by Strata 0.1.59
  (v0.2.13.1 line, source build, sm_120, CUDA 13.0) from the SC117 abliterated GSQ-RCO IQ3_S pack at
  262,144 context on a single RTX 5090 Laptop (24 GB VRAM, 91 GB RAM).
- Serving kit: the arm config (23 pool workers, `--pcie-frac 0`, `--spec-min-p 0.7`, int8 KV with
  32,768 resident, MTP with draft vocab, vision, conversation cache 6144 x 2), the systemd user unit,
  the supervisor (readiness, load-shape and GPU-ceiling checks), the env example and the serve-layer
  sanitizer patch.
- Measured on the reference machine: decode 121-128 t/s on short prompts (109.7-131.4 across runs), prefill
  3,523-3,569 t/s on a fresh 28.8K prompt, 28,843-token needle exact, `391` greedy exact, tool calls
  exact, vision pass, load ~30 s at 22,786 MiB VRAM and 53 GiB tree RSS.
- Records the three-arm quantization comparison that picked IQ3_S (IQ3_XXS faster at agent decode,
  NVFP4 the quality ceiling at 40% of the decode) and why the engine is Strata (80.7 vs 29.1 t/s for
  the same shards on llama.cpp).

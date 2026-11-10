# Changelog

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

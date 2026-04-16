# Roadmap

`uttera-stt-vllm` is pre-alpha. This document tracks what is in flight and what is planned. Contributions welcome — file an issue or open a PR against any of the items below.

## v0.1.x — stabilise the scaffold

- [x] Single-process FastAPI wrapper around vLLM `AsyncLLM` + stock `OpenAIServingTranscription` / `OpenAIServingTranslation` handlers.
- [x] `/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/models`, `/health` aligned with `uttera-stt-hotcold`.
- [x] Redis self-registration (optional) for upstream router discovery.
- [x] Dockerfile, `docker-compose.yml`, systemd unit, `.env.example`.
- [ ] End-to-end smoke test script (`tests/smoke.sh`) — `curl` round-trip with a canned WAV.
- [ ] Concurrency benchmark ported from `whisper-vllm-test/bench_stt.py` into `tests/bench_400x.py`.
- [ ] CI workflow (lint + import-smoke + optional self-hosted GPU smoke), mirroring `uttera-tts-hotcold`.

## v0.2 — production hardening

- [ ] Structured logging with request IDs, audio duration, and queue-time breakdowns.
- [ ] Rate-limit knobs at the FastAPI layer (X-API-Key + tier quotas) so the vLLM server can sit behind the Uttera gatekeeper without duplicate middleware.
- [ ] Prometheus `/metrics` endpoint exposing `in_flight`, `ema_rps`, `total_completed`, `total_errors`, `vram_free_gb`, and vLLM internals where available.
- [ ] Health check that probes the engine (tiny synthetic audio decode) rather than just checking the flag.

## v0.3 — feature surface

- [ ] `response_format=verbose_json` with word- and segment-level timestamps (vLLM already supports this; we just need end-to-end validation).
- [ ] Streaming `/v1/audio/transcriptions?stream=true` end-to-end test and documentation.
- [ ] Model hot-swap endpoint (`POST /admin/model`) gated by admin token — unload + reload a different HuggingFace model without restarting the process.
- [ ] Explicit support matrix for Whisper variants (tiny/base/small/medium/large-v2/large-v3/turbo) — which actually work with vLLM's transcription path, which don't, measured on an RTX 5090.

## v1.0 — parity with the hotcold sibling

- [ ] Feature-parity checklist vs. `uttera-stt-hotcold`: same endpoints, same response shapes, same env var surface where applicable.
- [ ] Deployment notes (single-tenant vs. multi-tenant, GPU sizing guide, comparison table of hotcold vs. vllm throughput per GPU).
- [ ] Release 1.0.0 with a semver guarantee around `/health`, `/v1/audio/transcriptions`, and `/v1/audio/translations`.

## Not planned

- **Hot/cold worker pool.** vLLM's continuous batching replaces it; the whole point of this repo is to avoid that complexity.
- **Python-level audio preprocessing.** Delegated to vLLM's `speech_to_text` module so we do not diverge from upstream.
- **CPU-only inference.** This is the GPU-oriented sibling; CPU users should use `uttera-stt-hotcold` with a small Whisper model.

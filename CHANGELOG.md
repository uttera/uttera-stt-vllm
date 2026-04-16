# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-16

First scaffold release. Pre-alpha — active development. Expect breaking
changes until the API surface stabilises.

### Added
- **Single-process FastAPI server** that embeds vLLM's `AsyncLLM` engine
  in the same Python process. Concurrency handled entirely by vLLM's
  continuous batching — no hot/cold worker pool, no shared work queue,
  no per-request subprocess spawning. One GPU, one process, N parallel
  requests batched on the fly.
- **OpenAI-compatible endpoints** wired to vLLM's stock handlers:
  - `POST /v1/audio/transcriptions` → `OpenAIServingTranscription`
  - `POST /v1/audio/translations` → `OpenAIServingTranslation`
  - `GET /v1/models` (custom, Uttera-flavoured)
  - `GET /health` (custom, aligned with the sibling `uttera-stt-hotcold`
    schema — `status`, `version`, `engine`, `model`, `routing.load_score`,
    `routing.accepts_requests`, `metrics.*`)
- **Configurable model** via `WHISPER_MODEL` env var. Default:
  `openai/whisper-large-v3-turbo`. Any HuggingFace model compatible with
  vLLM's `task="transcription"` path should work.
- **Optional Redis self-registration** (parity with `uttera-stt-hotcold`
  and `uttera-tts-hotcold`). When `REDIS_URL` is set, the server
  publishes `{load_score, accepts_requests, host, port, version,
  engine="vllm", model, ts}` to `stt:nodes:{NODE_ID}` with a short TTL
  on a background tick. No-op when `REDIS_URL` is unset.
- **Engine tuning env vars**: `VLLM_DTYPE`, `VLLM_GPU_MEM_UTIL`,
  `VLLM_MAX_NUM_SEQS`, `VLLM_MAX_MODEL_LEN`, `VLLM_ENFORCE_EAGER`.
  Defaults validated on RTX 5090 + Whisper-large-v3-turbo.
- **Asset pre-provisioning**: `setup_assets.sh` runs
  `huggingface_hub.snapshot_download` for the configured model so the
  first request does not pay a ~1 GB cold-start download.
- **OSS scaffolding** shared with the rest of the Uttera stack:
  `LICENSE` (Apache-2.0), `NOTICE`, `README.md`, `API.md`, `HISTORY.md`,
  `AUTHORS.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CONTRIBUTING.md`,
  `CODEOWNERS`, `.github/` PR + issue templates, `.gitignore`,
  `docs/img/` banner/logo assets, Dockerfile + `docker-compose.yml` with
  NVIDIA GPU passthrough, systemd unit (`uttera-stt-vllm.yml`).

### Not implemented (yet)
- Benchmark harness in `tests/`. The informal test that produced the
  17.74 rps / 400-concurrent / 411× RTF number lived in
  `/home/claw/tmp/whisper-vllm-test/bench_stt.py`; it will be ported
  in a later release.
- CI workflow — the `uttera-tts-hotcold` pattern (lint + structure +
  optional GPU smoke) has not been carried over yet.
- Streaming chunked responses for `/v1/audio/transcriptions?stream=true`
  are passed through verbatim from vLLM but not yet exercised.

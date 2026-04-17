# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-04-17

Translation works again — and now works better than Whisper's own translate.

### Added
- **`/v1/audio/translations` pipeline via LibreTranslate**. The endpoint
  now transcribes the audio with Whisper (which works on turbo and every
  other multilingual Whisper variant) and then posts the text to a
  LibreTranslate instance for the final translation. This has three
  advantages over relying on Whisper's native `translate` task:
  1. Works with `openai/whisper-large-v3-turbo` (the v1.0.0 default),
     which was trained without the `translate` task.
  2. Supports arbitrary target languages via the `to_language` request
     parameter, not only English. LibreTranslate ships ~49 language
     pairs; Whisper's own translate only goes → English.
  3. Decouples transcription quality from translation quality; both can
     be tuned independently.
- New env vars:
  - `LIBRETRANSLATE_URL` — base URL of a running LibreTranslate instance
    (e.g. `http://localhost:5200`). **Required** for the translation
    endpoint; if unset, `/v1/audio/translations` returns HTTP 501 with a
    clear message.
  - `LIBRETRANSLATE_API_KEY` — optional key if your instance requires
    one.
  - `LIBRETRANSLATE_TIMEOUT_S` — timeout for the translation call
    (default 30 s).
- `to_language` request form field honoured on `/v1/audio/translations`.
  Defaults to `"en"` to preserve OpenAI-compatible behaviour when the
  caller does not specify one.

### Changed
- **`SERVER_VERSION` bumped to `1.1.0`.**
- `/v1/audio/translations` behaviour: when the detected source language
  equals `to_language`, LibreTranslate is skipped and the raw Whisper
  transcription is returned unchanged — saves a round-trip for the
  no-op case.
- On LibreTranslate failure (network, HTTP, malformed response), the
  endpoint returns HTTP 502 with the exception type and message. We do
  not fall back silently to the untranslated transcription, because
  that would leak source-language text under a response schema that
  promises the target language.

### Known limitations
- Language-code mapping between Whisper output and LibreTranslate is
  direct for most codes (ISO-639-1). For Chinese, Whisper emits `zh`
  while LibreTranslate expects `zh-Hans` / `zh-Hant`; a small mapping
  in `main_stt.py` handles this case. Other mismatches may exist for
  uncommon languages — file an issue if you hit one.

## [1.0.0] - 2026-04-17

First stable release. Functionally complete, lint- and structure-verified
via CI, benchmarked against a sibling hot/cold Whisper implementation on
LibriSpeech and an internal Spanish corpus. Suitable for cloud/multi-tenant
STT deployments on single GPUs with ≥32 GB VRAM.

### Added
- **GitHub Actions CI** (`.github/workflows/ci.yml`) with two always-on
  jobs (ruff lint + module syntax + endpoint declaration check) and one
  optional GPU smoke job gated to self-hosted runners.
- **Positioning section** in `README.md` with explicit VRAM thresholds
  and a pointer to the sibling [`uttera-stt-hotcold`](https://github.com/uttera/uttera-stt-hotcold)
  for 8–24 GB deployments.
- **Published reproducible benchmarks** at [`uttera-benchmarks`](https://github.com/uttera/uttera-benchmarks):
  - 18.19 rps sustained on burst N=1024 (zero failures) against
    LibriSpeech test-clean.
  - 18.19 rps on the internal Spanish WAV corpus (clips 13–27 s, zero
    failures) — duration-insensitive, unlike the hotcold sibling which
    saturates at N=1024 on the same corpus.
  - Single-request p50 of 82 ms on ~14 s LibriSpeech clips, 120 ms on
    ~20 s Spanish clips.

### Changed
- **`SERVER_VERSION` bumped to `1.0.0`.** Reflected in `/health`,
  `/v1/models`, and the FastAPI app metadata.
- **`requirements.txt`** now pins `vllm[audio]>=0.19.0,<0.20.0` (was
  plain `vllm`). The `[audio]` extra pulls in `resampy`, `av`, `scipy`,
  `soundfile`, and `mistral_common[audio]`, without which
  `/v1/audio/transcriptions` raises HTTP 500 on every request because
  vLLM's internal `load_audio()` cannot resample to 16 kHz.

### Fixed
- Import paths updated to the real vLLM 0.19 layout:
  - `TranscriptionRequest` / `TranslationRequest` now imported from
    `vllm.entrypoints.openai.speech_to_text.protocol` (was the
    non-existent `vllm.entrypoints.openai.protocol`).
  - `OpenAIServingModels` / `BaseModelPath` from
    `vllm.entrypoints.openai.models.serving` (was the non-existent
    `vllm.entrypoints.openai.serving_models`).
- Dropped `task="transcription"` from `AsyncEngineArgs`: vLLM 0.19
  infers the runner from the Whisper model architecture; the kwarg is
  not accepted.
- Dropped `model_config=...` from `OpenAIServingModels.__init__`;
  vLLM 0.19 does not accept it.
- Guarded `serving_models.init_static_loras()` behind `hasattr` so the
  code tolerates the method being dropped in a future vLLM patch.

### Known limitations
- **`openai/whisper-large-v3-turbo` does not support translation to
  English.** Upstream HuggingFace model card: the turbo variant was
  trained without the translation task. `/v1/audio/translations`
  returns the transcription unchanged on turbo; for real translation,
  set `WHISPER_MODEL=openai/whisper-large-v3`. This is a model-level
  limitation, not a bug in this server.
- The GPU smoke CI job is defined but gated off (`if: false`) because
  no self-hosted GPU runner is configured yet.

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

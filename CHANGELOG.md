# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-04-21

Prometheus `/metrics` endpoint. Additive only — all existing
endpoints unchanged.

### Added

- **`GET /metrics`** — OpenMetrics-format scrape endpoint using the
  default `prometheus_client` global registry. Scrape with Telegraf's
  `inputs.prometheus` plugin, Prometheus itself, or any other
  OpenMetrics-compatible consumer.
- **HTTP-level metrics** (bounded cardinality — unknown paths fall
  into `"other"`):
  - `uttera_stt_requests_total{endpoint, method, status}` — Counter
    of HTTP requests.
  - `uttera_stt_request_duration_seconds{endpoint, method}` —
    Histogram of HTTP RTT, buckets 25 ms → 60 s.
  - `uttera_stt_inflight_requests` — Gauge reflecting `_in_flight`.
- **STT-specific metrics**:
  - `uttera_stt_transcriptions_total{response_format}` — Counter
    broken down by the requested `response_format` (`json`, `text`,
    `verbose_json`, `srt`, `vtt`).
  - `uttera_stt_translations_total{mode, response_format}` — Counter
    broken down by translation path (`libretranslate` — all traffic
    today — plus the `response_format` for output shape).
  - `uttera_stt_audio_seconds_total{endpoint}` — Counter summing
    the audio seconds successfully processed. Useful as a billing /
    throughput proxy; tapped from the duration field in every
    successful `verbose_json` response.
  - `uttera_stt_inference_duration_seconds{op}` — Histogram per
    model call kind: `whisper_transcribe` (vLLM Whisper call) and
    `libretranslate` (the LibreTranslate HTTP round-trip). Lets
    dashboards show GPU time vs external-dependency time separately.
- **State gauges** (refreshed on every `/metrics` scrape so they're
  always current):
  - `uttera_stt_engine_ready` — 1 when the vLLM engine has passed
    warmup, 0 otherwise.
  - `uttera_stt_libretranslate_configured` — 1 if `LIBRETRANSLATE_URL`
    was set at startup, 0 otherwise.
- **`uttera_stt_errors_total{type}`** — Counter of errors by cause.
  Types: `upstream` (vLLM returned an ErrorResponse — mapped to the
  real HTTP code), `model` (uncaught inference exception),
  `libretranslate` (LibreTranslate HTTP/network failure). Generic
  4xx failures are already visible via the `status` label on
  `requests_total`.
- **`uttera_stt_build_info{version, engine, model}`** — Gauge set
  to `1` with the running `SERVER_VERSION`, engine (`vllm`), and
  the actual `WHISPER_MODEL` as labels, so dashboards can show
  "version + model in the field" without a separate lookup.

### Changed

- **New runtime dep**: `prometheus-client>=0.20.0`. Fully self-
  contained; uses the default global registry.
- **`SERVER_VERSION` bumped to `1.4.0`.**
- **New runtime dep**: `httpx>=0.27.0` pinned in requirements.txt
  (was previously pulled in only transitively from vllm[audio];
  making it explicit so the LibreTranslate path doesn't break if
  vllm drops the transitive dep in a future release).

### Not changed

- `/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/models`,
  `/health` behave identically to v1.3.0. The `/health` body still
  reports `in_flight` / `total_completed` / `total_errors` for
  callers that have them hardcoded; the Prometheus counters are the
  new canonical observability path.

## [1.3.0] - 2026-04-18

### Changed

- **Default port migrated from `5000` → `9005`** in lockstep with the
  sibling `uttera-stt-hotcold` v2.3.0. Canonical Uttera-stack scheme:
  STT services on `9005`, TTS services on `9004`. The Gatekeeper and
  clients route by service family (STT/TTS) without needing to know
  which backend (hotcold vs vllm) is active behind the port.

  **Why move off `5000`:** collision with macOS AirPlay Receiver
  (since Monterey) and with Docker Registry v2 default. The
  `9000-9099` range is IANA "User Ports" without canonical
  assignment and is collision-free on mainstream systems.

  Artefacts updated: `PORT` env default in `main_stt.py`, `Dockerfile`
  `EXPOSE`/`CMD`, `docker-compose.yml` port mapping and healthcheck,
  `.env.example`, `README.md`, `API.md`, `.github/workflows/ci.yml`,
  issue template health-probe URL.

### Migration

Deployments that already override `PORT` via env var: no change
required. Deployments using the old default (`:5000`):
- Repoint your Gatekeeper / reverse proxy at `:9005`.
- Or set `PORT=5000` in your env to preserve the old endpoint.
- Docker users: update your `-p` flag or `docker-compose.yml`.

### Related

- `uttera-stt-hotcold` v2.3.0 adopts the same `9005` port.
- `uttera-tts-hotcold` v2.3.0 and `uttera-tts-vllm` v1.3.0 adopt
  `9004` for the TTS pair.

## [1.2.0] - 2026-04-18

OpenAI-compatibility polish sweep. Driven by a full endpoint validation
run (128-concurrent burst + 16 single-shot feature tests across
`/v1/audio/transcriptions` and `/v1/audio/translations`). Seven rough
edges fixed. All backward-compatible — existing clients see no change;
strict clients now get the documented OpenAI contract.

### Added

- **Real SRT / WebVTT / verbose_json responses** on both transcriptions
  and translations. The wrapper now always requests `verbose_json` from
  vLLM internally and renders the final wire format itself:
  - `json` → compact `{"text": "..."}` (OpenAI-compact; unchanged default).
  - `text` → plain text body with `text/plain; charset=utf-8` Content-Type.
  - `verbose_json` → full vLLM result (text + segments + language +
    words + usage).
  - `srt` → SubRip subtitle file (`HH:MM:SS,mmm`, `application/x-subrip`).
  - `vtt` → WebVTT subtitle file (`HH:MM:SS.mmm`, `text/vtt`).
  Previously `srt` and `vtt` returned HTTP 200 with a vLLM error body
  (`{"error": {"code": 400, ...}}`); `text` returned JSON instead of a
  plain-text body.
- **Per-segment LibreTranslate translations**. When the translation
  endpoint is called with `response_format=srt|vtt|verbose_json`, each
  transcribed segment is individually translated through LibreTranslate
  (in parallel via `asyncio.gather`) so the resulting subtitles keep
  their original timings aligned to the translated text. For `text` /
  `json` formats a single LibreTranslate call is still used.
- **`X-Translation-Mode: libretranslate` response header** on
  `/v1/audio/translations`. Matches the sibling `uttera-stt-hotcold`
  v2.2.0 for observability symmetry.
- **`HEAD /health` support**. Load balancers and uptime probes that use
  HEAD no longer get HTTP 405.
- **Opt-in `CORSMiddleware`** gated on `CORS_ALLOW_ORIGINS` env var
  (comma-separated list, or `"*"`). Disabled by default — API-first
  deployments don't need CORS, and enabling it unconditionally broadens
  the attack surface.
- **`temperature` range validation** in `[0.0, 1.0]` per the OpenAI
  spec. Out-of-range values return HTTP 422 with an explicit message.
  vLLM itself accepts any float and silently produces garbage for
  values > 1 (previously responded 200 with gibberish) and raises an
  internal error for negatives (previously responded 500).

### Changed

- **`SERVER_VERSION` bumped to `1.2.0`.**
- **vLLM-side errors now surface with the real status code.** The
  wrapper now looks at `result.error.code` (vLLM's `ErrorResponse`
  shape) instead of a non-existent top-level `result.code`, so a
  rejected request no longer comes back as HTTP 200 with an error body.
- **Whisper "Unsupported language: XX" errors** surface as HTTP 400
  with the actual message (including the list of supported codes),
  instead of HTTP 500 "Internal Server Error".
- **Non-audio / undecodeable file bodies** return HTTP 400 with
  `"Failed to decode audio body — not a valid audio stream or
  unsupported codec."` instead of HTTP 500.

### Verified

- 128-concurrent burst on Whisper-large-v3-turbo: 128/128 OK, 0
  failures, 16.6 rps (within 5% of v1.1.0 baseline — no regression).
- CORS preflight + actual POST emit the expected headers when
  `CORS_ALLOW_ORIGINS` is set.
- ES → FR and ES → ZH translation pipelines produce correct output on
  the internal Spanish corpus, including segment-by-segment SRT/VTT
  with preserved 0:00→0:07.64, 0:08.80→0:16.96 timings.

### Backward compatibility

- Every previously-valid request continues to return the same shape.
- Default `response_format=json` behaviour is unchanged.
- CORS is disabled by default; existing deployments see no header
  changes unless they opt in.
- The `usage: {"type": "duration", "seconds": N}` field that vLLM
  populated in v1.1.0 compact-JSON responses is dropped from the
  default `json` format (OpenAI-compact does not include it) but is
  preserved in `verbose_json`.

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

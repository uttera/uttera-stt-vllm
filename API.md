# uttera-stt-vllm API

Uttera STT vLLM exposes an OpenAI-compatible speech-to-text surface,
built on top of vLLM's native Whisper path. Until v1.1 the wrapper
forwarded requests verbatim to vLLM's `speech_to_text` module. Since
v1.2.0 the wrapper **always requests `verbose_json` from vLLM
internally** and renders the final wire format itself — this gives us
real SRT / WebVTT subtitles, a plain-text body for `text`, and
per-segment translations aligned to original timings. The request
contract remains OpenAI-compatible.

Base URL (default): `http://localhost:9005`

## `POST /v1/audio/transcriptions`

Transcribe audio in the **source language**. Mirrors OpenAI's `/v1/audio/transcriptions`.

**Request** — `multipart/form-data`:

| Field | Type | Notes |
|---|---|---|
| `file` | file (required) | Audio file. Any format ffmpeg / librosa can decode (wav, mp3, flac, m4a, ogg, …). Undecodeable / non-audio bodies → HTTP 400 with a typed decode error. |
| `model` | string | Ignored; the model is fixed by `WHISPER_MODEL` at startup. Sent for OpenAI-client compatibility. |
| `language` | string | ISO-639-1 code (`en`, `es`, `fr`, …). Omit to let Whisper auto-detect. Unsupported codes → HTTP 400 with the list of supported languages (was generic HTTP 500 before v1.2.0). |
| `prompt` | string | Optional initial prompt (style, vocabulary hints). |
| `response_format` | string | One of `json` (default), `text`, `verbose_json`, `srt`, `vtt`. Anything else → HTTP 422. The wrapper renders all five formats itself since v1.2.0 — SRT uses `HH:MM:SS,mmm` timecodes with `application/x-subrip`; VTT uses `HH:MM:SS.mmm` with a `WEBVTT` header; `text` emits `text/plain; charset=utf-8`. |
| `timestamp_granularities` | list | `["word"]`, `["segment"]`, or both. Only meaningful with `response_format=verbose_json`. |
| `temperature` | float | Default `0.0`. Valid range `[0.0, 1.0]` (OpenAI spec) — out-of-range → HTTP 422 (v1.2.0). |
| `stream` | bool | `true` returns Server-Sent Events directly from vLLM; default `false`. See the Streaming section below for the constraint on response formats when streaming. |

**Response** (non-stream, `response_format=json`):

```json
{
  "text": "La tecnología impulsa el progreso humano..."
}
```

**Response** (non-stream, `response_format=verbose_json`):

```json
{
  "duration": "6.23",
  "language": "es",
  "text": "...",
  "segments": [{ "id": 0, "start": 0.0, "end": 3.1, "text": "...", ... }],
  "words":    [{ "word": "La", "start": 0.00, "end": 0.12 }, ...]
}
```

**Example**:

```bash
curl -X POST http://localhost:9005/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "language=es" \
  -F "response_format=json"
```

## `POST /v1/audio/translations`

Translate audio to a **target language** (default English, matching OpenAI). Mirrors OpenAI's `/v1/audio/translations` and extends it with an explicit `to_language` field.

Implementation: Whisper transcribes in the source language (auto-detected or forced via `language`), then the text is posted to a LibreTranslate instance for the final translation. This:

- works with any multilingual Whisper model — including `whisper-large-v3-turbo`, which was trained without the native `translate` task;
- supports target languages other than English, which Whisper native translate does not.

**Per-segment translations** (v1.2.0): when the caller requests
`response_format=srt|vtt|verbose_json`, each transcribed segment is
individually translated through LibreTranslate in parallel (via
`asyncio.gather`), so the resulting subtitles keep their original
timings aligned to the translated text. For `response_format=json`
or `text`, a single LibreTranslate call is used on the full text.

**Response header:** `X-Translation-Mode: libretranslate` is emitted
whenever the LibreTranslate path runs (i.e. always, since the native
Whisper-translate path is not used by this server). The header is
exposed to browser clients via CORS.

### Extra request fields vs. transcriptions

| Field | Type | Notes |
|---|---|---|
| `to_language` | string | ISO-639-1 target (`en`, `es`, `fr`, `de`, …). Defaults to `"en"` for OpenAI-compatibility. |
| `language` | string | Optional **source** language hint for Whisper. Omit to auto-detect. |

### Examples

```bash
# OpenAI-default behaviour: Spanish → English
curl -X POST http://localhost:9005/v1/audio/translations \
  -F "file=@spanish-clip.wav"

# Spanish → French
curl -X POST http://localhost:9005/v1/audio/translations \
  -F "file=@spanish-clip.wav" \
  -F "to_language=fr"

# Same source and target — LibreTranslate is skipped, transcription returned
curl -X POST http://localhost:9005/v1/audio/translations \
  -F "file=@spanish-clip.wav" \
  -F "to_language=es"
```

### Errors

| HTTP | When |
|---|---|
| 400 | Undecodeable audio body, unsupported Whisper language code (with the message naming valid codes), or vLLM surfacing a client-level rejection (status extracted from `result.error.code` — fixed in v1.2.0 from the prior silent-HTTP-200-with-error-body bug). |
| 422 | `response_format` not in the supported set, or `temperature` out of `[0.0, 1.0]`. |
| 501 | `LIBRETRANSLATE_URL` is not configured. The response `detail` tells the caller to either set it or run a Whisper model with native translate support. |
| 502 | LibreTranslate call failed (network, HTTP error, or malformed response). We do **not** fall back silently to the untranslated transcription, because that would leak source-language text under a response schema that promises the target language. |
| 503 | Engine not ready (startup) or fatal engine error. `/health` returns the same status while in this state. |

## `GET /v1/models`

```json
{
  "object": "list",
  "data": [{
    "id": "whisper-1",
    "object": "model",
    "created": 1776379200,
    "owned_by": "uttera"
  }]
}
```

`id` is controlled by the `SERVED_MODEL_NAME` env var (default `whisper-1`, matching OpenAI's public identifier so unmodified SDK clients work).

## `GET /health` and `HEAD /health`

Liveness + throughput + routing signal. Responds `200` when the
engine is ready, `503` during startup or after a fatal engine error.
`HEAD` returns the same headers with an empty body — useful for
uptime probes that don't want to parse JSON.

```json
{
  "status": "ok",
  "version": "1.3.0",
  "engine": "vllm",
  "model": "openai/whisper-large-v3-turbo",
  "served_as": "whisper-1",
  "engine_ready": true,
  "engine_error": null,
  "routing": {
    "load_score": 0.12,
    "accepts_requests": true
  },
  "metrics": {
    "in_flight": 8,
    "total_completed": 1042,
    "total_errors": 0,
    "ema_rps": 17.4,
    "vram_free_gb": 18.23,
    "max_num_seqs": 64,
    "max_model_len": 448,
    "gpu_memory_utilization": 0.9
  }
}
```

The `routing` block matches `uttera-stt-hotcold`'s `/health` so a
shared upstream router can consume both backends with one schema.

## Streaming

`stream=true` emits Server-Sent Events directly from vLLM — the
wrapper does **not** re-render streaming chunks into SRT/VTT/text.
Consequently, when `stream=true` the effective `response_format`
must be one vLLM itself supports natively (`json` / `verbose_json`).
Requesting `response_format=srt|vtt|text` together with `stream=true`
produces vLLM's stream of verbose_json events; subtitle rendering is
a non-streaming feature.

## CORS

Disabled by default — this server is API-first, typically consumed
by backend-to-backend callers or served through the Uttera
gatekeeper.

To enable browser-origin access, set `CORS_ALLOW_ORIGINS` to a
comma-separated list of origins (or `*` for permissive):

```bash
CORS_ALLOW_ORIGINS="https://app.uttera.ai,https://dev.uttera.ai"
# or:
CORS_ALLOW_ORIGINS="*"
```

Methods, headers, and credentials follow the FastAPI
`CORSMiddleware` defaults (allow all methods, allow all headers,
credentials enabled). The `X-Translation-Mode` response header is
explicitly exposed so browser clients can read it from JavaScript.

## Authentication

No authentication in this repo by design. Deploy behind the Uttera
gatekeeper (or any reverse proxy) for API keys, quotas, and rate
limits.

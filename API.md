# API

Uttera STT vLLM exposes an OpenAI-compatible speech-to-text surface. The full request/response shape is defined by vLLM's `speech_to_text` module (we forward requests verbatim); this document summarises the contract.

Base URL (default): `http://localhost:9005`

## `POST /v1/audio/transcriptions`

Transcribe audio in the **source language**. Mirrors OpenAI's `/v1/audio/transcriptions`.

**Request** — `multipart/form-data`:

| Field | Type | Notes |
|---|---|---|
| `file` | file (required) | Audio file. Any format ffmpeg / librosa can decode (wav, mp3, flac, m4a, ogg, …). |
| `model` | string | Ignored; the model is fixed by `WHISPER_MODEL` at startup. Sent for OpenAI-client compatibility. |
| `language` | string | ISO-639-1 code (`en`, `es`, `fr`, …). Omit to let Whisper auto-detect. |
| `prompt` | string | Optional initial prompt (style, vocabulary hints). |
| `response_format` | string | `json` (default), `text`, `srt`, `vtt`, or `verbose_json`. |
| `timestamp_granularities` | list | `["word"]`, `["segment"]`, or both. Only meaningful with `response_format=verbose_json`. |
| `temperature` | float | Default `0.0`. |
| `stream` | bool | `true` returns Server-Sent Events; default `false`. |

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
| 501 | `LIBRETRANSLATE_URL` is not configured. The response `detail` tells the caller to either set it or run a Whisper model with native translate support. |
| 502 | LibreTranslate call failed (network, HTTP error, or malformed response). We do **not** fall back silently to the untranslated transcription, because that would leak source-language text under a response schema that promises the target language. |

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

## `GET /health`

Liveness + throughput + routing signal. Responds `200` when the engine is ready, `503` during startup or after a fatal engine error.

```json
{
  "status": "ok",
  "version": "0.1.0",
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

The `routing` block matches `uttera-stt-hotcold`'s `/health` so a shared upstream router can consume both backends with one schema.

## Streaming

Pass `stream=true` with any response format other than `json`/`verbose_json` to receive Server-Sent Events. The SSE body is produced directly by vLLM and is not re-formatted by this wrapper.

## Authentication

No authentication in this repo by design. Deploy behind the Uttera gatekeeper (or any reverse proxy) for API keys, quotas, and rate limits.

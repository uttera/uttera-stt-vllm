# uttera-stt-vllm

<p align="center">
  <a href="https://uttera.ai">
    <img src="docs/img/banner.png" alt="uttera.ai — The voice layer for your AI" width="800">
  </a>
</p>

High-throughput Speech-to-Text server built on **vLLM continuous batching**.
Whisper-large-v3-turbo today, room for future Transformer STT backends.

> **Status**: v1.1.0 — stable. `/v1/audio/translations` now works with
> Whisper-turbo via a LibreTranslate post-processing pipeline and
> supports arbitrary target languages (not just English). See
> [CHANGELOG.md](CHANGELOG.md) and [ROADMAP.md](ROADMAP.md).

## Positioning

| Use case | This repo | Sibling repo |
|---|---|---|
| Cloud, multi-tenant, large GPU (≥24 GB) | ✅ [uttera-stt-vllm](https://github.com/uttera/uttera-stt-vllm) | — |
| Home-lab, personal, small/mid GPU (8–16 GB) | — | [uttera-stt-hotcold](https://github.com/uttera/uttera-stt-hotcold) |

**Choose `uttera-stt-vllm` when**:
- You transcribe hours of audio per day across many concurrent streams.
- You want continuous batching to maximise GPU utilisation.
- You have large-VRAM GPUs dedicated to inference.
- **You have 32 GB+ of VRAM** (vLLM reserves ~22–29 GB at startup
  depending on `gpu_memory_utilization`; below 32 GB total you either
  run out of headroom or lose the batching advantage that justifies
  the reservation).

**Choose `uttera-stt-hotcold` when**:
- You have consumer GPUs (RTX 4070, 4080) and transcribe occasionally.
- Personal or single-user deployment.
- You want to share the GPU with other workloads.
- **You have 8–24 GB of VRAM.** vLLM does not fit comfortably in this
  range: at 8–16 GB the KV cache is too small for continuous batching
  to beat hotcold; at 16–24 GB vLLM works but reserves 11–22 GB
  permanently, wasting the co-location flexibility that is hotcold's
  reason to exist on mid-sized GPUs.

See [`uttera-benchmarks`](https://github.com/uttera/uttera-benchmarks)
for reproducible head-to-head numbers across four load profiles
(latency, burst up to N=1024, sustained) and two corpora (LibriSpeech
test-clean and an internal Spanish WAV corpus).

## Architecture

Built on [vLLM](https://github.com/vllm-project/vllm) (native Whisper
support since v0.6.6). A **single Python process** hosts:

- `vllm.v1.engine.async_llm.AsyncLLM` — the model + continuous batcher.
- `OpenAIServingTranscription` / `OpenAIServingTranslation` — vLLM's stock
  handlers for audio preprocessing (resample to 16 kHz, chunk at 30 s),
  sampling, and OpenAI-shaped responses.
- A thin FastAPI layer (`main_stt.py`) that exposes the four endpoints
  Uttera expects — `/v1/audio/transcriptions`, `/v1/audio/translations`,
  `/v1/models`, `/health` — and carries the Redis self-registration
  protocol from the sibling repos.

**What is *not* here**:
- No hot/cold worker pool, no shared work queue, no subprocess spawning.
- No hand-rolled audio preprocessing. vLLM does it and we do not second-guess.

See [API.md](API.md) for endpoint details, [HISTORY.md](HISTORY.md) for
why we have two STT repos, and [ROADMAP.md](ROADMAP.md) for what is in
flight.

## Benchmarks (preview)

Empirical results on 1× RTX 5090, Whisper-large-v3-turbo via vLLM 0.19,
400 concurrent requests (20 min of audio each):

| Metric | Value |
|---|---|
| Throughput | **17.74 req/s** |
| RTF | **411× real-time** |
| Audio processed in total | 2h 34min |
| Wall-clock elapsed | 22.6 s |
| Failures | 0 |

A **58-minute YouTube video** transcribed in **11.6 seconds** end-to-end.

Bench harness in `tests/` pending port — see ROADMAP. The informal
numbers above were captured on an earlier test harness that will be
re-run and published in `tests/bench_400x.py`.

## Quickstart

```bash
git clone https://github.com/uttera/uttera-stt-vllm.git
cd uttera-stt-vllm
cp .env.example .env      # tweak WHISPER_MODEL, VLLM_* if needed
./setup.sh                # creates venv, installs vLLM, pre-downloads the model
source venv/bin/activate
uvicorn main_stt:app --host 0.0.0.0 --port 5000
```

Then:

```bash
curl -X POST http://localhost:5000/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "language=es"
```

## Configuration

All tuning is env var driven. See [.env.example](.env.example) for the
full surface. The most common overrides:

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `openai/whisper-large-v3-turbo` | Any HF model compatible with vLLM transcription. |
| `SERVED_MODEL_NAME` | `whisper-1` | Name advertised via `/v1/models`. |
| `VLLM_DTYPE` | `float16` | `float16` / `bfloat16` / `float32`. |
| `VLLM_GPU_MEM_UTIL` | `0.9` | Fraction of VRAM vLLM is allowed to claim. |
| `VLLM_MAX_NUM_SEQS` | `64` | Maximum in-flight sequences (batch size). |
| `VLLM_MAX_MODEL_LEN` | `448` | Whisper context cap (tokens). |
| `PORT` | `5000` | HTTP port. |
| `LIBRETRANSLATE_URL` | _(empty)_ | Base URL of a [LibreTranslate](https://libretranslate.com) instance. Required for `/v1/audio/translations` to work on a Whisper model without native translate (incl. turbo). |
| `LIBRETRANSLATE_API_KEY` | _(empty)_ | Optional, only if your LibreTranslate instance enforces a key. |
| `LIBRETRANSLATE_TIMEOUT_S` | `30` | Timeout for the translation call. |
| `REDIS_URL` | _(empty)_ | Optional; enables self-registration for a router. |

## Deployment

- **Docker**: `docker compose up -d` (GPU passthrough configured in
  `docker-compose.yml`).
- **systemd**: `uttera-stt-vllm.yml` is a ready-to-adapt unit file;
  install at `/etc/systemd/system/uttera-stt-vllm.service`.

## Hardware requirements

- GPU: NVIDIA with 24 GB+ VRAM recommended for Whisper-large-v3-turbo at
  high concurrency. Smaller Whisper variants run on lower-VRAM GPUs.
- Blackwell (RTX 5090) supported with CUDA 12.8.

## 🛡 License

**Server source code**: [Apache License 2.0](LICENSE). Commercial use permitted.

**Whisper model weights** (OpenAI): released under the MIT License —
commercial use permitted, no restrictions. See [NOTICE](NOTICE) for full
attributions.

Created and maintained by [Hugo L. Espuny](https://github.com/fakehec),
with contributions acknowledged in [AUTHORS.md](AUTHORS.md).

## ☕ Community

If you want to follow the project or get involved:

- ⭐ Star this repo to help discoverability.
- 🐛 Report issues via the [issue tracker](../../issues).
- 💬 Join the conversation in [Discussions](../../discussions).
- 📰 Technical posts at [blog.uttera.ai](https://blog.uttera.ai).
- 🌐 Uttera Cloud: [https://uttera.ai](https://uttera.ai) (EU-hosted,
  solar-powered, subscription flat-rate).

---

*Uttera /ˈʌt.ər.ə/ — from the English verb "to utter" (to speak aloud, to
pronounce, to give audible expression to). Formally, the name is a backronym
of **U**niversal **T**ext **T**ransformer **E**ngine for **R**ealtime **A**udio
— reflecting the project's origin as a STT/TTS server and its underlying
Transformer architecture.*

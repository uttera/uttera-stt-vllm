# uttera-stt-vllm

<p align="center">
  <img src="docs/img/banner.png" alt="uttera.ai — The voice layer for your AI" width="800">
</p>

High-throughput Speech-to-Text server built on **vLLM continuous batching**.
Whisper-large-v3-turbo today, room for future Transformer STT backends.

> **Status**: pre-alpha skeleton. Active development. See
> [ROADMAP.md](ROADMAP.md) for what's planned. First release expected
> Q2 2026.

## Positioning

| Use case | This repo | Sibling repo |
|---|---|---|
| Cloud, multi-tenant, large GPU (≥24 GB) | ✅ [uttera-stt-vllm](https://github.com/uttera/uttera-stt-vllm) | — |
| Home-lab, personal, small/mid GPU (8–16 GB) | — | [uttera-stt-hotcold](https://github.com/uttera/uttera-stt-hotcold) |

**Choose `uttera-stt-vllm` when**:
- You transcribe hours of audio per day across many concurrent streams.
- Continuous batching maximizes GPU utilization.
- You have large-VRAM GPUs dedicated to inference.

**Choose `uttera-stt-hotcold` when**:
- You have consumer GPUs (RTX 4070, 4080) and transcribe occasionally.
- Personal or single-user deployment.
- You want to share the GPU with other workloads.

## Architecture

Built on [vLLM](https://github.com/vllm-project/vllm) (native Whisper
support since v0.6.6). Handles audio chunking, continuous batching, and
OpenAI-compatible API out of the box.

- **Auto-chunking**: audio files longer than Whisper's 30 s context are
  split, transcribed in parallel batches, and stitched back together
  transparently. A one-hour podcast becomes ~120 chunks, all fused into
  one batch.
- **OpenAI-compatible**: drop-in for `openai.audio.transcriptions.create()`.
- **Long-audio support**: configurable `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`
  env var lets you transcribe audios of several hours in a single request.

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
See [benchmarks/](benchmarks/) for the methodology.

## Quickstart (coming soon)

```bash
git clone https://github.com/uttera/uttera-stt-vllm.git
cd uttera-stt-vllm
./setup.sh
VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=500 ./scripts/run.sh
```

## Hardware requirements

- GPU: NVIDIA with 24 GB+ VRAM recommended for Whisper-large-v3-turbo
  at high concurrency. Lower-VRAM GPUs can run smaller Whisper variants.
- Blackwell (RTX 5090) supported with CUDA 12.8. See
  [docs/blackwell.md](docs/blackwell.md) for the build procedure.

## License

[Apache License 2.0](LICENSE). Whisper model weights are released by OpenAI
under the MIT License — commercial use permitted without restriction.
See [NOTICE](NOTICE) for full attributions.

---

*Uttera /ˈʌt.ər.ə/ — from the English verb "to utter" (to speak aloud).
Also the backronym **U**niversal **T**ext **T**ransformer **E**ngine for
**R**ealtime **A**I.*

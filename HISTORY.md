# History

`uttera-stt-vllm` is the high-throughput sibling of [`uttera-stt-hotcold`](https://github.com/uttera/uttera-stt-hotcold). Both solve the same problem — a self-hosted, OpenAI-compatible speech-to-text API — but they make opposite assumptions about the runtime.

## Why two repos?

- **`uttera-stt-hotcold`** targets home-lab, personal, and small-to-mid GPU deployments (8–16 GB). It runs vanilla `openai-whisper` on a single GPU and squeezes concurrency out of a custom *hot worker + on-demand cold pool* architecture that spawns and reaps Python subprocesses dynamically. Great on a single 3090; works on a 4060.

- **`uttera-stt-vllm`** targets cloud, multi-tenant, and large-GPU deployments (≥24 GB). It delegates concurrency to **vLLM's continuous batching** — a single engine process serves dozens of parallel requests at near-optimal GPU utilisation by dynamically batching in-flight sequences on every decode step. The hot/cold pool disappears; the engine is the concurrency primitive.

Same API, same brand, different engine; pick whichever matches your hardware.

## Genesis

The original Whisper-STT hot/cold server was part of the [Stark Fleet internal toolbox](https://github.com/uttera/uttera-stt-hotcold/blob/master/HISTORY.md) and later open-sourced. During Q2 2026, a series of benchmarks on an RTX 5090 demonstrated that vLLM's continuous-batching path for Whisper-large-v3-turbo reached ~17.7 rps at 400 concurrent requests and transcribed a 58-minute YouTube clip in ~11.6 s end-to-end — numbers that the hot/cold architecture cannot match on the same hardware.

Rather than retrofit vLLM into the hot/cold repo (two backends, two test matrices, two ways to reason about latency), we split: `uttera-stt-hotcold` keeps its niche, `uttera-stt-vllm` owns the high-throughput story.

## Acknowledgments

- **OpenAI** for [Whisper](https://github.com/openai/whisper) and the reference API surface.
- **vLLM team** for [vllm-project/vllm](https://github.com/vllm-project/vllm), the continuous-batching engine this repo is built around, and specifically for the `speech_to_text` serving path that handles audio preprocessing, chunking, and streaming.
- **HuggingFace** for model hosting and the `huggingface_hub` client.
- Contributors to the sibling repos whose OSS scaffolding (LICENSE, NOTICE, community files, Docker/systemd templates) landed here verbatim.

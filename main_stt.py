#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Uttera STT vLLM Server (Single-Process, Continuous Batching)
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025-2026 Hugo L. Espuny
# Original work created with assistance from Google Gemini and Anthropic Claude
#
# Part of the Uttera voice stack (https://uttera.ai).
# See LICENSE and NOTICE for full terms and attributions.
#
# Package: uttera-stt-vllm
# Version: 1.5.0
# Maintainer: Uttera, Hugo L. Espuny
# Description: High-throughput Whisper STT server on vLLM continuous batching.
#              A single Python process hosts vLLM's AsyncLLM engine; concurrency
#              is handled by the engine's internal batching — no hot/cold pool,
#              no per-request worker spawning, no shared work queue.
#
# CHANGELOG:
# - 1.5.0 (2026-09-24): Robustness sweep, mirroring hardening applied to the
#   private Uttera deployment. Five additions, all env-gated and safe by
#   default:
#   1. Optional frozen-model mode. Set UTTERA_OFFLINE=1 and the server sets
#      HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / MODELSCOPE_OFFLINE before the ML
#      libraries import, so a validated model never silently re-downloads or
#      changes after a reboot. Online by default (a fresh install can fetch).
#   2. Voice-activity gate (VAD). Whisper hallucinates text on silence
#      (learned "Thank you."/"Gracias." from subtitle training data); the gate
#      returns an empty transcription when the whole clip has no speech, and on
#      any doubt transcribes normally. Silero JIT model, CPU, decides 30 s in
#      milliseconds. Env: VAD_ENABLED, VAD_THRESHOLD, VAD_JIT, VAD_STRIDE.
#      Requires the `silero-vad` package; degrades gracefully if absent.
#   3. Engine circuit breaker. N consecutive engine (5xx) failures flip the
#      server to not-ready so /health returns 503; a later success clears it.
#      Only 5xx count — 4xx (unreadable audio, unsupported language) are the
#      caller's fault and never open it. Env: ENGINE_FAIL_THRESHOLD.
#   4. Recovery self-probe. While the breaker is open, an in-process probe
#      (a synthetic tone, via httpx ASGITransport — no network) retries every
#      ENGINE_PROBE_SECONDS and auto-clears the breaker on success, so a
#      transient fault doesn't need a manual restart.
#   5. Correct HTTP status codes instead of a blanket 500: oversized upload →
#      413, text over the model's context → 413, GPU OOM → 503 (busy, not
#      broken; excluded from the breaker), malformed JSON → 400. Env:
#      MAX_FILESIZE_MB (default 250). Tracebacks are stripped from error
#      bodies. `consecutive_engine_failures` is now reported in /health.
#   Removed: the optional Redis self-registration (this server is now
#   standalone — one process, one model, an OpenAI-compatible API and a
#   /health); the /health `routing` block went with it.
#   Requires vLLM >= 0.24 (was 0.19.x): clears four published advisories in the
#   0.19 line (incl. a critical auth bypass) and follows the speech-to-text
#   handlers to their entrypoints.speech_to_text.* import path. See
#   requirements.txt for the sm_120/Blackwell FlashInfer note.
# - 1.4.0 (2026-04-21): Prometheus `/metrics` endpoint. Exposes
#   request counters (by endpoint/method/status), request duration
#   histograms, in-flight gauge, engine-ready gauge, STT-specific
#   counters (transcriptions by response_format, translations by
#   mode+format), audio-seconds processed counter, per-op inference
#   duration histograms (whisper_transcribe, libretranslate), error
#   counters typed by cause, and a build_info gauge with version +
#   engine + model labels. Scrape with Telegraf's inputs.prometheus
#   or any OpenMetrics consumer. Additive — existing endpoints
#   unchanged.
# - 1.3.0 (2026-04-18): Default port migrated from 5000 → 9005 in
#   lockstep with the sibling `uttera-stt-hotcold` v2.3.0, so both STT
#   backends expose the same default port and are drop-in swappable.
#   Rationale: port 5000 has known collisions with macOS AirPlay
#   Receiver (since Monterey) and with Docker Registry v2. The
#   9000-9099 range is IANA "User Ports" without canonical assignment.
#   Updated artefacts: `PORT` env default in main_stt.py, Dockerfile
#   EXPOSE and CMD, docker-compose port mapping and healthcheck,
#   .env.example, README + API.md, CI workflow. Migration: set
#   `PORT=5000` in your env if you need to preserve the old endpoint.
# - 1.2.0 (2026-04-18): OpenAI-compat polish sweep. Seven rough edges
#   uncovered by the full endpoint validation run against v1.1.0 are
#   now fixed. All backward-compatible; strict clients now get the
#   documented OpenAI contract instead of approximations:
#   1. `response_format=srt|vtt` returned HTTP 200 with an error body
#      `{"error":..., "code":400}`. Fixed: we now force vLLM to do
#      `verbose_json` internally and render the requested SRT/WebVTT
#      body with correct timecodes + content-type.
#   2. `response_format=text` returned JSON `{"text":..., "usage":...}`
#      with `application/json` Content-Type. Fixed: real `text/plain`
#      body with just the transcription text.
#   3. `temperature` outside [0.0, 1.0] was either 500 (negative) or
#      200 with complete gibberish (> 1.0). Fixed: validated at wrapper
#      → HTTP 422 with an explicit range message.
#   4. `language=xyzzy`, non-audio bodies, empty bodies all returned
#      HTTP 500 "Internal Server Error". Fixed: mapped vLLM's
#      ValueError("Invalid or unsupported audio file.") to HTTP 400
#      with a decode message, and Whisper language errors to HTTP 400
#      with the actual message.
#   5. `/v1/audio/translations` ignored `response_format` entirely —
#      always returned JSON `{"text":...}`. Fixed: the translation path
#      now translates each segment through LibreTranslate (in parallel)
#      so SRT/WebVTT translations preserve original timings.
#   6. `HEAD /health` returned HTTP 405. Fixed: the route now accepts
#      both GET and HEAD via `@app.api_route(methods=["GET", "HEAD"])`.
#   7. No CORS middleware. Added opt-in `CORSMiddleware` gated on the
#      `CORS_ALLOW_ORIGINS` env var (comma-separated list, or `"*"`).
#      Disabled by default — API-first deployments don't need it.
#   Also added: `X-Translation-Mode: libretranslate` response header on
#   the LibreTranslate-mediated translation path, matching the sibling
#   uttera-stt-hotcold v2.2.0 for observability symmetry.
# - 1.1.0 (2026-04-17): /v1/audio/translations now works with Whisper-turbo
#   (which lacks the native translate task) via a Whisper-transcribe →
#   LibreTranslate post-processing pipeline. Controlled by the new env var
#   LIBRETRANSLATE_URL (+ optional LIBRETRANSLATE_API_KEY,
#   LIBRETRANSLATE_TIMEOUT_S). When the URL is unset, the endpoint returns
#   HTTP 501 with a message telling the caller to either configure it or
#   switch to a model with native translate support (e.g. whisper-large-v3).
#   The pipeline also unlocks target languages other than English — Whisper
#   native translate only goes to English; LibreTranslate supports 49
#   cross-language pairs (es → fr, ru → ca, etc.). If source == target
#   (detected language matches `to_language`), LibreTranslate is skipped and
#   the raw transcription is returned.
# - 1.0.0 (2026-04-17): First stable release. Functionally complete and
#   benchmarked against uttera-stt-hotcold on LibriSpeech and an internal
#   Spanish corpus (see github.com/uttera/uttera-benchmarks). Added
#   GitHub Actions CI (lint + structure + optional GPU smoke). Pinned
#   vllm[audio] extra so resampy/av/soundfile are pulled in, without
#   which /v1/audio/transcriptions raises HTTP 500. Import paths
#   corrected to the actual vLLM 0.19 layout
#   (vllm.entrypoints.openai.speech_to_text.protocol and
#   vllm.entrypoints.openai.models.serving — not the names a research
#   agent originally cited). Dropped task="transcription" and
#   model_config= kwargs that vLLM 0.19 does not accept.
# - 0.1.0 (2026-04-16): Initial scaffold. FastAPI app that embeds vLLM's
#   AsyncLLM in-process with the stock OpenAIServingTranscription /
#   OpenAIServingTranslation handlers. OpenAI-compatible endpoints for
#   transcription and translation, custom /health and /v1/models aligned
#   with uttera-stt-hotcold house style. Pre-release — active development.
#
# --- Architecture Summary (v1.1.0) ---
#
# * SINGLE-PROCESS ENGINE
#   vllm.v1.engine.async_llm.AsyncLLM is instantiated once at startup
#   (lifespan) and kept resident for the lifetime of the server. It runs
#   in the same Python process as FastAPI — no subprocess, no HTTP
#   passthrough, no worker pool. Concurrency is handled entirely by
#   vLLM's continuous batching.
#
# * ENDPOINT HANDLERS
#   OpenAIServingTranscription and OpenAIServingTranslation from
#   vllm.entrypoints.openai.speech_to_text.serving do the audio
#   preprocessing (resample to 16 kHz, chunk at 30 s), prompt assembly,
#   sampling, and response shaping. We construct them once in the
#   lifespan and dispatch each request to their create_transcription /
#   create_translation coroutines.
#
# * WHAT IS *NOT* HERE (vs. uttera-stt-hotcold)
#   - cold_worker.py — vLLM has no worker subprocess concept.
#   - Work queue, hot/cold loops, pool manager, VRAM pre-checks,
#     COLD_POOL_SIZE / HOT_QUEUE_SAFETY_FACTOR — all removed.
#   - Cold-start EMA and pool sizing formulas.
#
# * STANDALONE
#   This server is self-contained: one process, one model, an OpenAI-
#   compatible HTTP API, and a /health that tells you if the engine is
#   alive. It has no external coordinator, service registry, or shared
#   datastore — run one, or run several behind any load balancer you like.
#

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated, Optional

# --- Optional frozen-model (offline) mode ------------------------------------
# ONLINE by default so a fresh install can fetch its model. Set UTTERA_OFFLINE=1
# to pin the engine to the local cache, so a model you have already validated
# can't silently re-download or change on a reboot. Applied BEFORE the ML
# libraries import (they read these variables at import time). You can also set
# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / MODELSCOPE_OFFLINE directly.
if os.environ.get("UTTERA_OFFLINE", "0").lower() in ("1", "true", "yes"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("MODELSCOPE_OFFLINE", "1")

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

# Load .env from the project directory or its parent
_base = os.path.dirname(os.path.abspath(__file__))
for _env_path in [os.path.join(_base, ".env"), os.path.join(os.path.dirname(_base), ".env")]:
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        break

# vLLM imports (after .env load so VLLM_* env vars are honoured).
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
# vLLM (>= 0.24) keeps the speech-to-text handlers under
# `entrypoints.speech_to_text.{transcription,translation}`. The class and
# method signatures are unchanged from the old 0.19 location
# (vllm.entrypoints.openai.speech_to_text.{protocol,serving}). Verified on
# 0.24 and 0.30.
from vllm.entrypoints.speech_to_text.transcription.protocol import (  # noqa: E402
    TranscriptionRequest,
)
from vllm.entrypoints.speech_to_text.translation.protocol import (  # noqa: E402
    TranslationRequest,
)
from vllm.entrypoints.openai.models.serving import (  # noqa: E402
    BaseModelPath,
    OpenAIServingModels,
)
from vllm.entrypoints.speech_to_text.transcription.serving import (  # noqa: E402
    OpenAIServingTranscription,
)
from vllm.entrypoints.speech_to_text.translation.serving import (  # noqa: E402
    OpenAIServingTranslation,
)
from vllm.usage.usage_lib import UsageContext  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402

# -------------------------------
# 1. Global Config & Logging
# -------------------------------

SERVER_VERSION = "1.5.0"

# Valid response formats per OpenAI spec. vLLM's own handler natively
# supports json/text/verbose_json but rejects srt/vtt; we always request
# verbose_json from vLLM internally and render the final response shape
# ourselves, so every documented format is honoured.
SUPPORTED_RESPONSE_FORMATS = {"json", "text", "srt", "vtt", "verbose_json"}

# Valid temperature range per OpenAI spec [0.0, 1.0]. Applied at the
# wrapper layer; vLLM itself accepts arbitrary temperatures and silently
# produces garbage for values outside this range.
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 1.0

DEBUG = os.environ.get("DEBUG", "false").lower() in ("1", "true", "yes")
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("uttera-stt-vllm")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODEL_CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.join(ASSETS_DIR, "models")),
    "huggingface",
)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)

# Model selection — default to Whisper-large-v3-turbo, override at will.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3-turbo")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "whisper-1")

# vLLM engine tuning. Defaults validated on RTX 5090 (32 GB) + Whisper-turbo.
VLLM_DTYPE = os.environ.get("VLLM_DTYPE", "float16")
VLLM_GPU_MEM_UTIL = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.9"))
VLLM_MAX_NUM_SEQS = int(os.environ.get("VLLM_MAX_NUM_SEQS", "64"))
VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "448"))
VLLM_ENFORCE_EAGER = os.environ.get("VLLM_ENFORCE_EAGER", "false").lower() in ("1", "true", "yes")

# LibreTranslate post-processing for /v1/audio/translations.
# When this URL is set, /v1/audio/translations first transcribes via the
# Whisper model (so turbo — which lacks the "translate" task — still works),
# then passes the text through LibreTranslate to reach the requested
# `to_language`. If LIBRETRANSLATE_URL is empty, /v1/audio/translations
# returns HTTP 501 with a message telling the caller to either configure it
# or switch to a model with native translate support (e.g. whisper-large-v3).
LIBRETRANSLATE_URL = os.environ.get("LIBRETRANSLATE_URL", "").rstrip("/")
LIBRETRANSLATE_API_KEY = os.environ.get("LIBRETRANSLATE_API_KEY", "")
LIBRETRANSLATE_TIMEOUT_S = float(os.environ.get("LIBRETRANSLATE_TIMEOUT_S", "30"))

# Maximum upload size. A body larger than this is rejected with HTTP 413
# (Payload Too Large) before it ever reaches the engine. 413 is a client
# error and does NOT count towards the engine circuit breaker, so a caller
# sending huge files cannot trip the breaker.
MAX_FILESIZE_MB = int(os.environ.get("MAX_FILESIZE_MB", "250"))

# Engine circuit breaker: N consecutive engine (5xx) failures mark the server
# as not-ready so /health returns 503 instead of routing work to a dead engine.
# A single later success clears it, so a transient fault doesn't wedge the
# server. Only 5xx count — 4xx are the caller's fault (unreadable audio,
# unsupported language) and must not open the breaker.
ENGINE_FAIL_THRESHOLD = int(os.environ.get("ENGINE_FAIL_THRESHOLD", "3"))

# Recovery self-probe: while the breaker is open, every ENGINE_PROBE_SECONDS
# the server sends a tiny in-process request to itself; a 200 clears the
# breaker automatically, without a manual restart. Costs nothing while the
# breaker is closed (a single boolean check per cycle).
ENGINE_PROBE_SECONDS = int(os.environ.get("ENGINE_PROBE_SECONDS", "30"))

# Voice-activity gate (VAD). Whisper hallucinates text on silence (it learned
# "Thank you." / "Gracias." after quiet stretches from its subtitle training
# data). The gate decides ONE thing: if the whole clip has no speech, the model
# is not called and an empty transcription is returned. It never trims audio,
# and on any doubt (no detector, decode failure, exception) it transcribes —
# the gate must never be the reason a good transcription is lost.
VAD_ENABLED = os.environ.get("VAD_ENABLED", "1").lower() not in ("0", "false", "no")
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
# Path to the Silero JIT model. Empty = look inside the installed package.
VAD_JIT = os.environ.get("VAD_JIT", "")
# Window stride. Silero's window is 512 samples = 32 ms. With stride 4 we look
# at 32 ms out of every 128, which does not miss speech (any real utterance
# lasts hundreds of ms and lands in several inspected windows); what it saves
# is the expensive case — proving 30 s of audio has NO speech means scanning
# all of it (with speech you exit on the first window).
VAD_STRIDE = max(1, int(os.environ.get("VAD_STRIDE", "4")))

# -------------------------------
# 2. Runtime State
# -------------------------------

_engine: Optional[AsyncLLM] = None
_transcription_handler: Optional[OpenAIServingTranscription] = None
_translation_handler: Optional[OpenAIServingTranslation] = None
_engine_ready: bool = False
_engine_error: Optional[str] = None

# Lightweight throughput telemetry (not used for routing, only reported).
_ema_rps: Optional[float] = None
_EMA_ALPHA_RPS = 0.2
_last_completion_ts: float = 0.0
_in_flight: int = 0
_total_completed: int = 0
_total_errors: int = 0

# Engine circuit breaker state.
_consecutive_engine_failures: int = 0
_engine_probe_task: Optional[asyncio.Task] = None

# VAD state. The Silero model carries internal state and is NOT thread-safe on
# a single instance, so every use is serialised under this lock.
_vad_model = None
_vad_broken: bool = False
_vad_lock = threading.Lock()


# -------------------------------
# 2b. Prometheus metrics
# -------------------------------
#
# Naming convention: `uttera_stt_<thing>`. Labels deliberately low-
# cardinality — no request_id, no detected-language (unbounded), no
# temperature. `endpoint` uses the fixed route list; anything off
# that list is clamped to "other" so a drive-by scanner hitting
# `/wp-admin` can't blow up label cardinality.

_HTTP_REQUESTS_TOTAL = Counter(
    "uttera_stt_requests_total",
    "HTTP requests by endpoint, method and status code",
    ["endpoint", "method", "status"],
)

_HTTP_REQUEST_DURATION = Histogram(
    "uttera_stt_request_duration_seconds",
    "HTTP request wall-clock duration in seconds",
    ["endpoint", "method"],
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

_INFLIGHT_GAUGE = Gauge(
    "uttera_stt_inflight_requests",
    "Requests currently being processed by the engine",
)

_ENGINE_READY_GAUGE = Gauge(
    "uttera_stt_engine_ready",
    "1 if the Whisper engine is loaded and ready, 0 otherwise",
)

_LIBRETRANSLATE_CONFIGURED_GAUGE = Gauge(
    "uttera_stt_libretranslate_configured",
    "1 if LIBRETRANSLATE_URL is set and translations go through LibreTranslate; 0 if only the legacy vLLM-native translate path is available",
)

_TRANSCRIPTIONS_TOTAL = Counter(
    "uttera_stt_transcriptions_total",
    "Transcription requests broken down by requested response_format",
    ["response_format"],            # json | text | verbose_json | srt | vtt
)

_TRANSLATIONS_TOTAL = Counter(
    "uttera_stt_translations_total",
    "Translation requests broken down by post-processing mode and response_format",
    ["mode", "response_format"],    # mode in {libretranslate, native}
)

_AUDIO_SECONDS_TOTAL = Counter(
    "uttera_stt_audio_seconds_total",
    "Total seconds of audio successfully processed (useful as a billing / throughput proxy)",
    ["endpoint"],                   # /v1/audio/transcriptions | /v1/audio/translations
)

_INFERENCE_DURATION = Histogram(
    "uttera_stt_inference_duration_seconds",
    "Per-call inference latency in seconds, by op",
    ["op"],                         # whisper_transcribe | libretranslate
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

_ERRORS_TOTAL = Counter(
    "uttera_stt_errors_total",
    "Errors by type",
    ["type"],                       # decode | validation | upstream | model | libretranslate
)

_BUILD_INFO = Gauge(
    "uttera_stt_build_info",
    "Build metadata (label values carry version, engine, and served model id)",
    ["version", "engine", "model"],
)

# Known HTTP routes — used to normalise the `endpoint` label so
# cardinality stays bounded even if someone probes unknown paths.
_KNOWN_ENDPOINTS = {
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/models",
    "/health",
    "/metrics",
}


# -------------------------------
# 3. Engine circuit breaker + recovery probe
# -------------------------------

def _engine_failure(http_status: int, exc: Optional[Exception] = None) -> None:
    """Count an engine failure. Open the breaker on reaching the threshold.

    Only 5xx count. Below the threshold nothing changes; once N consecutive
    engine failures are seen the server flips to not-ready and /health serves
    503 until a later success (or the recovery probe) clears it.
    """
    global _consecutive_engine_failures, _engine_ready, _engine_error
    if http_status < 500:
        return                      # caller's fault — the engine is fine
    _consecutive_engine_failures += 1
    if _consecutive_engine_failures >= ENGINE_FAIL_THRESHOLD and _engine_ready:
        _engine_ready = False
        _engine_error = ("circuit_breaker: %d consecutive engine failures; last: %s"
                         % (_consecutive_engine_failures,
                            f"{type(exc).__name__}: {exc}" if exc else "unknown"))[:300]
        log.error("CIRCUIT BREAKER OPEN: %s — /health now reports unavailable",
                  _engine_error)


def _engine_ok() -> None:
    """A success clears the breaker and resets the count."""
    global _consecutive_engine_failures, _engine_ready, _engine_error
    _consecutive_engine_failures = 0
    if not _engine_ready:
        _engine_ready = True
        _engine_error = None
        log.warning("CIRCUIT BREAKER CLOSED: the engine responds again")


def _probe_wav(seconds: float = 1.0, hz: int = 440, sr: int = 16000) -> bytes:
    """A mono 16 kHz WAV holding a tone. A tone, not silence: pure silence can
    make speech models fail, and a failing probe would keep the breaker open
    forever."""
    import io as _bio
    import math
    import struct
    import wave as _w
    n = int(seconds * sr)
    samples = b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * hz * i / sr)))
                       for i in range(n))
    buf = _bio.BytesIO()
    with _w.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(samples)
    return buf.getvalue()


async def _engine_probe_loop() -> None:
    """While the engine is not ready, send a tiny in-process request to the
    app every ENGINE_PROBE_SECONDS. A 200 makes the endpoint call _engine_ok(),
    which clears the breaker — no manual restart needed. Runs entirely in
    process via httpx's ASGITransport, so it needs no network or open port.
    """
    import httpx
    while True:
        try:
            await asyncio.sleep(ENGINE_PROBE_SECONDS)
            # Probe whenever the engine is not ready — whether the breaker
            # opened it or startup failed outright.
            if _engine_ready:
                continue
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://probe",
                                         timeout=120.0) as cli:
                r = await cli.post("/v1/audio/transcriptions",
                                   files={"file": ("probe.wav", _probe_wav(), "audio/wav")},
                                   data={"model": "whisper-1"})
            if r.status_code == 200:
                log.info("probe: the engine responds again")
            else:
                log.warning("probe: the engine is still down (HTTP %s)", r.status_code)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("probe: failed to probe: %s", e)


# -------------------------------
# 3b. Lifespan — engine + handlers
# -------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _engine, _transcription_handler, _translation_handler
    global _engine_ready, _engine_error, _engine_probe_task
    global _engine, _transcription_handler, _translation_handler
    global _engine_ready, _engine_error

    log.info(f"Starting Uttera STT vLLM v{SERVER_VERSION} — model={WHISPER_MODEL}")

    try:
        # vLLM infers the runner from the model architecture (Whisper triggers
        # the transcription path automatically); no `task` kwarg. Verified on
        # 0.24 and 0.30.
        engine_args = AsyncEngineArgs(
            model=WHISPER_MODEL,
            served_model_name=SERVED_MODEL_NAME,
            dtype=VLLM_DTYPE,
            gpu_memory_utilization=VLLM_GPU_MEM_UTIL,
            max_num_seqs=VLLM_MAX_NUM_SEQS,
            max_model_len=VLLM_MAX_MODEL_LEN,
            enforce_eager=VLLM_ENFORCE_EAGER,
            download_dir=MODEL_CACHE_DIR,
        )
        vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.OPENAI_API_SERVER,
        )

        _engine = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=UsageContext.OPENAI_API_SERVER,
            disable_log_stats=not DEBUG,
        )
        await _engine.reset_mm_cache()

        base_paths = [
            BaseModelPath(name=SERVED_MODEL_NAME, model_path=WHISPER_MODEL),
        ]
        serving_models = OpenAIServingModels(
            engine_client=_engine,
            base_model_paths=base_paths,
            lora_modules=[],
        )
        if hasattr(serving_models, "init_static_loras"):
            await serving_models.init_static_loras()

        _transcription_handler = OpenAIServingTranscription(
            _engine,
            serving_models,
            request_logger=None,
            return_tokens_as_token_ids=False,
            enable_force_include_usage=False,
        )
        _translation_handler = OpenAIServingTranslation(
            _engine,
            serving_models,
            request_logger=None,
            return_tokens_as_token_ids=False,
            enable_force_include_usage=False,
        )

        _engine_ready = True
        log.info(f"vLLM engine ready. VRAM free: {_vram_free_gb():.2f} GB")

    except Exception as e:
        _engine_error = str(e)
        log.exception("Engine init failed — server will serve /health 503")

    # Recovery self-probe: clears the breaker (or a failed startup) without a
    # manual restart. Cheap while the engine is ready (one boolean per cycle).
    _engine_probe_task = asyncio.create_task(_engine_probe_loop())

    yield

    # Shutdown
    log.info("Shutting down…")
    if _engine_probe_task:
        _engine_probe_task.cancel()
        try:
            await _engine_probe_task
        except (asyncio.CancelledError, Exception):
            pass
    if _engine is not None:
        try:
            _engine.shutdown()
        except Exception:
            pass


app = FastAPI(
    title="Uttera STT vLLM Server",
    version=SERVER_VERSION,
    lifespan=_lifespan,
)


# ── The right status code, not a blanket 500 ────────────────────────────────
# A 500 says "I broke". When the request is at fault, say so with a 4xx — and
# it's not cosmetic: 5xx failures count towards the engine circuit breaker, so
# a caller sending oversized text could otherwise trip the breaker for everyone.
#   · truncated JSON body            -> 400 (was 500)
#   · text over max_model_len        -> 413 (was 500)
#   · GPU OOM on a busy server       -> 503 (busy, not broken; also excluded
#                                            from the breaker, with Retry-After)
import re
from fastapi.responses import JSONResponse as _ErrResp

_SIGNS_TOO_LONG = ("max_model_len", "prompt_len", "context length", "too long",
                   "maximum context", "exceeds")
_SIGNS_OOM = ("out of memory", "outofmemoryerror", "cuda error: out of memory",
              "cublas_status_alloc_failed")


def _useful_line(exc) -> str:
    """Pull the line that explains the limit and DROP the traceback.

    A traceback in the response would leak the server's internal paths and
    package names, and is unreadable. We return just the line that mentions
    the limit."""
    for line in reversed(str(exc).splitlines()):
        low = line.lower()
        if any(s in low for s in _SIGNS_TOO_LONG) and 'file "' not in low:
            return re.sub(r"^[A-Za-z_]+Error:\s*", "", line.strip())[:300]
    return "the request exceeds the model's limit"


def _classify_error(exc):
    """Return (status, detail) by inspecting the error text, or (None, None)."""
    t = str(exc).lower()
    if any(s in t for s in _SIGNS_OOM):
        # Busy, not broken. A 503 doesn't count towards the breaker and tells
        # the caller to retry, which is the truth.
        return 503, "the server has no free memory right now; retry shortly"
    if any(s in t for s in _SIGNS_TOO_LONG):
        return 413, _useful_line(exc)
    return None, None


def _install_error_handlers(app):
    @app.exception_handler(json.JSONDecodeError)
    async def _on_json_error(request, exc):
        return _ErrResp(status_code=400,
                        content={"detail": "malformed JSON body: %s" % exc})

    async def _on_generic_error(request, exc):
        status, detail = _classify_error(exc)
        if status == 503:
            return _ErrResp(status_code=503, headers={"Retry-After": "30"},
                            content={"detail": detail})
        if status:
            return _ErrResp(status_code=status, content={"detail": detail})
        # What we can't classify stays a 500: we don't disguise a possible
        # server fault as a client error.
        return _ErrResp(status_code=500,
                        content={"detail": "%s: %s" % (type(exc).__name__, str(exc)[:300])})

    app.add_exception_handler(ValueError, _on_generic_error)
    app.add_exception_handler(RuntimeError, _on_generic_error)   # includes OutOfMemoryError


_install_error_handlers(app)
# ────────────────────────────────────────────────────────────────────────────

# Opt-in CORS middleware. API-first deployments don't need it, so CORS
# stays disabled by default. Set CORS_ALLOW_ORIGINS to a comma-separated
# list of origins or "*" to enable it.
_cors_origins_env = os.environ.get("CORS_ALLOW_ORIGINS", "").strip()
if _cors_origins_env:
    _cors_origins = ["*"] if _cors_origins_env == "*" else [
        o.strip() for o in _cors_origins_env.split(",") if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Translation-Mode"],
    )


# Prometheus middleware — tracks every HTTP request generically.
# Endpoint-specific labels (response_format, translation mode, audio
# seconds) are attached inside the endpoint handlers for richer
# breakdowns.

class _PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        method = request.method
        # Don't self-meter — /metrics would otherwise tick every scrape.
        if path == "/metrics":
            return await call_next(request)
        endpoint = path if path in _KNOWN_ENDPOINTS else "other"
        t0 = time.monotonic()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed = time.monotonic() - t0
            _HTTP_REQUESTS_TOTAL.labels(
                endpoint=endpoint, method=method, status=str(status)
            ).inc()
            _HTTP_REQUEST_DURATION.labels(
                endpoint=endpoint, method=method
            ).observe(elapsed)

app.add_middleware(_PrometheusMiddleware)

# Build_info is a static gauge — set once at module import time.
_BUILD_INFO.labels(
    version=SERVER_VERSION,
    engine="vllm",
    model=os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3-turbo"),
).set(1)
# LibreTranslate gauge — set once from env config. The server's
# behaviour doesn't change at runtime based on this; it's exposed
# here so dashboards can distinguish the two translation paths.
_LIBRETRANSLATE_CONFIGURED_GAUGE.set(
    1 if os.environ.get("LIBRETRANSLATE_URL", "").strip() else 0
)


# -------------------------------
# 4. Helpers
# -------------------------------

def _vram_free_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free_bytes, _total = torch.cuda.mem_get_info()
    return free_bytes / (1024 ** 3)


def _update_rps() -> None:
    global _ema_rps, _last_completion_ts, _total_completed
    now = time.time()
    _total_completed += 1
    if _last_completion_ts > 0:
        dt = now - _last_completion_ts
        if dt > 0:
            inst_rps = 1.0 / dt
            if _ema_rps is None:
                _ema_rps = inst_rps
            else:
                _ema_rps = _EMA_ALPHA_RPS * inst_rps + (1.0 - _EMA_ALPHA_RPS) * _ema_rps
    _last_completion_ts = now


# Whisper emits ISO-639-1 codes (e.g. "zh") but LibreTranslate expects
# different codes for a few Asian languages. Translate them at the boundary.
_WHISPER_TO_LIBRETRANSLATE_LANG = {
    "zh": "zh-Hans",
    "zh-cn": "zh-Hans",
    "zh-tw": "zh-Hant",
}


def _normalise_lang_for_libretranslate(code: str) -> str:
    if not code:
        return code
    code = code.lower()
    return _WHISPER_TO_LIBRETRANSLATE_LANG.get(code, code)


async def _libretranslate(text: str, source: str, target: str) -> str:
    """Call LibreTranslate. Raises on network or HTTP errors; caller maps to 502/501."""
    import httpx  # already a transitive dep; import lazily so tests without it still start
    src = _normalise_lang_for_libretranslate(source) or "auto"
    tgt = _normalise_lang_for_libretranslate(target)
    payload: dict = {"q": text, "source": src, "target": tgt, "format": "text"}
    if LIBRETRANSLATE_API_KEY:
        payload["api_key"] = LIBRETRANSLATE_API_KEY
    async with httpx.AsyncClient(timeout=LIBRETRANSLATE_TIMEOUT_S) as client:
        r = await client.post(f"{LIBRETRANSLATE_URL}/translate", json=payload)
        r.raise_for_status()
        data = r.json()
    out = data.get("translatedText")
    if not isinstance(out, str):
        raise RuntimeError(f"Unexpected LibreTranslate response: {data}")
    return out


def _validate_temperature(temperature: float) -> None:
    """Validate `temperature` is in the OpenAI spec range [0.0, 1.0].

    vLLM accepts any float and silently produces gibberish for out-of-range
    values; the wrapper enforces the contract and returns HTTP 422 early.
    """
    if not (TEMPERATURE_MIN <= temperature <= TEMPERATURE_MAX):
        raise HTTPException(
            status_code=422,
            detail=(
                f"temperature {temperature} out of range. "
                f"Must be in [{TEMPERATURE_MIN}, {TEMPERATURE_MAX}]."
            ),
        )


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SubRip (SRT)."""
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for WebVTT."""
    return _format_timestamp_srt(seconds).replace(",", ".")


def _segments_to_srt(segments: list) -> str:
    """Render whisper-style segments [{start, end, text, ...}] as SRT."""
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        start = _format_timestamp_srt(float(seg.get("start", 0.0)))
        end = _format_timestamp_srt(float(seg.get("end", 0.0)))
        text = (seg.get("text") or "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def _segments_to_vtt(segments: list) -> str:
    """Render whisper-style segments as WebVTT."""
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp_vtt(float(seg.get("start", 0.0)))
        end = _format_timestamp_vtt(float(seg.get("end", 0.0)))
        text = (seg.get("text") or "").strip()
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _result_to_dict(result) -> dict:
    """Convert vLLM's TranscriptionResponse/Verbose into a plain dict."""
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    return {"text": str(result)}


def _unwrap_vllm_error(result) -> Optional[tuple[int, str]]:
    """If `result` is a vLLM ErrorResponse, return `(http_code, message)`.

    vLLM's ErrorResponse has shape `{"error": {"message":..., "code":N, ...}}`;
    our original wrapper checked for a top-level `.code` which never existed,
    so every error surfaced as HTTP 200 with an error body. Now we look at
    `.error.code` (or the equivalent dict key) and surface the real code.
    """
    err = getattr(result, "error", None)
    if err is None and isinstance(result, dict):
        err = result.get("error")
    if err is None:
        return None
    code = getattr(err, "code", None) if not isinstance(err, dict) else err.get("code")
    message = getattr(err, "message", None) if not isinstance(err, dict) else err.get("message")
    return (int(code or 500), str(message or "Unknown error"))


def _render_response(
    result_dict: dict,
    response_format: str,
    extra_headers: Optional[dict] = None,
) -> Response:
    """Render a whisper-shape result dict in the requested OpenAI format.

    - `json`: compact `{"text": "..."}` (OpenAI spec — no segments).
    - `text`: plain text body with `text/plain` Content-Type.
    - `verbose_json`: full result (text + segments + language + usage).
    - `srt`: SubRip subtitle file (Content-Type: application/x-subrip).
    - `vtt`: WebVTT subtitle file (Content-Type: text/vtt).
    """
    headers = dict(extra_headers or {})
    text = result_dict.get("text") or ""
    segments = result_dict.get("segments") or []

    if response_format == "text":
        return PlainTextResponse(content=text, headers=headers)
    if response_format == "srt":
        return PlainTextResponse(
            content=_segments_to_srt(segments),
            media_type="application/x-subrip",
            headers=headers,
        )
    if response_format == "vtt":
        return PlainTextResponse(
            content=_segments_to_vtt(segments),
            media_type="text/vtt",
            headers=headers,
        )
    if response_format == "verbose_json":
        # The full whisper result — keep whatever extra fields (usage,
        # words, duration) vLLM populated.
        return JSONResponse(content=result_dict, headers=headers)
    # Default: OpenAI-compact json — only {"text": ...}.
    return JSONResponse(content={"text": text}, headers=headers)


def _map_engine_exception(exc: Exception) -> HTTPException:
    """Translate vLLM/Whisper internal exceptions into meaningful HTTP errors.

    vLLM raises ValueError("Invalid or unsupported audio file.") when the
    audio preprocessor can't decode the body, and ValueError for unsupported
    language codes. Both are caller errors, not server faults, so surface
    them as HTTP 400 with an actionable message.
    """
    msg = str(exc)
    if isinstance(exc, ValueError):
        low = msg.lower()
        if "invalid or unsupported audio" in low or "failed to load audio" in low:
            return HTTPException(
                status_code=400,
                detail="Failed to decode audio body — not a valid audio stream or unsupported codec.",
            )
        if "unsupported language" in low or ("language" in low and "not supported" in low):
            return HTTPException(status_code=400, detail=msg)
        # Generic ValueError from the engine — safer to surface as 400 with
        # the actual message than as an opaque 500.
        return HTTPException(status_code=400, detail=msg)
    return HTTPException(
        status_code=500,
        detail="Transcription failed. Check server logs.",
    )


# -------------------------------
# 4b. Voice-activity gate (VAD)
# -------------------------------

def _vad_loaded():
    """The Silero JIT model, loaded BY HAND without importing `silero_vad`.

    Two reasons:
      1. `import silero_vad` drags `onnxruntime` into the process even when the
         torch model is the one used. For a gate the per-window probability is
         enough — no package, no segmentation needed.
      2. The model carries state and is NOT safe to call from several threads
         on one instance (reproduced: concurrent requests -> `free(): invalid
         pointer` and a process crash). Hence the lock around every use.
    """
    global _vad_model, _vad_broken
    if _vad_model is not None or _vad_broken:
        return _vad_model
    with _vad_lock:
        if _vad_model is None and not _vad_broken:
            try:
                path = VAD_JIT
                if not path:
                    import importlib.util as _u
                    path = os.path.join(
                        os.path.dirname(_u.find_spec("silero_vad").origin),
                        "data", "silero_vad.jit")
                _vad_model = torch.jit.load(path)
                _vad_model.eval()
                print("VAD: model loaded from %s (threshold %.2f)" % (path, VAD_THRESHOLD),
                      file=sys.stderr, flush=True)
            except Exception as e:
                _vad_broken = True
                print("VAD: unavailable (%s); transcribing everything, as before" % e,
                      file=sys.stderr, flush=True)
    return _vad_model


def _to_16k(audio_bytes: bytes):
    """The audio as float32 mono at 16 kHz via ffmpeg — the same path the model
    uses to read it, not a second route that could disagree."""
    import numpy as np
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", "pipe:0",
           "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "pipe:1"]
    p = subprocess.run(cmd, input=audio_bytes, capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.int16).flatten().astype("float32") / 32768.0


def _no_voice(audio_bytes: bytes) -> bool:
    """True ONLY if the detector asserts there is no speech anywhere in the clip.

    On any doubt it returns False and the audio is transcribed: the gate must
    never be the reason a good transcription is lost.
    """
    if not VAD_ENABLED or _vad_loaded() is None:
        return False
    try:
        pcm = _to_16k(audio_bytes)
        if pcm.size < 512:
            return False
        m = _vad_model
        with _vad_lock:            # a stateful instance: one thread at a time
            m.reset_states()
            with torch.no_grad():
                for i in range(0, len(pcm) - 512, 512 * VAD_STRIDE):
                    chunk = torch.from_numpy(pcm[i:i + 512]).unsqueeze(0)
                    if float(m(chunk, 16000).item()) >= VAD_THRESHOLD:
                        return False          # speech found
        return True
    except Exception as e:
        print("VAD: could not decide (%s); transcribing" % e,
              file=sys.stderr, flush=True)
        return False


def _empty_result() -> dict:
    return {"text": "", "segments": [], "language": ""}


# -------------------------------
# 5. Endpoints
# -------------------------------

@app.get("/metrics")
async def metrics():
    """Prometheus-format scrape endpoint.

    Scrape with Telegraf's `inputs.prometheus` plugin, Prometheus
    itself, or any OpenMetrics-compatible consumer. Cardinality is
    bounded by design (fixed endpoint list, no per-request-id labels).
    """
    # Reflect current liveness state into the gauges on every scrape
    # so they're always accurate without hooking every state transition.
    _ENGINE_READY_GAUGE.set(1 if _engine_ready else 0)
    _INFLIGHT_GAUGE.set(_in_flight)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/audio/transcriptions")
async def create_transcriptions(
    raw_request: Request,
    request: Annotated[TranscriptionRequest, Form()],
):
    global _in_flight, _total_errors
    if _transcription_handler is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    # Wrapper-level validation that vLLM skips.
    _validate_temperature(request.temperature)

    # User's requested format (pydantic Literal has already rejected
    # unknown values with HTTP 422 before we got here).
    user_format = (request.response_format or "json")
    _TRANSCRIPTIONS_TOTAL.labels(response_format=user_format).inc()

    # Streaming stays untouched — vLLM emits SSE. When `stream=True` is
    # used we keep the original behaviour (no custom format rendering).
    if getattr(request, "stream", False):
        audio_data = await request.file.read()
        _in_flight += 1
        _INFLIGHT_GAUGE.inc()
        try:
            with _INFERENCE_DURATION.labels(op="whisper_transcribe").time():
                result = await _transcription_handler.create_transcription(
                    audio_data, request, raw_request,
                )
        except Exception as exc:
            _total_errors += 1
            _ERRORS_TOTAL.labels(type="model").inc()
            _http_err = _map_engine_exception(exc)
            _engine_failure(_http_err.status_code, exc)
            raise _http_err from exc
        finally:
            _in_flight -= 1
            _INFLIGHT_GAUGE.dec()
            _update_rps()
        if hasattr(result, "__aiter__"):
            return StreamingResponse(result, media_type="text/event-stream")
        err = _unwrap_vllm_error(result)
        if err is not None:
            _ERRORS_TOTAL.labels(type="upstream").inc()
            raise HTTPException(status_code=err[0], detail=err[1])
        return JSONResponse(content=_result_to_dict(result))

    # Non-streaming path: force vLLM to do `verbose_json` so we always
    # have `segments` available for SRT/VTT rendering and can control
    # the wire format ourselves.
    internal_request = request.model_copy(update={"response_format": "verbose_json"})

    audio_data = await request.file.read()
    # 413, not 400: the file is valid, it just doesn't fit. And a 4xx doesn't
    # count towards the circuit breaker, so a caller with huge files can't take
    # the server down.
    _mb = len(audio_data) / 1048576
    if _mb > MAX_FILESIZE_MB:
        raise HTTPException(status_code=413,
                            detail=f"File is {_mb:.0f} MB; the limit is {MAX_FILESIZE_MB} MB.")
    # The gate goes here, BEFORE occupying the engine: with no speech there is
    # nothing to transcribe and we save the GPU. Non-streaming path only.
    if await asyncio.to_thread(_no_voice, audio_data):
        print("VAD: no speech; the model is not called", file=sys.stderr, flush=True)
        _engine_ok()
        return _render_response(_empty_result(), user_format)
    _in_flight += 1
    _INFLIGHT_GAUGE.inc()
    try:
        with _INFERENCE_DURATION.labels(op="whisper_transcribe").time():
            result = await _transcription_handler.create_transcription(
                audio_data, internal_request, raw_request,
            )
    except Exception as exc:
        _total_errors += 1
        _ERRORS_TOTAL.labels(type="model").inc()
        _http_err = _map_engine_exception(exc)
        _engine_failure(_http_err.status_code, exc)
        raise _http_err from exc
    finally:
        _in_flight -= 1
        _INFLIGHT_GAUGE.dec()
        _update_rps()

    # Was it a vLLM-side error? vLLM returns an ErrorResponse model for
    # things like invalid model names — surface its real code/message.
    err = _unwrap_vllm_error(result)
    if err is not None:
        _ERRORS_TOTAL.labels(type="upstream").inc()
        raise HTTPException(status_code=err[0], detail=err[1])

    result_dict = _result_to_dict(result)
    # Tap audio duration for the billing/throughput counter.
    _dur = result_dict.get("duration")
    if _dur is None:
        segs = result_dict.get("segments") or []
        if segs and isinstance(segs[-1], dict):
            _dur = segs[-1].get("end")
    if isinstance(_dur, (int, float)) and _dur > 0:
        _AUDIO_SECONDS_TOTAL.labels(endpoint="/v1/audio/transcriptions").inc(float(_dur))
    _engine_ok()
    return _render_response(result_dict, user_format)


@app.post("/v1/audio/translations")
async def create_translations(
    raw_request: Request,
    request: Annotated[TranslationRequest, Form()],
):
    """
    Pipeline: Whisper transcribes in the source language, LibreTranslate
    translates the text to `to_language` (default "en"). This works for any
    Whisper model — including turbo, which lacks the native `translate`
    task — and supports target languages beyond English, which plain
    Whisper translate does not.

    Requires LIBRETRANSLATE_URL; without it the endpoint returns HTTP 501.
    """
    global _in_flight, _total_errors
    if _transcription_handler is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    if not LIBRETRANSLATE_URL:
        raise HTTPException(
            status_code=501,
            detail=(
                "Translation is disabled: LIBRETRANSLATE_URL is not set. "
                "Either configure a LibreTranslate endpoint or run this "
                "server with a Whisper model that supports the native "
                "`translate` task (e.g. openai/whisper-large-v3)."
            ),
        )

    _validate_temperature(request.temperature)

    user_format = (request.response_format or "json")
    target_lang = (request.to_language or "en").lower()
    _TRANSLATIONS_TOTAL.labels(mode="libretranslate", response_format=user_format).inc()

    # Build a TranscriptionRequest that mirrors the TranslationRequest but
    # forces verbose_json so we receive the detected source language AND
    # the per-segment timings (needed to render SRT/VTT translations).
    transcription_req = TranscriptionRequest(
        file=request.file,
        model=request.model,
        prompt=request.prompt,
        response_format="verbose_json",
        temperature=request.temperature,
        language=request.language,  # None means Whisper auto-detect
    )

    audio_data = await request.file.read()
    _mb = len(audio_data) / 1048576
    if _mb > MAX_FILESIZE_MB:
        raise HTTPException(status_code=413,
                            detail=f"File is {_mb:.0f} MB; the limit is {MAX_FILESIZE_MB} MB.")
    _in_flight += 1
    _INFLIGHT_GAUGE.inc()
    try:
        with _INFERENCE_DURATION.labels(op="whisper_transcribe").time():
            transcribed = await _transcription_handler.create_transcription(
                audio_data, transcription_req, raw_request,
            )
    except Exception as exc:
        _total_errors += 1
        _ERRORS_TOTAL.labels(type="model").inc()
        _http_err = _map_engine_exception(exc)
        _engine_failure(_http_err.status_code, exc)
        raise _http_err from exc
    finally:
        _in_flight -= 1
        _INFLIGHT_GAUGE.dec()
        _update_rps()

    # The transcription succeeded — clear the breaker regardless of what the
    # LibreTranslate step does next (that's a separate backend).
    _engine_ok()

    err = _unwrap_vllm_error(transcribed)
    if err is not None:
        _ERRORS_TOTAL.labels(type="upstream").inc()
        raise HTTPException(status_code=err[0], detail=err[1])
    if hasattr(transcribed, "__aiter__"):
        _ERRORS_TOTAL.labels(type="model").inc()
        raise HTTPException(
            status_code=500,
            detail="Transcription handler returned a stream unexpectedly.",
        )

    result_dict = _result_to_dict(transcribed)
    # Tap audio duration (billing/throughput counter). Count against the
    # translations endpoint since this is what the caller hit.
    _dur = result_dict.get("duration")
    if _dur is None:
        _segs_probe = result_dict.get("segments") or []
        if _segs_probe and isinstance(_segs_probe[-1], dict):
            _dur = _segs_probe[-1].get("end")
    if isinstance(_dur, (int, float)) and _dur > 0:
        _AUDIO_SECONDS_TOTAL.labels(endpoint="/v1/audio/translations").inc(float(_dur))
    text = (result_dict.get("text") or "")
    segments = result_dict.get("segments") or []
    source_lang = (result_dict.get("language") or "").lower()

    log.info(
        f"[translate] source={source_lang!r} target={target_lang!r} "
        f"n_segments={len(segments)} text_preview={text[:80]!r}"
    )

    extra_headers = {"X-Translation-Mode": "libretranslate"}

    # No translation needed — empty audio or source already matches target.
    if not text.strip() or (source_lang and source_lang == target_lang):
        if source_lang and source_lang == target_lang:
            log.info("[translate] source==target, skipping LibreTranslate")
        return _render_response(result_dict, user_format, extra_headers=extra_headers)

    # Translate the full text and each segment in parallel. We translate
    # segments individually so SRT/VTT subtitles keep their original
    # timings aligned to the correct translated text. For compact JSON /
    # plain text responses we only need `translated_text`, but doing both
    # in one gather keeps the total latency ~ 1× a single LibreTranslate call.
    lt_source = source_lang or "auto"
    try:
        tasks: list = [
            _libretranslate(text, lt_source, target_lang),
        ]
        if user_format in ("srt", "vtt", "verbose_json"):
            tasks.extend(
                _libretranslate((seg.get("text") or "").strip(), lt_source, target_lang)
                for seg in segments
            )
        with _INFERENCE_DURATION.labels(op="libretranslate").time():
            translations = await asyncio.gather(*tasks)
    except Exception as e:
        _ERRORS_TOTAL.labels(type="libretranslate").inc()
        log.warning(f"LibreTranslate call failed: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Translation backend failure: {type(e).__name__}: {e}",
        )

    translated_text = translations[0]
    if len(translations) > 1:
        # Zip the translated per-segment text back into the segments list
        # so the downstream SRT/VTT/verbose_json renderer sees timed
        # subtitle cues in the target language.
        translated_segments: list = []
        for seg, seg_text in zip(segments, translations[1:]):
            new_seg = dict(seg)
            new_seg["text"] = seg_text
            translated_segments.append(new_seg)
        result_dict = dict(result_dict)
        result_dict["text"] = translated_text
        result_dict["segments"] = translated_segments
        # Record the new language in verbose_json too.
        result_dict["language"] = target_lang
    else:
        result_dict = {"text": translated_text}

    return _render_response(result_dict, user_format, extra_headers=extra_headers)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": SERVED_MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "uttera",
        }],
    }


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    body = {
        "status": "ok" if _engine_ready else "starting",
        "version": SERVER_VERSION,
        "engine": "vllm",
        "model": WHISPER_MODEL,
        "served_as": SERVED_MODEL_NAME,
        "engine_ready": _engine_ready,
        "engine_error": _engine_error,
        "metrics": {
            "in_flight": _in_flight,
            "total_completed": _total_completed,
            "total_errors": _total_errors,
            "consecutive_engine_failures": _consecutive_engine_failures,
            "ema_rps": _ema_rps,
            "vram_free_gb": round(_vram_free_gb(), 2),
            "max_num_seqs": VLLM_MAX_NUM_SEQS,
            "max_model_len": VLLM_MAX_MODEL_LEN,
            "gpu_memory_utilization": VLLM_GPU_MEM_UTIL,
        },
    }
    status_code = 200 if _engine_ready else 503
    return JSONResponse(status_code=status_code, content=body)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "9005"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("main_stt:app", host=host, port=port, log_level="debug" if DEBUG else "info")

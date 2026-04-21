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
# Version: 1.4.0
# Maintainer: J.A.R.V.I.S. A.I., Hugo L. Espuny
# Description: High-throughput Whisper STT server on vLLM continuous batching.
#              A single Python process hosts vLLM's AsyncLLM engine; concurrency
#              is handled by the engine's internal batching — no hot/cold pool,
#              no per-request worker spawning, no shared work queue.
#
# CHANGELOG:
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
#   lockstep with the sibling `uttera-stt-hotcold` v2.3.0. The Uttera
#   stack now uses a canonical port scheme keyed by service family
#   (STT=9005, TTS=9004) — the Gatekeeper routes to a single port per
#   family regardless of which backend (hotcold / vllm) is behind it.
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
#   with uttera-stt-hotcold house style. Redis self-registration carried
#   over from the sibling repo. Pre-release — active development.
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
# * OPTIONAL REDIS SELF-REGISTRATION
#   If REDIS_URL is set, the server publishes {load_score,
#   accepts_requests, host, port, version, ts} to stt:nodes:{NODE_ID}
#   with a short TTL on a background tick, matching the sibling repos
#   so a front-end router can discover all Uttera nodes uniformly.
#
# * WHAT IS *NOT* HERE (vs. uttera-stt-hotcold)
#   - cold_worker.py — vLLM has no worker subprocess concept.
#   - Work queue, hot/cold loops, pool manager, VRAM pre-checks,
#     COLD_POOL_SIZE / HOT_QUEUE_SAFETY_FACTOR — all removed.
#   - Cold-start EMA and pool sizing formulas.
#

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated, Optional

import redis.asyncio as aioredis
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
from vllm.entrypoints.openai.speech_to_text.protocol import (  # noqa: E402
    TranscriptionRequest,
    TranslationRequest,
)
from vllm.entrypoints.openai.models.serving import (  # noqa: E402
    BaseModelPath,
    OpenAIServingModels,
)
from vllm.entrypoints.openai.speech_to_text.serving import (  # noqa: E402
    OpenAIServingTranscription,
    OpenAIServingTranslation,
)
from vllm.usage.usage_lib import UsageContext  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402

# -------------------------------
# 1. Global Config & Logging
# -------------------------------

SERVER_VERSION = "1.4.0"

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

# Drain time (seconds) considered 100% load for routing score. Matches the
# hotcold sibling so the upstream router can use a single threshold.
ROUTING_DRAIN_CAP_SECONDS = float(os.environ.get("ROUTING_DRAIN_CAP_SECONDS", "120"))

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

# Redis self-registration (opt-in). If REDIS_URL is unset, publishing is skipped.
REDIS_URL = os.environ.get("REDIS_URL", "")
REDIS_NODE_HOST = os.environ.get("NODE_HOST", "localhost")
REDIS_NODE_PORT = int(os.environ.get("NODE_PORT", "9005"))
REDIS_NODE_ID = os.environ.get("NODE_ID", "") or f"{REDIS_NODE_HOST}:{REDIS_NODE_PORT}"
REDIS_KEY = f"stt:nodes:{REDIS_NODE_ID}"
REDIS_PUBLISH_INTERVAL = float(os.environ.get("REDIS_PUBLISH_INTERVAL", "0.5"))
REDIS_TTL = max(2, int(REDIS_PUBLISH_INTERVAL * 3 + 1))

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

_redis: Optional[aioredis.Redis] = None
_redis_task: Optional[asyncio.Task] = None


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
# 3. Lifespan — engine + handlers + Redis
# -------------------------------

async def _publish_to_redis_loop() -> None:
    """Background task: publish routing state to Redis every interval.

    No-op if Redis is unreachable; failures never affect request serving.
    """
    global _redis
    while True:
        try:
            await asyncio.sleep(REDIS_PUBLISH_INTERVAL)
            if _redis is None:
                continue
            load = min(1.0, _in_flight / max(1, VLLM_MAX_NUM_SEQS))
            accepts = bool(_engine_ready) and load < 1.0
            payload = json.dumps({
                "load_score":       load,
                "accepts_requests": accepts,
                "host":              REDIS_NODE_HOST,
                "port":              REDIS_NODE_PORT,
                "version":           SERVER_VERSION,
                "engine":            "vllm",
                "model":             SERVED_MODEL_NAME,
                "ts":                time.time(),
            })
            try:
                await _redis.set(REDIS_KEY, payload, ex=REDIS_TTL)
            except Exception as e:
                log.debug(f"Redis publish failed (non-fatal): {e}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning(f"Redis publish loop error: {e}")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _engine, _transcription_handler, _translation_handler
    global _engine_ready, _engine_error, _redis, _redis_task

    log.info(f"Starting Uttera STT vLLM v{SERVER_VERSION} — model={WHISPER_MODEL}")

    try:
        # vLLM 0.19 infers the runner from the model architecture (Whisper
        # triggers the transcription path automatically); no `task` kwarg.
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

    # Redis self-registration (optional).
    if REDIS_URL:
        try:
            _redis = aioredis.from_url(REDIS_URL, decode_responses=False)
            await _redis.ping()
            _redis_task = asyncio.create_task(_publish_to_redis_loop())
            log.info(f"Redis registered at {REDIS_KEY}")
        except Exception as e:
            log.warning(f"Redis unavailable, skipping self-registration: {e}")
            _redis = None

    yield

    # Shutdown
    log.info("Shutting down…")
    if _redis_task:
        _redis_task.cancel()
        try:
            await _redis_task
        except (asyncio.CancelledError, Exception):
            pass
    if _redis:
        try:
            await _redis.delete(REDIS_KEY)
        except Exception:
            pass
        try:
            await _redis.aclose()
        except Exception:
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


def _routing_state() -> dict:
    """Current load/accepts snapshot. Shared by /health and Redis publisher."""
    load = min(1.0, _in_flight / max(1, VLLM_MAX_NUM_SEQS))
    accepts = bool(_engine_ready) and load < 1.0
    return {"load_score": load, "accepts_requests": accepts}


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
            raise _map_engine_exception(exc) from exc
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
        raise _map_engine_exception(exc) from exc
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
        raise _map_engine_exception(exc) from exc
    finally:
        _in_flight -= 1
        _INFLIGHT_GAUGE.dec()
        _update_rps()

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
    # in one gather keeps the total latency ~ 1× LibreTranslate call
    # (sphinx:5200 handles the parallelism trivially).
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
    state = _routing_state()
    body = {
        "status": "ok" if _engine_ready else "starting",
        "version": SERVER_VERSION,
        "engine": "vllm",
        "model": WHISPER_MODEL,
        "served_as": SERVED_MODEL_NAME,
        "engine_ready": _engine_ready,
        "engine_error": _engine_error,
        "routing": state,
        "metrics": {
            "in_flight": _in_flight,
            "total_completed": _total_completed,
            "total_errors": _total_errors,
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

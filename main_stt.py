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
# Version: 1.1.0
# Maintainer: J.A.R.V.I.S. A.I., Hugo L. Espuny
# Description: High-throughput Whisper STT server on vLLM continuous batching.
#              A single Python process hosts vLLM's AsyncLLM engine; concurrency
#              is handled by the engine's internal batching — no hot/cold pool,
#              no per-request worker spawning, no shared work queue.
#
# CHANGELOG:
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
from fastapi.responses import JSONResponse, StreamingResponse

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

SERVER_VERSION = "1.1.0"

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
REDIS_NODE_PORT = int(os.environ.get("NODE_PORT", "5000"))
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
    src = _normalise_lang_for_libretranslate(source)
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


# -------------------------------
# 5. Endpoints
# -------------------------------

@app.post("/v1/audio/transcriptions")
async def create_transcriptions(
    raw_request: Request,
    request: Annotated[TranscriptionRequest, Form()],
):
    global _in_flight, _total_errors
    if _transcription_handler is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    audio_data = await request.file.read()
    _in_flight += 1
    try:
        result = await _transcription_handler.create_transcription(
            audio_data, request, raw_request,
        )
    except Exception:
        _total_errors += 1
        raise
    finally:
        _in_flight -= 1
        _update_rps()

    # result can be: TranscriptionResponse[Verbose], ErrorResponse, or AsyncGenerator (stream)
    if hasattr(result, "__aiter__"):
        return StreamingResponse(result, media_type="text/event-stream")
    if hasattr(result, "code"):
        return JSONResponse(
            status_code=getattr(result, "code", 500),
            content=result.model_dump(),
        )
    return JSONResponse(content=result.model_dump())


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

    target_lang = (request.to_language or "en").lower()

    # Build a TranscriptionRequest that mirrors the TranslationRequest but
    # forces verbose_json so we receive the detected source language.
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
    try:
        transcribed = await _transcription_handler.create_transcription(
            audio_data, transcription_req, raw_request,
        )
    except Exception:
        _total_errors += 1
        raise
    finally:
        _in_flight -= 1
        _update_rps()

    # Propagate transcription errors as-is.
    if hasattr(transcribed, "code"):
        return JSONResponse(
            status_code=getattr(transcribed, "code", 500),
            content=transcribed.model_dump(),
        )
    if hasattr(transcribed, "__aiter__"):
        # We requested a non-streaming verbose response; if the handler
        # returned a generator something went wrong upstream.
        raise HTTPException(
            status_code=500,
            detail="Transcription handler returned a stream unexpectedly.",
        )

    text = getattr(transcribed, "text", "") or ""
    source_lang = (getattr(transcribed, "language", "") or "").lower()
    log.info(f"[translate] transcribed type={type(transcribed).__name__} "
             f"source={source_lang!r} target={target_lang!r} text_preview={text[:80]!r}")

    # No translation needed.
    if not text.strip():
        return JSONResponse(content={"text": text})
    if source_lang and source_lang == target_lang:
        log.info("[translate] source==target, skipping LibreTranslate")
        return JSONResponse(content={"text": text})

    # Translate via LibreTranslate.
    try:
        translated = await _libretranslate(text, source_lang or "auto", target_lang)
    except Exception as e:
        log.warning(f"LibreTranslate call failed: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Translation backend failure: {type(e).__name__}: {e}",
        )

    return JSONResponse(content={"text": translated})


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


@app.get("/health")
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
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("main_stt:app", host=host, port=port, log_level="debug" if DEBUG else "info")

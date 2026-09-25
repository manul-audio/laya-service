"""
Laya decision service — thin FastAPI wrapper around convaiinnovations/laya.

Exposes a single POST /predict endpoint that Paperclip agents can call
instead of burning a full LLM call on routine typed decisions (routing,
triage, moderation flags, etc.).

Env vars:
  LAYA_MODEL   — which checkpoint to load (default: convaiinnovations/laya)
  API_KEY      — if set, requests must send it as `Authorization: Bearer <API_KEY>`
"""

import ctypes
import gc
import os
from typing import Any

import laya
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

MODEL_NAME = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")
API_KEY = os.environ.get("API_KEY")  # optional shared-secret auth
USE_BF16 = os.environ.get("LAYA_CPU_AMP", "").lower() in ("bf16", "bfloat16")

app = FastAPI(title="Laya Decision Service")

# Loaded once at startup so requests don't pay model-load latency.
agent = None


def _release_freed_memory() -> None:
    """glibc's allocator does not return freed heap pages to the OS on its own,
    so after replacing large fp32 tensors with bf16 ones, RSS stays near the
    transient peak unless we ask for it back explicitly."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass  # not on glibc (e.g. different base image) — safe to skip


@app.on_event("startup")
def load_model() -> None:
    global agent
    agent = laya.load(MODEL_NAME)
    if USE_BF16:
        # laya's own LAYA_CPU_AMP only enables torch.autocast (compute-time casting);
        # it does not shrink the resident weight memory. Casting the weights themselves
        # is what actually reduces the footprint on CPU.
        agent.model = agent.model.to(torch.bfloat16)
        _release_freed_memory()


class PredictRequest(BaseModel):
    state: dict[str, Any] | str
    questions: dict[str, Any]


def _check_auth(authorization: str | None) -> None:
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME, "loaded": agent is not None}


@app.post("/predict")
def predict(
    body: PredictRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_auth(authorization)
    if agent is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    result = agent.predict(body.state, body.questions)
    return {"result": result}

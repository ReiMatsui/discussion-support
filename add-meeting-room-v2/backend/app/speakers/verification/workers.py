from __future__ import annotations

import sys
import time
import types
from typing import Any

from app.speakers.verification.base import (
    BackendName,
    EmbeddingResult,
    normalize_embedding,
)

_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def embed_in_worker(
    backend: BackendName,
    model_name: str,
    device: str,
    pcm: bytes,
    sample_rate: int,
) -> EmbeddingResult:
    cache_key = (backend, model_name, device)
    load_ms = 0.0
    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        started = time.perf_counter()
        model = _load_model(backend, model_name=model_name, device=device)
        _MODEL_CACHE[cache_key] = model
        load_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    if backend == "speechbrain":
        values = _speechbrain_embedding(model, pcm)
    elif backend == "wespeaker":
        values = _wespeaker_embedding(model, pcm, sample_rate=sample_rate)
    else:
        raise ValueError(f"Unsupported speaker embedding backend: {backend}")
    inference_ms = (time.perf_counter() - started) * 1000

    return EmbeddingResult(
        backend=backend,
        embedding=normalize_embedding(values),
        load_ms=load_ms,
        inference_ms=inference_ms,
    )


def _load_model(backend: BackendName, *, model_name: str, device: str) -> Any:
    if backend == "speechbrain":
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise RuntimeError(
                "SpeechBrain backend requires the speaker-speechbrain extra"
            ) from exc
        return EncoderClassifier.from_hparams(
            source=model_name,
            run_opts={"device": device},
        )

    if backend == "wespeaker":
        try:
            frontend_stub = types.ModuleType("wespeaker.frontend")
            frontend_stub.frontend_class_dict = {"fbank": None}
            sys.modules.setdefault("wespeaker.frontend", frontend_stub)
            import wespeaker
        except ImportError as exc:
            raise RuntimeError(
                "WeSpeaker backend requires the speaker-wespeaker extra"
            ) from exc
        model = wespeaker.load_model(model_name)
        model.set_device(device)
        if model_name == "campplus":
            model.set_wavform_norm(True)
            model.set_window_type("povey")
        return model

    raise ValueError(f"Unsupported speaker embedding backend: {backend}")


def _pcm_tensor(pcm: bytes) -> Any:
    import torch

    values = torch.frombuffer(bytearray(pcm), dtype=torch.int16)
    return values.to(torch.float32).div_(32768.0).unsqueeze(0)


def _speechbrain_embedding(model: Any, pcm: bytes) -> list[float]:
    waveform = _pcm_tensor(pcm)
    embedding = model.encode_batch(waveform)
    return embedding.detach().cpu().reshape(-1).tolist()


def _wespeaker_embedding(model: Any, pcm: bytes, *, sample_rate: int) -> list[float]:
    waveform = _pcm_tensor(pcm)
    embedding = model.extract_embedding_from_pcm(waveform, sample_rate)
    if embedding is None:
        raise RuntimeError("WeSpeaker did not produce an embedding")
    return embedding.detach().cpu().reshape(-1).tolist()

"""ResidencyManager — exactly one pipeline resident on the GPU at a time.

On select(): completely clear the current pipeline off the GPU (drop refs, empty
CUDA cache), then warm up the requested one in a BACKGROUND thread so /status
keeps responding `warming`. predict() runs only against the warm pipeline.
"""
from __future__ import annotations

import gc
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import config
from . import pipelines


def _utc_now():
    return datetime.now(timezone.utc)


class ResidencyManager:
    def __init__(self):
        self.device = config.get_device()
        self._lock = threading.Lock()          # guards state mutations (fast)
        self._infer_lock = threading.Lock()    # serializes GPU inference
        self._states = {k: "cold" for k in pipelines.all_keys()}
        self._resident_key: str | None = None
        self._pipeline = None                  # warm BasePipeline instance
        self._busy = False                     # a warm-up is in progress
        self._last_error: str | None = None

    # --- status ----------------------------------------------------------
    def status(self) -> dict:
        with self._lock:
            infos = []
            for meta in pipelines.list_pipelines():
                infos.append({**meta, "state": self._states[meta["key"]]})
            return {
                "resident": self._resident_key,
                "busy": self._busy,
                "device": str(self.device),
                "pipelines": infos,
            }

    # --- selection / warm-up --------------------------------------------
    def select(self, key: str) -> dict:
        if key not in self._states:
            raise KeyError(key)
        with self._lock:
            if self._busy:
                return {"accepted": False, "state": self._states[key],
                        "detail": "a warm-up is already in progress"}
            if self._resident_key == key and self._states[key] == "warm":
                return {"accepted": True, "state": "warm", "detail": "already warm"}
            self._unload_locked()              # clear whatever is resident
            self._states[key] = "warming"
            self._busy = True
            self._last_error = None
        threading.Thread(target=self._warm_worker, args=(key,), daemon=True).start()
        return {"accepted": True, "state": "warming", "detail": None}

    def _unload_locked(self) -> None:
        """Drop the resident pipeline and free GPU memory. Caller holds _lock."""
        if self._pipeline is not None:
            self._pipeline.model = None
            self._pipeline = None
        if self._resident_key is not None:
            self._states[self._resident_key] = "cold"
            self._resident_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _warm_worker(self, key: str) -> None:
        try:
            pipe = pipelines.get_pipeline(key, self.device)
            pipe.warmup()                       # heavy: build + (download) + load
            with self._lock:
                self._pipeline = pipe
                self._resident_key = key
                self._states[key] = "warm"
        except Exception:
            err = traceback.format_exc()
            with self._lock:
                self._states[key] = "cold"
                self._last_error = err
            print(f"[manager] warm-up failed for {key}:\n{err}")
        finally:
            with self._lock:
                self._busy = False

    # --- inference -------------------------------------------------------
    def predict(self, image_bytes: bytes, filename: str) -> dict:
        with self._lock:
            if self._pipeline is None or self._states.get(self._resident_key) != "warm":
                raise RuntimeError("no pipeline is warm — select one first")
            pipe = self._pipeline
            key = self._resident_key

        # Persist the upload first, then run inference (clip-probe embeds by path).
        ts = _utc_now()
        stamp = ts.strftime("%Y%m%dT%H%M%S_%fZ")
        out_dir = config.PREDICTIONS_DIR / pipe.pipeline / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = out_dir / filename
        image_path.write_bytes(image_bytes)

        with self._infer_lock:                  # one inference at a time on the GPU
            result = pipe.predict(image_path)

        record = {
            "pipeline": pipe.pipeline,
            "key": key,
            "timestamp": ts.isoformat(timespec="seconds"),
            "image": filename,
            "final": result["final"],
            "components": result["components"],
            "saved_to": str(out_dir),
        }
        (out_dir / "prediction.json").write_text(_dumps(record))
        return record

    def explain(self, image_bytes: bytes, filename: str) -> dict:
        with self._lock:
            if self._pipeline is None or self._states.get(self._resident_key) != "warm":
                raise RuntimeError("no pipeline is warm — select one first")
            pipe = self._pipeline

        import os
        import tempfile
        suffix = Path(filename).suffix or ".png"
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        Path(tmp).write_bytes(image_bytes)
        try:
            with self._infer_lock:
                res = pipe.explain(tmp)
        finally:
            os.remove(tmp)

        overlay = res.get("overlay")
        return {
            "available": bool(res.get("available")),
            "method": res.get("method"),
            "overlay_png_b64": _png_b64(overlay) if overlay is not None else None,
            "reason": res.get("reason"),
        }


def _dumps(obj) -> str:
    import json
    return json.dumps(obj, indent=2)


def _png_b64(arr) -> str:
    """Encode an HWC uint8 array as a base64 PNG string."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype="uint8")).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

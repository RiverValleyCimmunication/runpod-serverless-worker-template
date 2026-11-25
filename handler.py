#!/usr/bin/env python3
"""RunPod serverless worker for the Emulator check engine.

Job input (one of):
  {"model_b64": "<base64 ONNX bytes>"}   — small models; RunPod caps
      the /runsync request at 20 MB, so raw files up to ~14 MB.
  {"model_url": "https://.../model.onnx"} — the worker fetches the
      file itself; for larger models staged at a URL by the service.

Returns the checker's JSON: {"runs", "pace_per_sec", "input",
"error", "cpus", "engine"}. The model file is written to a temp path
inside the worker and deleted in a finally block — worker-side copies
are ephemeral by design; the service keeps only the fingerprint and
the result (Mario's rule, 2026-11-24).
"""
import base64
import json
import os
import tempfile
import urllib.request

import runpod

from check_logic import measure

MAX_URL_BYTES = 2 * 1024 * 1024 * 1024  # match the service's upload cap


def _fetch_to_temp(job_input):
    """Materialize the model to a temp file; return its path."""
    if job_input.get("model_b64"):
        data = base64.b64decode(job_input["model_b64"])
        fd, path = tempfile.mkstemp(suffix=".onnx")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path
    if job_input.get("model_url"):
        fd, path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        req = urllib.request.Request(
            job_input["model_url"],
            headers={"User-Agent": "emulator-check-worker/1"})
        with urllib.request.urlopen(req, timeout=300) as r, \
                open(path, "wb") as f:
            total = 0
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_URL_BYTES:
                    os.remove(path)
                    raise ValueError("model file exceeds the 2 GB cap")
                f.write(chunk)
        return path
    raise ValueError("no model supplied: send model_b64 or model_url")


def _stage_callback(job_input):
    """Build the stage callback from the job's stage_url/stage_token/
    check_id fields, or None when the job carries none. Each report is
    a short POST on its own daemon thread: the customer's door hears
    the real stages (running -> measuring) and a dead callback can
    never slow or fail the check."""
    url = job_input.get("stage_url")
    token = job_input.get("stage_token")
    check_id = job_input.get("check_id")
    if not url or not token or check_id is None:
        return None
    import threading
    import urllib.request as _ur

    def report(name):
        def _send():
            try:
                body = json.dumps({"check_id": check_id, "token": token,
                                   "stage": name}).encode()
                req = _ur.Request(url, data=body, method="POST")
                req.add_header("Content-Type", "application/json")
                with _ur.urlopen(req, timeout=5) as r:
                    r.read()
            except Exception:
                pass
        threading.Thread(target=_send, daemon=True).start()

    return report


def handler(job):
    job_input = (job or {}).get("input") or {}
    path = None
    try:
        path = _fetch_to_temp(job_input)
        # Optional declared preprocessing ("imagenet") — the checker
        # bakes it into the graph and the recipe on the result says so.
        return measure(path, job_input.get("preprocess"),
                       _stage_callback(job_input))
    except Exception as e:
        return {"runs": False, "pace_per_sec": 0.0, "input": "",
                "error": f"{type(e).__name__}: {e}"[:400]}
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

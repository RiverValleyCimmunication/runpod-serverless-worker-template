"""The Emulator's generic ONNX check, as a callable.

This is box_runner.py's measurement logic in function form — the same
code the Vultr route ran on the box: load the model with OpenVINO on
CPU, resolve its input shape (batch dim -> 1, other dynamic dims ->
224), warm up, then time inference for ~4 seconds and report the
pace. The quote is computed from THIS measurement, on THIS machine
class — never from a plan-level or remembered speed.
"""
import time

import numpy as np


def measure(model_path):
    out = {"runs": False, "pace_per_sec": 0.0, "input": "", "error": ""}
    try:
        import openvino as ov
    except Exception as e:
        out["error"] = f"engine unavailable: {e}"
        return out
    try:
        import os
        out["cpus"] = os.cpu_count()
        out["engine"] = ov.__version__
        core = ov.Core()
        model = core.read_model(model_path)
        inp = model.input(0)
        ps = inp.partial_shape
        etype = str(inp.element_type)
        # Resolve dynamic dims conservatively: batch 1, others 224
        # for spatial-looking dims, else 1. NOTE: .shape throws on
        # dynamic shapes — partial_shape is the safe read.
        resolved = []
        for i in range(len(ps)):
            d = ps[i]
            if d.is_dynamic:
                resolved.append(1 if i == 0 else 224)
            else:
                resolved.append(d.get_length())
        out["input"] = "x".join(str(x) for x in resolved) + " " + etype
        compiled = core.compile_model(model, "CPU",
                                      {"PERFORMANCE_HINT": "THROUGHPUT"})
        rng = np.random.default_rng(7)
        if "u8" in etype or "uint8" in etype:
            sample = rng.integers(0, 256, size=resolved, dtype=np.uint8)
        else:
            sample = rng.random(resolved, dtype=np.float32)
        key = compiled.input(0)
        # Warmup
        for _ in range(5):
            compiled({key: sample})
        # Timed: at least ~4 seconds of inference
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < 4.0:
            compiled({key: sample})
            n += 1
        dt = time.perf_counter() - t0
        items = n * (resolved[0] if resolved else 1)
        out["pace_per_sec"] = round(items / dt, 2)
        out["runs"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:400]
    return out

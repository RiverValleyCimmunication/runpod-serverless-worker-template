"""The Emulator's generic ONNX check, as a callable — FORMULA EDITION.

Mario's ruling (2026-11-25): the locked formula is the checker's
STANDARD measurement, not a special mode. Where it applies, the checker
measures with the tuned pipeline the run lane delivers:

  * AsyncInferQueue (Test 4: ~2x over sync on the same host)
  * batch-4 for any model with a usable batch dimension (Tests 6-7:
    the measured sweet spot)
  * preprocessing baked into the graph via OpenVINO's PrePostProcessor
    when the model's preprocessing is declared (job field
    "preprocess": "imagenet" — raw 0..255 NCHW pixels in, ImageNet
    scale/mean/std fused into the model)

INTEGRITY RULE: the quoted pace is the pace of the recipe named on the
result — checker and runner use the same recipe, never a tuned number
for an untuned run. HONEST FALLBACK: when a formula element cannot
apply to a submitted model, the checker measures as-submitted for that
element and the recipe string on the result says exactly what ran
(e.g. "as-submitted: async only" or a declined-bake note). A pace
number belongs to a host AND a recipe; both ride on every result.
"""
import time

import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _resolve(ps, batch):
    """Resolve a partial shape: batch dim -> batch, other dynamic dims
    -> 224 for spatial-looking positions, else 1 (the original rule)."""
    resolved = []
    for i in range(len(ps)):
        d = ps[i]
        if d.is_dynamic:
            resolved.append(batch if i == 0 else 224)
        else:
            resolved.append(d.get_length())
    return resolved


def measure(model_path, preprocess=None):
    out = {"runs": False, "pace_per_sec": 0.0, "input": "", "error": "",
           "recipe": ""}
    try:
        import openvino as ov
    except Exception as e:
        out["error"] = f"engine unavailable: {e}"
        out["recipe"] = "none (engine unavailable)"
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
        rank = len(ps)

        # --- formula element 1: batch-4 where a batch dim applies ---
        batch = 1
        batch_note = ""
        if rank >= 1:
            d0 = ps[0]
            if d0.is_dynamic or d0.get_length() == 1:
                batch = 4
            else:
                batch_note = f"batch fixed at {d0.get_length()} by the model"
        resolved = _resolve(ps, batch)
        out["input"] = "x".join(str(x) for x in resolved) + " " + etype

        # --- formula element 2: baked preprocessing, when declared ---
        baked = False
        bake_note = ""
        if preprocess:
            if preprocess != "imagenet":
                bake_note = f"preprocessing '{preprocess}' not known; not baked"
            elif rank != 4:
                bake_note = "bake declined: imagenet preprocessing needs a 4-D input"
            elif not (ps[1].is_static and ps[1].get_length() == 3):
                bake_note = "bake declined: imagenet preprocessing needs 3 channels"
            elif "f32" not in etype and "float" not in etype:
                bake_note = "bake declined: imagenet preprocessing needs a float input"
            else:
                ppp = ov.preprocess.PrePostProcessor(model)
                # Per-channel steps need the channel dim named.
                ppp.input().tensor().set_layout(ov.Layout("NCHW"))
                ppp.input().preprocess().scale(
                    [255.0, 255.0, 255.0]).mean(IMAGENET_MEAN).scale(
                        IMAGENET_STD)
                model = ppp.build()
                baked = True
                # PPP rebuilds the model; the old input port is stale.
                inp = model.input(0)

        # Reshape to the resolved (possibly batch-4) input. If the model
        # refuses the batch-4 reshape, fall back to batch 1 honestly.
        try:
            model.reshape({inp: resolved})
        except Exception:
            if batch != 1:
                batch = 1
                batch_note = "batch-4 reshape refused by the model; measured at batch 1"
                resolved = _resolve(ps, batch)
                out["input"] = "x".join(
                    str(x) for x in resolved) + " " + etype
                model.reshape({inp: resolved})
            else:
                raise

        compiled = core.compile_model(
            model, "CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
        rng = np.random.default_rng(7)
        if "u8" in etype or "uint8" in etype:
            sample = rng.integers(0, 256, size=resolved, dtype=np.uint8)
        elif baked:
            # Declared contract: raw pixels, 0..255.
            sample = (rng.random(resolved, dtype=np.float32) * 255.0)
        else:
            sample = rng.random(resolved, dtype=np.float32)
        key = compiled.input(0)

        # Warmup (sync, a few passes)
        for _ in range(3):
            compiled({key: sample})

        # --- formula element 3: async queue, timed ~4 seconds ---
        jobs = 8
        queue = ov.AsyncInferQueue(compiled, jobs)
        done = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 4.0:
            for _ in range(jobs):
                queue.start_async({key: sample})
            queue.wait_all()
            done += jobs
        dt = time.perf_counter() - t0
        items = done * (resolved[0] if resolved else 1)
        out["pace_per_sec"] = round(items / dt, 2)
        out["runs"] = True

        parts = []
        if baked:
            parts.append("baked-prep(imagenet)")
        if batch == 4:
            parts.append("batch-4")
        parts.append("async")
        notes = [n for n in (bake_note, batch_note) if n]
        if baked or batch == 4:
            out["recipe"] = "formula: " + "+".join(parts)
        else:
            out["recipe"] = "as-submitted: " + "+".join(parts)
        if notes:
            out["recipe"] += " (" + "; ".join(notes) + ")"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:400]
        if not out["recipe"]:
            out["recipe"] = "none (measurement failed)"
    return out

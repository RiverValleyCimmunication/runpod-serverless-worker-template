# Emulator check worker

RunPod serverless CPU worker for the Emulator check engine: receives an ONNX model (base64 or URL), measures it with OpenVINO on CPU, returns the pace JSON. Built to GHCR by Actions on every push to main.

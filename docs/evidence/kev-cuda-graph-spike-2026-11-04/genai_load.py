import json
from pathlib import Path

import onnxruntime as ort

ort.preload_dlls()
import onnxruntime_genai as og

path = "/home/mark/Projects/AI-models/kev/spike-2026-11-04/export-cuda-fp32"
out = str(Path(__file__).resolve().parent / "results" / "load-model-genai.json")
result = {"graph": path + "/model.onnx", "onnxruntime_genai": og.__version__, "onnxruntime": ort.__version__}
try:
    og.Model(path)
    result["ok"] = True
except Exception as e:  # noqa: BLE001 - the error text is the result
    result["ok"] = False
    result["error"] = str(e).strip()[:600]
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1)
    f.write("\n")
print(result)

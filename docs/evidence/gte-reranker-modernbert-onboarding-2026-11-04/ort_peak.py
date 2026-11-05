"""Peak memory of ONNX Runtime alone on one long pair: no PyTorch in the process.

Usage: python -I ort_peak.py <snapshot> <tokens> <runs> <arena|noarena>. One setting per process: ru_maxrss never falls.
"""

import resource
import sys
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


snapshot, tokens, runs, arena = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
raw = Tokenizer.from_file(snapshot + "/tokenizer.json")
raw.no_padding()
raw.enable_truncation(tokens, strategy="longest_first", direction="right")
query = "how to implement quick sort in python?"
document = "the quick brown fox jumps over the lazy dog " * 1000
ids = raw.encode(query, document).ids
assert len(ids) == tokens, len(ids)
x = np.array([ids], dtype=np.int64)
mask = np.ones_like(x)

options = ort.SessionOptions()
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
options.enable_cpu_mem_arena = arena == "arena"
t0 = time.time()
session = ort.InferenceSession(snapshot + "/onnx/model.onnx", options, providers=["CPUExecutionProvider"])
t1 = time.time()
print(f"onnxruntime {ort.__version__} {arena} session load {t1 - t0:.1f}s peak {peak_gb():.2f}G", flush=True)
for run in range(1, runs + 1):
    t2 = time.time()
    logits = session.run(["logits"], {"input_ids": x, "attention_mask": mask})[0]
    t3 = time.time()
    print(f"tokens {tokens} run {run} logit {logits.item():.6f} time {t3 - t2:.1f}s peak {peak_gb():.2f}G", flush=True)

import resource
import sys
import time

import numpy as np
import onnxruntime as ort
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

S = sys.argv[1]
torch.set_num_threads(4)
raw = Tokenizer.from_file(S + "/tokenizer.json")
raw.no_padding()
raw.enable_truncation(8192, strategy="longest_first", direction="right")
q = "how to implement quick sort in python?"
d = "the quick brown fox jumps over the lazy dog " * 1000
ids = raw.encode(q, d).ids
auto = AutoTokenizer.from_pretrained(S)(q, d, truncation=True)["input_ids"]
print("len", len(ids), "same as transformers truncation=True", ids == auto, flush=True)
x = np.array([ids], dtype=np.int64)
m = np.ones_like(x)
model = AutoModelForSequenceClassification.from_pretrained(S, dtype=torch.float32).eval()
t0 = time.time()
with torch.inference_mode():
    a = model(input_ids=torch.from_numpy(x), attention_mask=torch.from_numpy(m)).logits.numpy()
t1 = time.time()
del model
o = ort.SessionOptions()
o.intra_op_num_threads = 4
o.inter_op_num_threads = 1
s = ort.InferenceSession(S + "/onnx/model.onnx", o, providers=["CPUExecutionProvider"])
t2 = time.time()
b = s.run(["logits"], {"input_ids": x, "attention_mask": m})[0]
t3 = time.time()
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print("pt", a.ravel(), "ort", b.ravel(), "d", abs(a - b).max(), f"pt {t1 - t0:.1f}s ort {t3 - t2:.1f}s rss {rss:.1f}G")

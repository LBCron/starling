"""Repair 1 for the onnxruntime-genai 0.17.1 `-e cuda -p fp32` export of Kev-0.8B (Qwen3.5 hybrid).

The builder's attention-mask reformat subgraph (used with MultiHeadAttention, the fp32 CUDA attention op) reads the
past sequence length from `past_key_values.0.key`, assuming layer 0 is an attention layer. In Qwen3.5 layer 0 is a
Gated DeltaNet layer, so that input does not exist and the model fails to load. This rewires every such reference to
the first attention layer's past key (`past_key_values.3.key`), which has the same [B, kv_heads, past, head_dim]
layout. Weights are untouched (same external data file).
"""

import sys

import onnx

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)
inputs = {i.name for i in m.graph.input}
first = next(n for n in (f"past_key_values.{i}.key" for i in range(64)) if n in inputs)
changed = []
for node in m.graph.node:
    for k, name in enumerate(node.input):
        if name.startswith("past_key_values.") and name not in inputs:
            layer_kind = name.rsplit(".", 1)[-1]
            node.input[k] = first if layer_kind == "key" else first.replace(".key", ".value")
            changed.append((node.name, name, node.input[k]))
print(changed)
onnx.save(m, dst)

"""Repair 1 for failure B of the onnxruntime-genai 0.17.1 `-e cuda -p fp32` export (MultiHeadAttention path).

Failure: a multi-token run on a non-empty cache (the Kev question branch continuing from the cached state) fails in
/model/attn_mask_reformat/input_ids_subgraph/Expand: the causal triangle is built as [S, S] (S = new tokens) and cannot
broadcast to [B, 1, S, past + S]. The builder's mask therefore supports prefill (past = 0) and one-token decode only.

Repair: build the triangle as [S, past + S] with row i allowing columns < past + i + 1:
  - ConstantOfShape_2's shape becomes [S, past + S] (Concat_2's second input -> Unsqueeze_3, which is past + S);
    its last dim then drives the existing column Range;
  - the row operand of Less becomes Reshape(Range(0, S) + 1 + past, [-1, 1]) instead of the column range reshaped.
Input: model-patched.onnx (repair 1 of failure A). Weights untouched.
"""

import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)
P = "/model/attn_mask_reformat/input_ids_subgraph/"
nodes = {n.name: n for n in m.graph.node}
concat2, less = nodes[P + "Concat_2"], nodes[P + "Less"]
s_scalar = nodes[P + "Gather_2"].output[0]
past_scalar = nodes["/model/attn_mask_reformat/past_key_subgraph/Gather"].output[0]
concat2.input[1] = nodes[P + "Unsqueeze_3"].output[0]  # past + S
m.graph.initializer.extend(
    [
        numpy_helper.from_array(np.array(0, dtype=np.int64), "kev_fix/zero"),
        numpy_helper.from_array(np.array(1, dtype=np.int64), "kev_fix/one"),
        numpy_helper.from_array(np.array([-1, 1], dtype=np.int64), "kev_fix/col_shape"),
    ]
)
new = [
    helper.make_node("Range", ["kev_fix/zero", s_scalar, "kev_fix/one"], ["kev_fix/rows"], name="kev_fix/Range"),
    helper.make_node("Add", ["kev_fix/rows", "kev_fix/one"], ["kev_fix/rows1"], name="kev_fix/Add1"),
    helper.make_node("Add", ["kev_fix/rows1", past_scalar], ["kev_fix/rows_past"], name="kev_fix/AddPast"),
    helper.make_node("Reshape", ["kev_fix/rows_past", "kev_fix/col_shape"], ["kev_fix/row_bound"], name="kev_fix/Reshape"),
]
less.input[1] = "kev_fix/row_bound"
# keep topological order: insert the new nodes just before Less
idx = list(m.graph.node).index(less)
for k, n in enumerate(new):
    m.graph.node.insert(idx + k, n)

onnx.save(m, dst)
print("patched", dst)

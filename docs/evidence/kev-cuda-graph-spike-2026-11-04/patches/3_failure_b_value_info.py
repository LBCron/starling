"""Repair 3, the second for failure B (cached continuation on the -e cuda -p fp32 graph): after the mask fix, ORT's
allocation planner reuses buffers across tensors whose declared symbolic dims say they are equal but are not when
past > 0 (repeat_kv: {1,703,8,256} != {1,693,8,256}). Drop all of the graph's intermediate value_info so ORT infers
shapes itself. Which entry declares the wrong dim was not found, so this removes the symptom, not its root cause."""

import sys

import onnx

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)
n_vi = len(m.graph.value_info)
del m.graph.value_info[:]
print("removed value_info entries:", n_vi)
onnx.save(m, dst)

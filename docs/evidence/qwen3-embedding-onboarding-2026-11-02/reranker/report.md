# Development parity: qwen3-reranker-0.6b/onnx-fp32-cpu

Status **failed**, 234 checks, 3 failed. References are proposed; bounds are the candidate's predeclared policy (issue #144).

| Comparison | Largest difference | Bound |
| --- | ---: | ---: |
| dotnet-ids/score_max_abs | 2.384e-05 | 0.001 |
| dotnet-public/score_max_abs | 1.619e-05 | 0.001 |
| dotnet-text/score_max_abs | 2.384e-05 | 0.001 |
| python-onnx/hidden_max_abs | 3.653e-03 | 0.002 |
| python-onnx/score_max_abs | 2.098e-05 | 0.001 |
| sentence-transformers/score_max_abs | 1.240e-05 | 0.001 |

Exact checks: 60 of 60 passed.

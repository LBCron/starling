# Development parity: qwen3-embedding-0.6b/onnx-fp32-cpu

Status **passed**, 558 checks, 0 failed. References are proposed; bounds are the candidate's predeclared policy (issue #144).

| Comparison | Largest difference | Bound |
| --- | ---: | ---: |
| dotnet-ids/embedding_max_abs | 6.034e-07 | 0.0001 |
| dotnet-ids/norm_max_abs | 5.674e-09 | 1e-05 |
| dotnet-text/cosine_max_abs | 3.576e-07 | 0.001 |
| dotnet-text/embedding_max_abs | 6.034e-07 | 0.0001 |
| dotnet-text/norm_max_abs | 5.674e-09 | 1e-05 |
| python-onnx/cosine_max_abs | 3.874e-07 | 0.001 |
| python-onnx/embedding_max_abs | 6.035e-07 | 0.0001 |
| python-onnx/hidden_max_abs | 1.070e-03 | 0.002 |
| sentence-transformers/embedding_max_abs | 2.552e-07 | 0.0001 |

Exact checks: 250 of 250 passed.

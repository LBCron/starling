"""Qwen3 Embedding series 0.6B (embedding and reranker) on ONNX Runtime CPU fp32 (starlinghq/starling#144).

A Qwen3 decoder used non-autoregressively. Explicit allow-list: the two pinned upstream checkpoints, the Starling-built
graphs hosted at `Starlinghq/qwen3-embedding-0.6b-onnx` and `Starlinghq/qwen3-reranker-0.6b-onnx`, and the behavior
contract of issue #144 are checked field by field here and in the .NET fidelity tool (against the constants and locks
of `Starlings.Models.Qwen3Embedding`). Anything else is refused. Stages:

    assets -> environment -> reference -> build -> onnx -> dotnet -> evaluate -> review

`--snapshot` is the pinned upstream snapshot (PyTorch weights and the files the hosted directory copies). The build
stage runs the export recipe (`reference/qwen3-embedding/export_qwen3_onnx.py`, then `release_qwen3_onnx.py`) and
requires the released graph and external data to be byte-identical to the hosted files the candidate pins, so the
evidence is about the bytes `Qwen3EmbeddingProfile.DownloadAsync` fetches. The onnx stage runs `verify_qwen3_onnx.py`
(ONNX Runtime against PyTorch, hidden states included, card cosines, sentence-transformers cross-checks); the dotnet
stage runs the .NET library on the released directory. References are proposals; nothing adopts protected fixtures.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import replace
from pathlib import Path

from model_pipeline import fixtures
from model_pipeline.onboarding_candidate import guard_output, repository_input, write_json
from model_pipeline.onboarding_runner import PipelineError, Stage, run_command

PLUGIN = "qwen3-embedding-onnx-cpu-v1"
SHARED_DISTRIBUTION = True
DIRECTORY = "model-pipeline/reference/qwen3-embedding"
TOOL = "ai-dotnet/tools/Starlings.Qwen3Embedding.Fidelity.Tool/Starlings.Qwen3Embedding.Fidelity.Tool.csproj"
STAGES = ["assets", "environment", "reference", "build", "onnx", "dotnet", "evaluate", "review"]
RUNTIME = {"backend": "onnx", "version": "1.30.0", "provider": "CPUExecutionProvider", "precision": "fp32"}
QUERY_PROMPT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
POLICY = {
    "token_ids": "exact",
    "non_finite": "fail",
    "hidden_max_abs": 2e-3,
    "embedding_max_abs": 1e-4,
    "score_max_abs": 1e-3,
    "card_cosine_max_abs": 1e-3,
    "norm_max_abs": 1e-5,
}
# Route caps of issue #144: a candidate policy may never exceed them.
CAPS = {"hidden_max_abs": 5e-3, "embedding_max_abs": 1e-3, "score_max_abs": 1e-2}
CAPABILITIES = {
    "embeddings": "experimental",
    "prompts": "experimental",
    "reranking": "experimental",
    "batching": "experimental",
    "long_input_8192": "experimental",
    "extensions_ai": "experimental",
    "download": "experimental",
    "matryoshka_dimensions": "unsupported",
    "cuda": "unsupported",
    "reduced_precision": "unsupported",
    "larger_checkpoints": "unsupported",
    "beyond_8192_tokens": "unsupported",
}
COMMON = {
    "architecture": "Qwen3ForCausalLM",
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "vocabulary_size": 151669,
    "max_tokens": 8192,
    "pad_token_id": 151643,
    "padding_side": "right",
    "tokenizer_file": "tokenizer.json",
    "graph": {
        "exporter": "dynamo",
        "opset": 20,
        "inputs": ["input_ids", "attention_mask"],
    },
    "capabilities": CAPABILITIES,
}
CONTRACTS = {
    "Qwen/Qwen3-Embedding-0.6B": {
        "id": "qwen3-embedding-0.6b/onnx-fp32-cpu",
        "task": "embeddings",
        "model": "embedding",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "files": (
            "1_Pooling/config.json",
            "README.md",
            "config.json",
            "config_sentence_transformers.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "modules.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ),
        "tokenizer_sha256": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
        "hosting": {
            "repo": "Starlinghq/qwen3-embedding-0.6b-onnx",
            "files": {
                "model.onnx": (337132, "de6805d74481fefc313ea470de3877f64fe4093c99ce9d1172a026128d90d864"),
                "model.onnx.data": (2383077376, "7c7569e58783ee0ad8c5fb797d7944aa4f5928af53fb4c1f626f71885af22969"),
            },
        },
        "behavior": {
            **COMMON,
            "graph": {**COMMON["graph"], "outputs": ["last_hidden_state"]},
            "tokenizer_profile": "qwen3-bpe-embedding",
            "eos_token_id": 151643,
            "append_eos": True,
            "pooling": "last_token",
            "normalize": "l2",
            "dimension": 1024,
            "prompts": {"none": "", "document": "", "query": QUERY_PROMPT},
        },
    },
    "Qwen/Qwen3-Reranker-0.6B": {
        "id": "qwen3-reranker-0.6b/onnx-fp32-cpu",
        "task": "reranking",
        "model": "reranker",
        "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "files": (
            "1_LogitScore/config.json",
            "README.md",
            "chat_template.jinja",
            "config.json",
            "config_sentence_transformers.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "modules.json",
            "sentence_bert_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ),
        "tokenizer_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "hosting": {
            "repo": "Starlinghq/qwen3-reranker-0.6b-onnx",
            "files": {
                "model.onnx": (337355, "b162410db82c0e26f231cda7ddf1ec659ed6707c90e82e5556fb8f1076997581"),
                "model.onnx.data": (2383085568, "fbfe2bddcffc9e8b4a1145bf02f4ebc99119e57e38d32bf7f04172a7d0cf9c7c"),
            },
        },
        "behavior": {
            **COMMON,
            "graph": {**COMMON["graph"], "outputs": ["yes_no_logits", "last_hidden_state"]},
            "tokenizer_profile": "qwen3-bpe-chat",
            "score": "logit_yes_minus_no",
            "yes_token_id": 9693,
            "no_token_id": 2152,
            "default_instruction": DEFAULT_INSTRUCTION,
            "prefix": RERANK_PREFIX,
            "suffix": RERANK_SUFFIX,
            "truncation": "content_right",
        },
    },
}


def _refuse(field: str, detail: str) -> None:
    raise PipelineError(f"unsupported Qwen3 Embedding contract field {field}: {detail}")


def _same(field: str, actual, expected) -> None:
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        _refuse(field, f"expected {expected!r}, found {actual!r}")


def validate_candidate_behavior(candidate: dict) -> dict:
    """Refuse every contract this route does not implement; returns the matching allow-list entry."""
    try:
        origin = candidate["origin"]
        repo = origin["repo"]
        if repo not in CONTRACTS:
            _refuse("origin.repo", f"{repo!r} is not an allow-listed Qwen3 Embedding series checkpoint")
        contract = CONTRACTS[repo]
        _same("format_version", candidate["format_version"], 1)
        _same("id", candidate["id"], contract["id"])
        _same("task", candidate["task"], contract["task"])
        _same("origin.revision", origin["revision"], contract["revision"])
        extra = set(origin) - {"repo", "revision", "files"}
        if extra:
            _refuse("origin", f"unknown fields {sorted(extra)}")
        paths = [record.get("path") for record in origin["files"]]
        if sorted(paths) != sorted(contract["files"]) or len(set(paths)) != len(paths):
            _refuse("origin.files", f"pinned files must be exactly {sorted(contract['files'])}")
        tokenizer = next(record for record in origin["files"] if record["path"] == "tokenizer.json")
        _same("origin.files[tokenizer.json].sha256", tokenizer["sha256"], contract["tokenizer_sha256"])
        _same("runtime", candidate["runtime"], RUNTIME)
        behavior, expected = candidate["behavior"], contract["behavior"]
        if set(behavior) != set(expected) | {"hosting"}:
            _refuse("behavior", f"fields must be exactly {sorted(set(expected) | {'hosting'})}")
        for key, value in expected.items():
            _same(f"behavior.{key}", behavior[key], value)
        hosting = behavior["hosting"]
        if set(hosting) != {"repo", "revision", "files"}:
            _refuse("behavior.hosting", "fields must be exactly ['files', 'repo', 'revision']")
        _same("behavior.hosting.repo", hosting["repo"], contract["hosting"]["repo"])
        revision = hosting["revision"]
        if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            _refuse("behavior.hosting.revision", "must be a full 40-character commit SHA")
        files = {record["path"]: (record["size"], record["sha256"]) for record in hosting["files"]}
        if len(files) != len(hosting["files"]):
            _refuse("behavior.hosting.files", "duplicate paths")
        _same("behavior.hosting.files", files, {k: list(v) for k, v in contract["hosting"]["files"].items()})
        _same("evaluation.status", candidate["evaluation"]["status"], "proposed")
        _same("evaluation.policy", candidate["evaluation"]["policy"], POLICY)
        for key, cap in CAPS.items():
            if not POLICY[key] <= cap:
                _refuse(f"evaluation.policy.{key}", f"exceeds the route cap {cap}")
        return contract
    except (KeyError, TypeError, AttributeError, StopIteration) as exc:
        raise PipelineError(f"unsupported or incomplete Qwen3 Embedding contract: {exc!r}") from exc


def _profile(recipe: dict, root: Path) -> tuple[Path, dict, dict]:
    from model_pipeline.onboarding_candidate import load_candidate

    if recipe.get("plugin") != PLUGIN or recipe.get("recipe_version") != 1:
        raise PipelineError(f"expected recipe_version 1 and plugin {PLUGIN}")
    path = repository_input(root, recipe["manifest"])
    candidate = load_candidate(path)
    return path, candidate, validate_candidate_behavior(candidate)


def describe_plan(recipe: dict, *, repo_root: Path) -> dict:
    _, candidate, contract = _profile(recipe, repo_root.resolve())
    hosting = candidate["behavior"]["hosting"]
    return {
        "plugin": PLUGIN,
        "recipe": recipe["id"],
        "profile": candidate["id"],
        "task": candidate["task"],
        "revision": candidate["origin"]["revision"],
        "hosting": {"repo": hosting["repo"], "revision": hosting["revision"]},
        "asset_count": len(candidate["origin"]["files"]),
        "stages": list(STAGES),
        "inputs": {
            "snapshot": f"local {candidate['origin']['repo']} snapshot at {candidate['origin']['revision']}",
            "reference_python": f"{DIRECTORY}/requirements.txt",
        },
        "policy": candidate["evaluation"]["policy"],
        "capabilities": candidate["behavior"]["capabilities"],
        "reference_status": "proposed; separate independent adoption required",
        "qualification_status": "unapproved",
        "model": contract["model"],
    }


def _check(checks: list, check_id: str, value: float, bound: float) -> None:
    ok = math.isfinite(value) and value <= bound
    checks.append({"id": check_id, "max_abs": value, "bound": bound, "status": "passed" if ok else "failed"})


def _exact(checks: list, check_id: str, ok: bool, detail: str = "") -> None:
    checks.append({"id": check_id, "status": "passed" if ok else "failed", **({"detail": detail} if detail else {})})


def evaluate(candidate: dict, reference: dict, build: dict, verify: dict, dotnet: dict) -> dict:
    """Independent comparison of the PyTorch reference, Python ONNX Runtime and .NET under the candidate policy."""
    policy = candidate["evaluation"]["policy"]
    task = candidate["task"]
    checks: list[dict] = []
    _exact(checks, "build/hosted-bytes", build["hosted_bytes_match"], json.dumps(build["released_graph"]))
    if verify["model"] != ("embedding" if task == "embeddings" else "reranker"):
        raise PipelineError("onnx verification ran on the other model")
    _check(checks, "python-onnx/all/hidden_max_abs", verify["hidden_max_abs"], policy["hidden_max_abs"])
    for case in verify["cases"]:
        _check(checks, f"python-onnx/{case['id']}/hidden_max_abs", case["hidden_max_abs"], policy["hidden_max_abs"])
    cases = {case["id"]: case for case in reference["cases"]}
    produced = {case["id"]: case for case in dotnet["cases"]}
    if set(produced) != set(cases):
        raise PipelineError(".NET cases differ from the reference cases")
    if task == "embeddings":
        _exact(checks, "python-onnx/eos-last", verify["eos_last_all_inputs"] is True)
        _check(checks, "python-onnx/all/embedding_max_abs", verify["embedding_max_abs"], policy["embedding_max_abs"])
        cosines = verify["card_cosines"]
        _check(
            checks,
            "python-onnx/card/cosine_max_abs",
            cosines["graph_max_abs_vs_published"],
            policy["card_cosine_max_abs"],
        )
        _check(
            checks,
            "sentence-transformers/card/embedding_max_abs",
            verify["sentence_transformers_embedding_max_abs"],
            policy["embedding_max_abs"],
        )
        for case_id, ref in cases.items():
            got = produced[case_id]
            _exact(checks, f"dotnet-text/{case_id}/token_ids", got["ids"] == ref["ids"])
            _exact(
                checks, f"dotnet-text/{case_id}/eos-last", got["ids"][-1:] == [candidate["behavior"]["eos_token_id"]]
            )
            expected = ref["embedding"]
            for path in ("text", "ids"):
                values = got[f"embedding_{path}"]
                finite = all(math.isfinite(v) for v in values) and len(values) == len(expected)
                _exact(checks, f"dotnet-{path}/{case_id}/finite", finite)
                diff = max(abs(a - b) for a, b in zip(values, expected, strict=True)) if finite else math.inf
                _check(checks, f"dotnet-{path}/{case_id}/embedding_max_abs", diff, policy["embedding_max_abs"])
                norm = math.sqrt(sum(v * v for v in values)) if finite else math.inf
                _check(checks, f"dotnet-{path}/{case_id}/norm_max_abs", abs(norm - 1.0), policy["norm_max_abs"])
        _check(
            checks,
            "dotnet-text/card/cosine_max_abs",
            max(
                abs(a - b)
                for r, p in zip(dotnet["card_cosines"], cosines["published"], strict=True)
                for a, b in zip(r, p, strict=True)
            ),
            policy["card_cosine_max_abs"],
        )
    else:
        _exact(checks, "python-onnx/chat-template", verify["chat_template_matches_prefix_content_suffix"] is True)
        _check(checks, "python-onnx/all/score_max_abs", verify["score_max_abs"], policy["score_max_abs"])
        _check(
            checks,
            "sentence-transformers/card/score_max_abs",
            verify["sentence_transformers"]["max_abs"],
            policy["score_max_abs"],
        )
        _exact(checks, "dotnet-text/template/prefix_ids", dotnet["prefix_ids"] == reference["prefix_ids"])
        _exact(checks, "dotnet-text/template/suffix_ids", dotnet["suffix_ids"] == reference["suffix_ids"])
        for case_id, ref in cases.items():
            got = produced[case_id]
            _exact(checks, f"dotnet-text/{case_id}/token_ids", got["ids"] == ref["ids"])
            for path in ("text", "ids"):
                value = got[f"score_{path}"]
                diff = abs(value - ref["score"]) if math.isfinite(value) else math.inf
                _check(checks, f"dotnet-{path}/{case_id}/score_max_abs", diff, policy["score_max_abs"])
        for item in dotnet["public_scores"]:
            ref = cases[item["id"]]["score"]
            _check(
                checks, f"dotnet-public/{item['id']}/score_max_abs", abs(item["score"] - ref), policy["score_max_abs"]
            )
    failed = [check for check in checks if check["status"] != "passed"]
    return {
        "profile": candidate["id"],
        "task": task,
        "policy": policy,
        "status": "passed" if not failed else "failed",
        "failed_checks": len(failed),
        "failures": [check["id"] for check in failed][:50],
        "checks": checks,
        "reference_status": "proposed",
        "qualification_status": "unapproved",
    }


def render_markdown(report: dict) -> str:
    worst: dict[str, dict] = {}
    for check in report["checks"]:
        if "max_abs" not in check:
            continue
        parts = check["id"].split("/")
        key = f"{parts[0]}/{parts[-1]}"
        if key not in worst or check["max_abs"] > worst[key]["max_abs"]:
            worst[key] = check
    exact = [check for check in report["checks"] if "max_abs" not in check]
    lines = [
        f"# Development parity: {report['profile']}",
        "",
        f"Status **{report['status']}**, {len(report['checks'])} checks, {report['failed_checks']} failed. "
        "References are proposed; bounds are the candidate's predeclared policy (issue #144).",
        "",
        "| Comparison | Largest difference | Bound |",
        "| --- | ---: | ---: |",
        *(f"| {key} | {check['max_abs']:.3e} | {check['bound']:g} |" for key, check in sorted(worst.items())),
        "",
        f"Exact checks: {sum(c['status'] == 'passed' for c in exact)} of {len(exact)} passed.",
        "",
    ]
    return "\n".join(lines)


def build_stages(
    recipe: dict,
    *,
    repo_root: Path,
    snapshot: Path,
    reference_python: Path,
    threads: int = 2,
    timeout: float = 3600,
    distribution: bool = False,
) -> list[Stage]:
    from model_pipeline.onboarding_candidate import candidate_stages, code_inputs, source_reader

    if distribution:
        raise PipelineError("Qwen3 Embedding distribution is not implemented by this plugin")
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise PipelineError("timeout must be finite, positive, and at most 86400 seconds")
    root, snapshot = repo_root.resolve(), snapshot.resolve()
    manifest, candidate, contract = _profile(recipe, root)
    corpus = repository_input(root, recipe["reference_corpus"])
    scripts = ("qwen3_graph.py", "export_qwen3_onnx.py", "release_qwen3_onnx.py", "verify_qwen3_onnx.py")
    _, initial, shared = candidate_stages(
        recipe,
        repo_root=root,
        snapshot=snapshot,
        reference_python=reference_python,
        threads=threads,
        timeout=timeout,
        source_files=(
            *(f"{DIRECTORY}/{name}" for name in (*scripts, "run_qwen3_reference.py")),
            "model-pipeline/src/model_pipeline/onboarding_qwen3_embedding.py",
        ),
    )
    python = Path(os.path.abspath(reference_python))
    model = contract["model"]
    env = {
        "PYTHONPATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "DOTNET_PROCESSOR_COUNT": str(threads),
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
    }
    parameters = {"plugin": PLUGIN, "candidate": candidate, "threads": threads, "timeout": timeout}

    def guard(out: Path) -> None:
        guard_output(
            out,
            (snapshot, root / "model-pipeline/protected", corpus, manifest),
            "stage output overlaps original source or protected evidence",
        )

    def script(out: Path, name: str, args: list[str], log: str = "command.log") -> dict:
        guard(out)
        return run_command(
            [str(python), str(root / DIRECTORY / name), *args],
            cwd=root / DIRECTORY,
            log_path=out / log,
            env=env,
            timeout=timeout,
        )

    def reference(out: Path, deps: dict) -> dict:
        script(
            out,
            "run_qwen3_reference.py",
            ["--model", model, "--model-dir", str(snapshot), "--corpus", str(corpus), "--out", str(out)],
        )
        record = json.loads((out / "reference.json").read_text(encoding="utf-8"))
        return {"cases": len(record["cases"]), "reference_status": "proposed"}

    def build(out: Path, deps: dict) -> dict:
        export, release = out / "export", out / "release"
        script(
            out,
            "export_qwen3_onnx.py",
            ["--model", model, "--model-dir", str(snapshot), "--out", str(export), "--threads", str(threads)],
            "export.log",
        )
        hosted = contract["hosting"]["files"]
        script(
            out,
            "release_qwen3_onnx.py",
            [
                "--model",
                model,
                "--model-dir",
                str(snapshot),
                "--export",
                str(export / "model.onnx"),
                "--out",
                str(release),
                "--expect-sha256",
                hosted["model.onnx"][1],
                "--expect-data-sha256",
                hosted["model.onnx.data"][1],
            ],
            "release.log",
        )
        # The unstripped export's 2.4 GB external data is not evidence; the released copy is.
        (export / "model.onnx.data").unlink()
        released = {
            name: [(release / name).stat().st_size, fixtures.sha256_file(release / name)] for name in sorted(hosted)
        }
        match = released == {name: list(value) for name, value in hosted.items()}
        export_record = json.loads((export / "export.json").read_text(encoding="utf-8"))
        write_json(
            out / "build.json",
            {
                "released_graph": released,
                "hosting": candidate["behavior"]["hosting"],
                "hosted_bytes_match": match,
                "exporter": export_record.get("exporter"),
                "opset": export_record.get("opset"),
            },
        )
        if not match:
            raise PipelineError(f"released graph differs from the hosted files: {released}")
        return {"hosted_bytes_match": True, "exporter": export_record.get("exporter")}

    def onnx(out: Path, deps: dict) -> dict:
        script(
            out,
            "verify_qwen3_onnx.py",
            [
                "--model",
                model,
                "--model-dir",
                str(snapshot),
                "--graph",
                str(deps["build"] / "release" / "model.onnx"),
                "--corpus",
                str(corpus),
                "--out",
                str(out / "verify.json"),
                "--threads",
                str(threads),
            ],
        )
        return {"verification": "written"}

    def dotnet(out: Path, deps: dict) -> dict:
        guard(out)
        project = root / TOOL
        if not project.is_file():
            raise PipelineError(f"Qwen3 Embedding .NET fidelity tool is missing: {TOOL}")
        return run_command(
            [
                shutil.which("dotnet") or "dotnet",
                "run",
                "--project",
                str(project),
                "-c",
                "Release",
                "--",
                "--profile",
                str(manifest),
                "--release",
                str(deps["build"] / "release"),
                "--reference",
                str(deps["reference"] / "reference.json"),
                "--out",
                str(out),
                "--threads",
                str(threads),
            ],
            cwd=root,
            log_path=out / "command.log",
            env=env,
            timeout=timeout,
        )

    def evaluation(out: Path, deps: dict) -> dict:
        guard(out)

        def load(stage: str, name: str) -> dict:
            return json.loads((deps[stage] / name).read_text(encoding="utf-8"))

        report = evaluate(
            candidate,
            load("reference", "reference.json"),
            load("build", "build.json"),
            load("onnx", "verify.json"),
            load("dotnet", "dotnet.json"),
        )
        write_json(out / "report.json", report)
        (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
        if report["status"] != "passed":
            raise PipelineError(
                f"Qwen3 Embedding development parity failed: {'; '.join(report['failures'])}; see {out / 'report.json'}"
            )
        return {"status": report["status"], "qualification_status": "unapproved", "reference_status": "proposed"}

    def review(out: Path, deps: dict) -> dict:
        guard(out)
        report = json.loads((deps["evaluate"] / "report.json").read_text(encoding="utf-8"))
        if report["status"] != "passed":
            raise PipelineError("review requires complete passed development parity")
        for name in ("report.json", "report.md"):
            shutil.copyfile(deps["evaluate"] / name, out / name)
        document = {
            "profile": candidate["id"],
            "task": candidate["task"],
            "development_parity": {k: report[k] for k in ("status", "failed_checks", "policy")},
            "checks": len(report["checks"]),
            "build": json.loads((deps["build"] / "build.json").read_text(encoding="utf-8")),
            "capabilities": candidate["behavior"]["capabilities"],
            "candidate_commit": fixtures.git_head(root),
            "source_lock": json.loads((deps["assets"] / "source-lock.json").read_text(encoding="utf-8")),
            "support_status": "experimental",
            "release_qualification": "unapproved",
            "reference_status": "proposed",
            "protected_fixture_adoption": False,
            "distribution": "not-run",
        }
        write_json(out / "review.json", document)
        lines = [
            f"# Candidate review: {candidate['id']}",
            "",
            f"Development parity passed ({len(report['checks'])} checks). The released graph is byte-identical to "
            f"{document['build']['hosting']['repo']} at {document['build']['hosting']['revision']}. References are "
            "proposed; bounds are the candidate's predeclared policy. Protected gates and release qualification are "
            "unapproved; distribution is not run.",
            "",
            "Capabilities: " + ", ".join(f"{k} {v}" for k, v in candidate["behavior"]["capabilities"].items()) + ".",
            "",
        ]
        (out / "review.md").write_text("\n".join(lines), encoding="utf-8")
        return {"development_parity": "passed", "release_qualification": "unapproved"}

    def stage(name: str, dependencies: tuple[str, ...], execute, inputs: dict | None = None) -> Stage:
        return Stage(name, dependencies, {**shared, **(inputs or {})}, parameters, execute)

    dotnet_inputs = code_inputs(
        root,
        files=(
            "global.json",
            "ai-dotnet/Directory.Build.props",
            "ai-dotnet/Directory.Packages.props",
        ),
        directories=tuple(
            relative
            for relative in (
                "ai-dotnet/tools/Starlings.Qwen3Embedding.Fidelity.Tool",
                "ai-dotnet/src/Starlings.Models.Qwen3Embedding",
                "ai-dotnet/src/Starlings.Runtime.Onnx",
                "ai-dotnet/src/Starlings.Acquisition",
            )
            if (root / relative).is_dir()
        ),
    )
    assets_stage, environment_stage = initial
    return [
        assets_stage,
        environment_stage,
        source_reader(
            stage("reference", ("assets", "environment"), reference, {"corpus": corpus}), candidate, snapshot
        ),
        source_reader(stage("build", ("assets", "environment"), build), candidate, snapshot),
        source_reader(stage("onnx", ("build", "environment"), onnx, {"corpus": corpus}), candidate, snapshot),
        stage("dotnet", ("reference", "build", "environment"), dotnet, dotnet_inputs),
        stage("evaluate", ("reference", "build", "onnx", "dotnet"), evaluation),
        replace(stage("review", ("assets", "build", "evaluate"), review)),
    ]

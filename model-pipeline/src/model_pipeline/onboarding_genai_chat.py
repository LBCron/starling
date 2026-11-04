"""Proposed data-driven GenAI chat onboarding with original/Python GenAI/owned .NET parity.

The stages follow onboarding_generation.py. That module stays as it is so the Qwen route's stage keys and evidence do
not move; the shared evaluator (evaluate_generation_candidate.py) is reused unchanged.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from model_pipeline import fixtures
from model_pipeline.evaluate_generation_candidate import POLICY, evaluate
from model_pipeline.onboarding_candidate import guard_output, repository_input, write_json
from model_pipeline.onboarding_runner import PipelineError, Stage, run_command

PLUGIN = "genai-chat-cpu-v1"
# The route's .NET projects are source only, so there is no package to take to a clean consumer.
SHARED_DISTRIBUTION = False
REFERENCE = "model-pipeline/reference/genai-chat/run_reference.py"
BUILDER = "model-pipeline/reference/genai-chat/build_bundle.py"
BACKEND = "model-pipeline/reference/genai-chat/run_genai.py"
FIDELITY = "ai-dotnet/tools/Starlings.GenAIChat.Fidelity/Starlings.GenAIChat.Fidelity.csproj"
# The route's allow-list as data (docs/design/generic-genai-chat-route.md, option C). Adding a repository means adding
# its entry there, a template port and a loader. A candidate's behavior must equal its entry field for field.
CONTRACTS_PATH = Path(__file__).resolve().parents[2] / "routes" / PLUGIN / "contracts.json"
CONTRACTS_FORMAT = "starlings.genai-chat-route-contracts"
TOKENIZER_PROFILES = {"phi4-bpe-chat"}
_BEHAVIOR_FIELDS = {
    "architecture": str,
    "model_type": str,
    "config": dict,
    "vocab_size": int,
    "eos_token_ids": list,
    "add_bos_token": bool,
    "pad_token_id_overlay": int,
    "pad_overlay_reason": str,
    "tokenizer_profile": str,
    "template": str,
    "template_sha256": str,
    "max_new_tokens": int,
    "max_context_tokens": int,
    "cache": str,
    "tools": str,
    "thinking": str,
    "multimodal": str,
    "long_context": str,
}
_CONFIG_FIELDS = {
    "hidden_size": int,
    "num_hidden_layers": int,
    "num_attention_heads": int,
    "num_key_value_heads": int,
    "head_dim": int,
    "intermediate_size": int,
    "max_position_embeddings": int,
    "original_max_position_embeddings": int,
    "partial_rotary_factor": float,
    "rope_type": str,
    "tie_word_embeddings": bool,
}
_STATUSES = {"supported", "unsupported", "experimental", "untested"}
EXPECTED_RUNTIME = {"backend": "ortgenai", "version": "0.17.1", "provider": "CPUExecutionProvider", "precision": "fp32"}
REQUIRED_EVIDENCE = [
    "prompt_text",
    "prompt_ids",
    "initial_logits",
    "output_ids",
    "finish_reason",
    "stream",
    "usage",
    "cancellation",
    "cache",
    "isolation",
]


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise PipelineError(f"GenAI chat route contracts: duplicate keys {duplicates}")
    return dict(pairs)


def validate_contracts(document: object) -> dict[str, dict]:
    """Refuse a table with another format, a missing or unknown field, or a value of the wrong type."""
    if not isinstance(document, dict) or set(document) != {"format", "format_version", "route", "contracts"}:
        raise PipelineError("GenAI chat route contracts: expected exactly format, format_version, route and contracts")
    version = document["format_version"]
    if (document["format"], type(version), version, document["route"]) != (CONTRACTS_FORMAT, int, 1, PLUGIN):
        raise PipelineError(f"GenAI chat route contracts: expected {CONTRACTS_FORMAT} version 1 for {PLUGIN}")
    contracts = document["contracts"]
    if not isinstance(contracts, dict) or not contracts:
        raise PipelineError("GenAI chat route contracts: contracts must be a non-empty object keyed by repository")
    for repo, spec in contracts.items():
        where = f"GenAI chat route contract {repo!r}"
        if repo.count("/") != 1 or not all(repo.split("/")):
            raise PipelineError(f"{where}: the key must be a Hugging Face repository id <owner>/<name>")
        if not isinstance(spec, dict) or set(spec) != {"id", "revision", "behavior"}:
            raise PipelineError(f"{where}: fields must be id, revision and behavior")
        revision = spec["revision"]
        if not isinstance(revision, str) or len(revision) != 40 or set(revision) - set("0123456789abcdef"):
            raise PipelineError(f"{where}: revision must be a full commit SHA")
        if not isinstance(spec["id"], str) or not spec["id"].endswith("/ortgenai-fp32-cpu"):
            raise PipelineError(f"{where}: id must name an ortgenai-fp32-cpu profile")
        behavior = spec["behavior"]
        for section, fields in (("behavior", _BEHAVIOR_FIELDS), ("config", _CONFIG_FIELDS)):
            value = behavior if section == "behavior" else behavior.get("config")
            if not isinstance(value, dict) or set(value) != set(fields):
                raise PipelineError(f"{where}: {section} fields must be {sorted(fields)}")
            for key, kind in fields.items():
                # bool is an int subclass: an integer field must not accept true or false.
                if type(value[key]) is not kind:
                    raise PipelineError(f"{where}: {key} must be a {kind.__name__}")
        vocabulary, eos = behavior["vocab_size"], behavior["eos_token_ids"]
        if not eos or any(type(token) is not int or not 0 <= token < vocabulary for token in eos):
            raise PipelineError(f"{where}: eos_token_ids must be ids inside the vocabulary")
        pad = behavior["pad_token_id_overlay"]
        if not 0 <= pad < vocabulary or pad in eos:
            raise PipelineError(f"{where}: the pad overlay must be a row inside the vocabulary, outside the EOS ids")
        if behavior["tokenizer_profile"] not in TOKENIZER_PROFILES:
            raise PipelineError(f"{where}: tokenizer_profile must be one of {sorted(TOKENIZER_PROFILES)}")
        digest = behavior["template_sha256"]
        if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise PipelineError(f"{where}: template_sha256 must be a lowercase SHA-256")
        if any(behavior[key] not in _STATUSES for key in ("tools", "thinking", "multimodal", "long_context")):
            raise PipelineError(f"{where}: every capability needs an explicit status {sorted(_STATUSES)}")
        if behavior["max_context_tokens"] > behavior["config"]["original_max_position_embeddings"]:
            raise PipelineError(f"{where}: max_context_tokens must stay inside the short rope factors")
    return contracts


def load_contracts(path: Path = CONTRACTS_PATH) -> dict[str, dict]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (OSError, ValueError) as exc:
        raise PipelineError(f"cannot read GenAI chat route contracts: {exc}") from exc
    return validate_contracts(document)


def _profile(recipe: dict, root: Path) -> tuple[Path, dict]:
    from model_pipeline.onboarding_candidate import load_candidate

    if recipe.get("plugin") != PLUGIN or recipe.get("recipe_version") != 1:
        raise PipelineError(f"expected recipe_version 1 and plugin {PLUGIN}")
    path = repository_input(root, recipe["manifest"])
    candidate = load_candidate(path)
    try:
        repository = candidate["origin"]["repo"]
        expected = load_contracts()[repository]
        if (
            candidate["format_version"] != 1
            or candidate["task"] != "generation"
            or candidate["id"] != expected["id"]
            or candidate["origin"]["revision"] != expected["revision"]
            or candidate["runtime"] != EXPECTED_RUNTIME
            or json.dumps(candidate["behavior"], sort_keys=True) != json.dumps(expected["behavior"], sort_keys=True)
            or candidate["evaluation"]["status"] != "proposed"
            or any(candidate["evaluation"]["policy"].get(key) != value for key, value in POLICY.items())
            or candidate["evaluation"]["policy"].get("required_evidence") != REQUIRED_EVIDENCE
        ):
            raise PipelineError("unsupported generation contract: exact pinned text-only CPU profiles required")
    except (KeyError, TypeError) as exc:
        raise PipelineError(f"incomplete or unsupported generation candidate: {exc}") from exc
    return path, candidate


def describe_plan(recipe: dict, *, repo_root: Path) -> dict:
    _, candidate = _profile(recipe, repo_root.resolve())
    return {
        "plugin": PLUGIN,
        "recipe": recipe["id"],
        "profile": candidate["id"],
        "revision": candidate["origin"]["revision"],
        "asset_count": len(candidate["origin"]["files"]),
        "stages": ["assets", "environment", "reference", "bundle", "backend", "dotnet", "evaluate", "review"],
        "required_gates": ["development-parity"],
        "optional_stages": [],
        "reference_status": "proposed; separate independent adoption required",
        "release_status": "unapproved; protected G1/G2 not awarded",
        "dependencies": {
            "python_packages": [
                "torch",
                "transformers",
                "tokenizers",
                "numpy",
                "safetensors",
                "onnxruntime-genai",
                "onnxruntime",
                "onnx",
                "onnx-ir",
            ],
            "dotnet_sdk": "global.json",
        },
    }


def build_stages(
    recipe: dict,
    *,
    repo_root: Path,
    snapshot: Path,
    reference_python: Path,
    threads: int = 2,
    timeout: float = 1800,
    distribution: bool = False,
) -> list[Stage]:
    from model_pipeline.onboarding_candidate import candidate_stages, code_inputs, source_reader

    if distribution:
        raise PipelineError("this route has no distribution stage: its .NET projects are source only")
    if type(threads) is not int or not 1 <= threads <= 2:
        raise PipelineError("GenAI chat CPU profiles permit one or two threads")
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise PipelineError("timeout must be finite, positive, and at most 86400 seconds")
    root, snapshot = repo_root.resolve(), snapshot.resolve()
    manifest_path, candidate = _profile(recipe, root)
    corpus = repository_input(root, recipe["reference_corpus"])
    _, stages, shared_inputs = candidate_stages(
        recipe,
        repo_root=root,
        snapshot=snapshot,
        reference_python=reference_python,
        threads=threads,
        timeout=timeout,
        source_files=(
            REFERENCE,
            BUILDER,
            BACKEND,
            "model-pipeline/reference/genai-chat/common.py",
            "model-pipeline/src/model_pipeline/onboarding_genai_chat.py",
            "model-pipeline/routes/genai-chat-cpu-v1/contracts.json",
            "model-pipeline/src/model_pipeline/evaluate_generation_candidate.py",
        ),
    )
    python = Path(os.path.abspath(reference_python))
    dotnet = shutil.which("dotnet") or "dotnet"
    env = {
        "PYTHONPATH": str(root / "model-pipeline/src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DOTNET_PROCESSOR_COUNT": str(threads),
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
    }

    def guard(out: Path) -> None:
        guard_output(
            out,
            (
                snapshot,
                root / "model-pipeline/protected",
                corpus,
                manifest_path,
            ),
            "stage output overlaps original source or protected evidence",
        )

    def python_run(out: Path, script: str, args: list[str]) -> dict:
        guard(out)
        return run_command(
            [str(python), str(root / script), "--candidate", str(manifest_path), *args, "--out", str(out)],
            cwd=root,
            log_path=out / "command.log",
            env=env,
            timeout=timeout,
        )

    def reference(out: Path, deps: dict) -> dict:
        return python_run(
            out, REFERENCE, ["--snapshot", str(snapshot), "--corpus", str(corpus), "--threads", str(threads)]
        )

    def bundle(out: Path, deps: dict) -> dict:
        return python_run(out, BUILDER, ["--snapshot", str(snapshot)])

    def backend(out: Path, deps: dict) -> dict:
        return python_run(
            out,
            BACKEND,
            [
                "--snapshot",
                str(snapshot),
                "--bundle",
                str(deps["bundle"]),
                "--corpus",
                str(corpus),
                "--threads",
                str(threads),
            ],
        )

    def dotnet_run(out: Path, deps: dict) -> dict:
        guard(out)
        # Build intermediates and the published tool (including native runtime libraries) are
        # transient: they must not be hashed into run.json, rehashed on resume, or proposed.
        with tempfile.TemporaryDirectory(prefix="dotnet-build-", dir=out) as build:
            published = Path(build) / "published"
            build_result = run_command(
                [
                    dotnet,
                    "publish",
                    str(root / FIDELITY),
                    "-c",
                    "Release",
                    "-m:1",
                    "--disable-build-servers",
                    "--artifacts-path",
                    build,
                    "--output",
                    str(published),
                    "-p:RestoreLockedMode=true",
                ],
                cwd=root,
                log_path=out / "build.log",
                env=env,
                timeout=timeout,
            )
            assembly = published / "Starlings.GenAIChat.Fidelity.dll"
            if not assembly.is_file():
                raise PipelineError("generation fidelity publish produced no tool assembly")
            result = run_command(
                [
                    dotnet,
                    str(assembly),
                    "--candidate",
                    str(manifest_path),
                    "--bundle",
                    str(deps["bundle"]),
                    "--corpus",
                    str(corpus),
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
            return {**result, "build": build_result}

    def evaluation(out: Path, deps: dict) -> dict:
        guard(out)
        report = evaluate(candidate, corpus, deps["reference"], deps["backend"], deps["dotnet"])
        write_json(out / "report.json", report)
        if report["status"] != "passed":
            raise PipelineError("development generation parity failed: " + "; ".join(report["failures"]))
        return report

    def review(out: Path, deps: dict) -> dict:
        guard(out)
        report = json.loads((deps["evaluate"] / "report.json").read_text(encoding="utf-8"))
        if report["status"] != "passed":
            raise PipelineError("review requires complete passed development parity")
        shutil.copyfile(deps["evaluate"] / "report.json", out / "report.json")
        data = {
            "profile": candidate["id"],
            "task": candidate["task"],
            "development_parity": report,
            "policy": candidate["evaluation"]["policy"],
            "candidate_commit": fixtures.git_head(root),
            "source_lock": json.loads((deps["assets"] / "source-lock.json").read_text(encoding="utf-8")),
            "support_status": "experimental",
            "support_status_changed": False,
            "release_qualification": "unapproved",
            "protected_release_qualification": "unapproved",
            "reference_status": "proposed",
            "protected_fixture_adoption": False,
            "distribution": "not-run",
        }
        write_json(out / "review.json", data)
        (out / "review.md").write_text(
            f"# Candidate review: {candidate['id']}\n\n"
            "Development task parity passed. References/policies remain proposed. "
            "Protected G1/G2 and full release qualification are unapproved; distribution is not run.\n",
            encoding="utf-8",
        )
        return {"development_parity": "passed", "release_qualification": "unapproved"}

    def stage(name: str, dependencies: tuple[str, ...], execute, *, extra_inputs=None) -> Stage:
        return Stage(
            name,
            dependencies,
            {**shared_inputs, **(extra_inputs or {})},
            {"plugin": PLUGIN, "profile": candidate, "threads": threads, "timeout": timeout},
            execute,
        )

    dotnet_inputs = code_inputs(
        root,
        files=(
            "global.json",
            "Directory.Build.props",
            "ai-dotnet/Directory.Build.props",
            "ai-dotnet/Directory.Build.targets",
            "ai-dotnet/Directory.Packages.props",
            *(str(path.relative_to(root)) for path in (root / "Directory.Build.targets",) if path.is_file()),
        ),
        directories=(
            "ai-dotnet/src/Starlings.Models.Phi4",
            "ai-dotnet/src/Starlings.Models.Phi4.OnnxGenAI",
            "ai-dotnet/src/Starlings.Generation.Abstractions",
            "ai-dotnet/src/Starlings.Generation.OnnxGenAI",
            "ai-dotnet/src/Starlings.Acquisition",
            "ai-dotnet/src/Shared/Threading",
            "ai-dotnet/tools/Shared",
            "ai-dotnet/tools/Starlings.GenAIChat.Fidelity",
        ),
    )
    stages.extend(
        [
            stage("reference", ("assets", "environment"), reference, extra_inputs={"corpus": corpus}),
            stage("bundle", ("assets", "environment"), bundle),
            stage("backend", ("bundle", "environment"), backend, extra_inputs={"corpus": corpus}),
            stage("dotnet", ("bundle", "environment"), dotnet_run, extra_inputs={"corpus": corpus, **dotnet_inputs}),
            stage("evaluate", ("reference", "backend", "dotnet"), evaluation, extra_inputs={"corpus": corpus}),
            dataclasses.replace(
                stage("review", ("assets", "evaluate"), review),
                parameters={
                    "plugin": PLUGIN,
                    "profile": candidate,
                    "threads": threads,
                    "timeout": timeout,
                    "candidate_commit": fixtures.git_head(root),
                },
            ),
        ]
    )
    return [
        source_reader(stage, candidate, snapshot) if stage.name in {"reference", "bundle", "backend"} else stage
        for stage in stages
    ]

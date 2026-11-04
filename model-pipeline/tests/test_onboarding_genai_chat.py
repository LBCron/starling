from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from model_pipeline import onboarding_candidate, onboarding_genai_chat
from model_pipeline.evaluate_generation_candidate import POLICY
from model_pipeline.onboarding_genai_chat import (
    EXPECTED_RUNTIME,
    PLUGIN,
    REQUIRED_EVIDENCE,
    describe_plan,
    load_contracts,
    validate_contracts,
)
from model_pipeline.onboarding_plugins import get_plugin
from model_pipeline.onboarding_runner import PipelineError, Stage

REPOSITORY = "microsoft/Phi-4-mini-instruct"
MODEL_PIPELINE = Path(__file__).resolve().parents[1]


def table() -> dict:
    return copy.deepcopy(json.loads((MODEL_PIPELINE / "routes" / PLUGIN / "contracts.json").read_text("utf-8")))


def profile(tmp_path):
    contract = load_contracts()[REPOSITORY]
    candidate = {
        "format_version": 1,
        "id": contract["id"],
        "task": "generation",
        "origin": {
            "repo": REPOSITORY,
            "revision": contract["revision"],
            "files": [{"path": name, "size": 1, "sha256": "a" * 64} for name in ("config.json", "tokenizer.json")],
        },
        "runtime": EXPECTED_RUNTIME,
        "behavior": contract["behavior"],
        "evaluation": {"status": "proposed", "policy": {**POLICY, "required_evidence": REQUIRED_EVIDENCE}},
    }
    recipe = {"recipe_version": 1, "id": "test", "plugin": PLUGIN, "manifest": "profile.json"}
    (tmp_path / "profile.json").write_text(json.dumps(candidate))
    return recipe, candidate


def test_committed_candidate_equals_the_contract_and_pins_no_remote_code():
    candidate = json.loads((MODEL_PIPELINE / "candidates/phi4-mini/profile.json").read_text("utf-8"))
    contract = load_contracts()[REPOSITORY]
    assert candidate["behavior"] == contract["behavior"]
    assert candidate["origin"]["revision"] == contract["revision"]
    assert candidate["evaluation"]["policy"]["logits_max_abs"] == POLICY["logits_max_abs"]
    paths = [record["path"] for record in candidate["origin"]["files"]]
    assert not [path for path in paths if path.endswith(".py")]
    assert {"LICENSE", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"} <= set(paths)


def test_plugin_is_registered_and_admits_the_pinned_profile(tmp_path):
    recipe, candidate = profile(tmp_path)
    report = get_plugin(recipe).describe_plan(recipe, repo_root=tmp_path)
    assert report["profile"] == candidate["id"]
    assert report["release_status"].startswith("unapproved")
    assert report["stages"] == [
        "assets",
        "environment",
        "reference",
        "bundle",
        "backend",
        "dotnet",
        "evaluate",
        "review",
    ]


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("runtime", "precision", "fp16"),
        ("behavior", "vocab_size", 200000),
        ("behavior", "eos_token_ids", [199999]),
        ("behavior", "tools", "experimental"),
        ("behavior", "pad_token_id_overlay", 199999),
        ("origin", "revision", "a" * 40),
    ],
)
def test_rejects_altered_contract(tmp_path, section, key, value):
    recipe, candidate = profile(tmp_path)
    candidate = copy.deepcopy(candidate)
    candidate[section][key] = value
    (tmp_path / "profile.json").write_text(json.dumps(candidate))
    with pytest.raises(PipelineError):
        describe_plan(recipe, repo_root=tmp_path)


def test_rejects_a_changed_rope_geometry_and_another_repository(tmp_path):
    recipe, candidate = profile(tmp_path)
    altered = copy.deepcopy(candidate)
    altered["behavior"]["config"]["partial_rotary_factor"] = 1.0
    (tmp_path / "profile.json").write_text(json.dumps(altered))
    with pytest.raises(PipelineError, match="unsupported generation contract"):
        describe_plan(recipe, repo_root=tmp_path)
    altered = copy.deepcopy(candidate)
    altered["origin"]["repo"] = "microsoft/phi-4"
    (tmp_path / "profile.json").write_text(json.dumps(altered))
    with pytest.raises(PipelineError):
        describe_plan(recipe, repo_root=tmp_path)


def _entry(document: dict) -> dict:
    return document["contracts"][REPOSITORY]


def _missing_status(document: dict) -> None:
    del _entry(document)["behavior"]["long_context"]


def _string_status(document: dict) -> None:
    _entry(document)["behavior"]["tools"] = "maybe"


def _short_revision(document: dict) -> None:
    _entry(document)["revision"] = "cfbefacb"


def _branch_revision(document: dict) -> None:
    _entry(document)["revision"] = "main"


def _boolean_integer(document: dict) -> None:
    _entry(document)["behavior"]["vocab_size"] = True


def _unknown_field(document: dict) -> None:
    _entry(document)["behavior"]["config"]["extra"] = 1


def _other_tokenizer(document: dict) -> None:
    _entry(document)["behavior"]["tokenizer_profile"] = "qwen-bpe-chat"


def _pad_outside(document: dict) -> None:
    _entry(document)["behavior"]["pad_token_id_overlay"] = 200064


def _long_context(document: dict) -> None:
    _entry(document)["behavior"]["max_context_tokens"] = 8192


def _other_route(document: dict) -> None:
    document["route"] = "qwen-genai-cpu-v1"


@pytest.mark.parametrize(
    "change",
    [
        _missing_status,
        _string_status,
        _short_revision,
        _branch_revision,
        _boolean_integer,
        _unknown_field,
        _other_tokenizer,
        _pad_outside,
        _long_context,
        _other_route,
    ],
)
def test_table_validation_refuses(change):
    document = table()
    change(document)
    with pytest.raises(PipelineError):
        validate_contracts(document)


def test_table_refuses_duplicate_keys(tmp_path):
    path = tmp_path / "contracts.json"
    path.write_text('{"format": "a", "format": "b"}')
    with pytest.raises(PipelineError, match="duplicate"):
        load_contracts(path)


def script(monkeypatch, name: str):
    directory = MODEL_PIPELINE / "reference/genai-chat"
    monkeypatch.syspath_prepend(str(directory))
    specification = importlib.util.spec_from_file_location(f"genai_chat_{name}", directory / f"{name}.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def original_config(behavior: dict) -> dict:
    geometry = {k: v for k, v in behavior["config"].items() if k not in {"head_dim", "rope_type"}}
    return {
        "architectures": [behavior["architecture"]],
        "model_type": behavior["model_type"],
        "torch_dtype": "bfloat16",
        "auto_map": {"AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM"},
        "rope_scaling": {"type": behavior["config"]["rope_type"]},
        **geometry,
    }


def test_float32_overlay_drops_auto_map_and_changes_only_dtype_fields(monkeypatch):
    module = script(monkeypatch, "build_bundle")
    behavior = load_contracts()[REPOSITORY]["behavior"]
    original = original_config(behavior)
    before = copy.deepcopy(original)
    result = module.float32_config(original, behavior)
    assert original == before
    expected = {key: value for key, value in before.items() if key != "auto_map"}
    assert result == {**expected, "dtype": "float32", "torch_dtype": "float32"}
    with pytest.raises(ValueError, match="flat"):
        module.float32_config({"model_type": "phi3", "text_config": {}}, behavior)
    with pytest.raises(ValueError, match="flat"):
        module.float32_config({"model_type": "qwen3"}, behavior)


@pytest.mark.parametrize(
    "key,value",
    [
        ("architectures", ["Phi4MMForCausalLM"]),
        ("partial_rotary_factor", 1.0),
        ("original_max_position_embeddings", 8192),
        ("rope_scaling", {"type": "default"}),
        ("num_key_value_heads", 24),
    ],
)
def test_geometry_check_refuses_a_different_checkpoint(monkeypatch, key, value):
    module = script(monkeypatch, "build_bundle")
    behavior = load_contracts()[REPOSITORY]["behavior"]
    module.check_geometry(original_config(behavior), behavior)
    with pytest.raises(ValueError):
        module.check_geometry({**original_config(behavior), key: value}, behavior)


def test_verified_copy_rejects_corruption_and_never_changes_source(tmp_path, monkeypatch):
    module = script(monkeypatch, "build_bundle")
    behavior = load_contracts()[REPOSITORY]["behavior"]
    snapshot = tmp_path / "source"
    snapshot.mkdir()
    contents = {
        "config.json": json.dumps(original_config(behavior)).encode(),
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
    }
    for name, raw in contents.items():
        (snapshot / name).write_bytes(raw)
    candidate = {
        "behavior": behavior,
        "origin": {
            "files": [
                {"path": name, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                for name, raw in contents.items()
            ]
        },
    }
    module.verified_copy(candidate, snapshot, tmp_path / "scratch")
    assert (snapshot / "config.json").read_bytes() == contents["config.json"]
    copied = json.loads((tmp_path / "scratch/config.json").read_text())
    assert copied["torch_dtype"] == "float32" and "auto_map" not in copied
    (snapshot / "tokenizer.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="asset mismatch"):
        module.verified_copy(candidate, snapshot, tmp_path / "corrupt-scratch")


def test_corpus_row_refused_at_the_long_context_boundary(monkeypatch):
    module = script(monkeypatch, "common")
    candidate = {"behavior": load_contracts()[REPOSITORY]["behavior"]}
    module.check_context([1] * 4079, 16, candidate)
    with pytest.raises(ValueError, match="long-context"):
        module.check_context([1] * 4080, 16, candidate)
    with pytest.raises(ValueError, match="positive"):
        module.check_context([1], 0, candidate)


def test_dotnet_stage_runs_the_route_tool_and_declares_the_route_code(tmp_path, monkeypatch):
    root, snapshot = tmp_path / "repo", tmp_path / "snapshot"
    root.mkdir()
    snapshot.mkdir()
    recipe, candidate = profile(root)
    (root / "corpus.jsonl").write_text("")
    recipe["reference_corpus"] = "corpus.jsonl"
    declared = {}

    def code_inputs(_root, *, files=(), directories=()):
        declared["files"], declared["directories"] = files, directories
        return {}

    monkeypatch.setattr(
        onboarding_candidate,
        "candidate_stages",
        lambda *args, **kwargs: (
            None,
            [Stage(name, (), {}, {}, lambda out, deps: {}) for name in ("assets", "environment")],
            {},
        ),
    )
    monkeypatch.setattr(onboarding_candidate, "code_inputs", code_inputs)
    monkeypatch.setattr(onboarding_candidate, "source_reader", lambda stage, *args: stage)
    commands = []

    def command(argv, **kwargs):
        commands.append(argv)
        kwargs["log_path"].write_text("okay")
        if argv[1] == "publish":
            published = Path(argv[argv.index("--output") + 1])
            published.mkdir(parents=True)
            (published / "Starlings.GenAIChat.Fidelity.dll").write_bytes(b"fixture")
        return {"returncode": 0}

    monkeypatch.setattr(onboarding_genai_chat, "run_command", command)
    graph = {
        stage.name: stage
        for stage in onboarding_genai_chat.build_stages(
            recipe, repo_root=root, snapshot=snapshot, reference_python=Path(sys.executable), threads=2, timeout=11
        )
    }
    for directory in (
        "Starlings.Models.Phi4",
        "Starlings.Models.Phi4.OnnxGenAI",
        "Starlings.Acquisition",
        "Shared/Threading",
    ):
        assert f"ai-dotnet/src/{directory}" in declared["directories"]
    assert "ai-dotnet/tools/Shared" in declared["directories"]
    assert "ai-dotnet/tools/Starlings.GenAIChat.Fidelity" in declared["directories"]
    out = tmp_path / "dotnet"
    result = graph["dotnet"].execute(out, {"bundle": tmp_path / "bundle"})
    publish, run = commands
    assert publish[1] == "publish" and "-p:RestoreLockedMode=true" in publish
    assert publish[2].endswith("Starlings.GenAIChat.Fidelity.csproj")
    assert run[1].endswith("Starlings.GenAIChat.Fidelity.dll")
    assert result["build"] == {"returncode": 0}

    evaluate, assets, review_out = tmp_path / "evaluate", tmp_path / "assets", tmp_path / "review"
    evaluate.mkdir()
    assets.mkdir()
    review_out.mkdir()
    (evaluate / "report.json").write_text(json.dumps({"status": "failed", "failures": ["x"]}))
    (assets / "source-lock.json").write_text(json.dumps({"profile": candidate["id"]}))
    with pytest.raises(PipelineError, match="passed"):
        graph["review"].execute(review_out, {"evaluate": evaluate, "assets": assets})
    (evaluate / "report.json").write_text(json.dumps({"status": "passed", "failures": []}))
    graph["review"].execute(review_out, {"evaluate": evaluate, "assets": assets})
    review = json.loads((review_out / "review.json").read_text())
    assert review["support_status"] == "experimental" and review["support_status_changed"] is False
    assert review["release_qualification"] == "unapproved" and review["reference_status"] == "proposed"

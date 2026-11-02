"""Weight-free checks of the Qwen3 Embedding series route: contract allow-list, registration, plan and evaluator."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from model_pipeline import onboarding_plugins
from model_pipeline import onboarding_qwen3_embedding as q
from model_pipeline.onboarding_candidate import load_candidate
from model_pipeline.onboarding_recipe import load_recipe
from model_pipeline.onboarding_runner import PipelineError

REPO = Path(__file__).resolve().parents[2]
NAMES = {"Qwen/Qwen3-Embedding-0.6B": "qwen3-embedding-0.6b", "Qwen/Qwen3-Reranker-0.6B": "qwen3-reranker-0.6b"}


def profile(repo: str) -> dict:
    return json.loads((REPO / f"model-pipeline/candidates/{NAMES[repo]}/profile.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("repo", sorted(NAMES))
def test_shipped_candidates_pass_schema_and_exact_contract(repo):
    candidate = load_candidate(REPO / f"model-pipeline/candidates/{NAMES[repo]}/profile.json")
    contract = q.validate_candidate_behavior(candidate)
    assert candidate["id"] == contract["id"]
    assert set(candidate["behavior"]["capabilities"].values()) <= {
        "supported",
        "unsupported",
        "experimental",
        "untested",
    }
    assert candidate["behavior"]["hosting"]["repo"].startswith("Starlinghq/")


def test_plugin_is_registered():
    assert onboarding_plugins.PLUGINS[q.PLUGIN] == "model_pipeline.onboarding_qwen3_embedding"
    assert onboarding_plugins.uses_shared_distribution(q) is True


@pytest.mark.parametrize("name", ["qwen3-embedding-0.6b", "qwen3-reranker-0.6b"])
def test_recipes_plan_without_weights(name):
    recipe = load_recipe(REPO / f"model-pipeline/recipes/{name}-cpu.yaml", repo_root=REPO)
    plan = q.describe_plan(recipe, repo_root=REPO)
    assert plan["stages"] == q.STAGES
    assert plan["qualification_status"] == "unapproved"


@pytest.mark.parametrize(
    ("repo", "path", "value"),
    [
        ("Qwen/Qwen3-Embedding-0.6B", ("behavior", "pooling"), "mean"),
        ("Qwen/Qwen3-Embedding-0.6B", ("behavior", "prompts", "query"), q.QUERY_PROMPT + " "),
        ("Qwen/Qwen3-Embedding-0.6B", ("origin", "revision"), "0" * 40),
        ("Qwen/Qwen3-Embedding-0.6B", ("behavior", "hosting", "revision"), "main"),
        ("Qwen/Qwen3-Embedding-0.6B", ("behavior", "hosting", "repo"), "Starlinghq/other"),
        ("Qwen/Qwen3-Embedding-0.6B", ("evaluation", "policy", "embedding_max_abs"), 1e-3),
        ("Qwen/Qwen3-Reranker-0.6B", ("behavior", "yes_token_id"), 9454),
        ("Qwen/Qwen3-Reranker-0.6B", ("behavior", "suffix"), "<|im_end|>\n<|im_start|>assistant\n"),
        ("Qwen/Qwen3-Reranker-0.6B", ("task",), "embeddings"),
        ("Qwen/Qwen3-Reranker-0.6B", ("runtime", "precision"), "fp16"),
    ],
)
def test_near_misses_are_refused(repo, path, value):
    candidate = profile(repo)
    target = candidate
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(PipelineError):
        q.validate_candidate_behavior(candidate)


def test_hosted_graph_digest_and_unknown_repository_are_refused():
    candidate = profile("Qwen/Qwen3-Embedding-0.6B")
    candidate["behavior"]["hosting"]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(PipelineError, match="hosting.files"):
        q.validate_candidate_behavior(candidate)
    other = profile("Qwen/Qwen3-Embedding-0.6B")
    other["origin"]["repo"] = "Qwen/Qwen3-Embedding-4B"
    with pytest.raises(PipelineError, match="allow-listed"):
        q.validate_candidate_behavior(other)


def test_policy_respects_the_route_caps():
    for key, cap in q.CAPS.items():
        assert q.POLICY[key] <= cap


def _embedding_inputs(delta: float = 0.0, ids_change: bool = False):
    candidate = profile("Qwen/Qwen3-Embedding-0.6B")
    vector = [1.0] + [0.0] * 1023
    reference = {"cases": [{"id": "a", "ids": [1, 151643], "embedding": vector}]}
    shifted = [vector[0] - delta, delta] + [0.0] * 1022
    norm = math.sqrt(sum(v * v for v in shifted))
    shifted = [v / norm for v in shifted]
    dotnet = {
        "cases": [
            {
                "id": "a",
                "ids": [2, 151643] if ids_change else [1, 151643],
                "embedding_text": shifted,
                "embedding_ids": vector,
            }
        ],
        "card_cosines": [[0.76455, 0.14142], [0.13549, 0.59995]],
    }
    verify = {
        "model": "embedding",
        "hidden_max_abs": 1e-3,
        "cases": [{"id": "a", "hidden_max_abs": 1e-3}],
        "eos_last_all_inputs": True,
        "embedding_max_abs": 1e-7,
        "card_cosines": {
            "graph_max_abs_vs_published": 1e-7,
            "published": [[0.7645568251609802, 0.14142508804798126], [0.13549736142158508, 0.5999549627304077]],
        },
        "sentence_transformers_embedding_max_abs": 1e-7,
    }
    build = {"hosted_bytes_match": True, "released_graph": {}}
    return candidate, reference, build, verify, dotnet


def test_evaluator_passes_within_bounds_and_fails_outside():
    report = q.evaluate(*_embedding_inputs())
    assert report["status"] == "passed", report["failures"]
    assert report["failed_checks"] == 0
    assert q.render_markdown(report).startswith("# Development parity")
    report = q.evaluate(*_embedding_inputs(delta=1e-3))
    assert report["status"] == "failed"
    assert "dotnet-text/a/embedding_max_abs" in report["failures"]
    report = q.evaluate(*_embedding_inputs(ids_change=True))
    assert "dotnet-text/a/token_ids" in report["failures"]


def test_evaluator_reranker_scores():
    candidate = profile("Qwen/Qwen3-Reranker-0.6B")
    reference = {"prefix_ids": [1], "suffix_ids": [2], "cases": [{"id": "card-0-0", "ids": [1, 5, 2], "score": 3.0}]}
    dotnet = {
        "prefix_ids": [1],
        "suffix_ids": [2],
        "cases": [{"id": "card-0-0", "ids": [1, 5, 2], "score_text": 3.0005, "score_ids": 3.0}],
        "public_scores": [{"id": "card-0-0", "score": 3.0}],
    }
    verify = {
        "model": "reranker",
        "hidden_max_abs": 1e-3,
        "cases": [],
        "chat_template_matches_prefix_content_suffix": True,
        "score_max_abs": 1e-4,
        "sentence_transformers": {"max_abs": 1e-4},
    }
    build = {"hosted_bytes_match": True, "released_graph": {}}
    assert q.evaluate(candidate, reference, build, verify, dotnet)["status"] == "passed"
    bad = copy.deepcopy(dotnet)
    bad["cases"][0]["score_ids"] = 3.01
    assert "dotnet-ids/card-0-0/score_max_abs" in q.evaluate(candidate, reference, build, verify, bad)["failures"]
    with pytest.raises(PipelineError, match="other model"):
        q.evaluate(candidate, reference, build, {**verify, "model": "embedding"}, dotnet)

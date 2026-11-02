"""PyTorch fp32 reference of the `qwen3-embedding-onnx-cpu-v1` onboarding route (transformers 5.18.0, CPU, sdpa).

Writes `reference.json`: the cases the .NET stage reproduces, each run alone and without padding.

- `embedding`: the model card's two queries (prompt `query`) and two documents (prompt `none`), every text of the
  corpus with prompt `none`, the first eight non-empty corpus texts with prompt `query`, and one long input of about
  4,000 tokens built by cycling through the corpus. Records each case's token ids (the pinned tokenizer, which appends
  `<|endoftext|>`) and its embedding: the `AutoModel` `last_hidden_state` at the last token, L2-normalized (the card's
  transformers snippet).
- `reranker`: the card's four query-document pairs (default instruction), one pair per corpus text
  (`"What does this text say?"`), two pairs with a custom instruction, and one long pair of about 4,000 tokens.
  Records the template's prefix and suffix ids, each case's full input ids (`prefix + content + suffix`) and the score
  `logit[yes = 9693] - logit[no = 2152]` at the last position of `AutoModelForCausalLM` (full language model head).

The values are proposals for the route's evaluator, never protected fixtures.

Usage:
    python run_qwen3_reference.py --model embedding|reranker --model-dir <snapshot> --corpus <jsonl> --out <dir>
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from qwen3_graph import MAX_SEQUENCE, NO_TOKEN_ID, YES_TOKEN_ID
from verify_qwen3_onnx import (
    CARD_DOCUMENTS,
    CARD_QUERIES,
    CARD_TASK,
    QUERY_PROMPT,
    RERANK_PREFIX,
    RERANK_SUFFIX,
    corpus_texts,
    long_text,
)

FORMAT = "starlings-qwen3-embedding-reference/1"
LONG_TOKENS = 4000
RERANK_CORPUS_QUERY = "What does this text say?"
CUSTOM_INSTRUCTION = "Judge whether the document is written in the same language as the query"


def embedding_cases(tokenizer, corpus: list[tuple[str, str]]) -> list[dict]:
    cases = [{"id": f"card-query-{i}", "prompt": "query", "text": q} for i, q in enumerate(CARD_QUERIES)]
    cases += [{"id": f"card-doc-{i}", "prompt": "none", "text": d} for i, d in enumerate(CARD_DOCUMENTS)]
    cases += [{"id": f"corpus-{name}", "prompt": "none", "text": text} for name, text in corpus]
    nonempty = [(name, text) for name, text in corpus if text]
    cases += [{"id": f"query-{name}", "prompt": "query", "text": text} for name, text in nonempty[:8]]
    texts = [text for _, text in nonempty]
    cases.append({"id": f"long-{LONG_TOKENS}", "prompt": "none", "text": long_text(tokenizer, texts, LONG_TOKENS - 1)})
    return cases


def rerank_cases(tokenizer, corpus: list[tuple[str, str]], budget: int) -> list[dict]:
    cases = [
        {"id": f"card-{q}-{d}", "instruction": CARD_TASK, "query": CARD_QUERIES[q], "document": CARD_DOCUMENTS[d]}
        for q in range(2)
        for d in range(2)
    ]
    cases += [
        {"id": f"corpus-{name}", "instruction": CARD_TASK, "query": RERANK_CORPUS_QUERY, "document": text}
        for name, text in corpus
    ]
    cases += [
        {"id": "custom-0", "instruction": CUSTOM_INSTRUCTION, "query": CARD_QUERIES[0], "document": CARD_DOCUMENTS[0]},
        {"id": "custom-1", "instruction": CUSTOM_INSTRUCTION, "query": "Ünïcödé 你好", "document": ""},
    ]
    texts = [text for _, text in corpus if text]
    head = len(tokenizer(content(CARD_TASK, CARD_QUERIES[0], ""), add_special_tokens=False)["input_ids"])
    document = long_text(tokenizer, texts, min(LONG_TOKENS, budget) - head)
    cases.append(
        {"id": f"long-{LONG_TOKENS}", "instruction": CARD_TASK, "query": CARD_QUERIES[0], "document": document}
    )
    return cases


def content(instruction: str, query: str, document: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def run_embedding(model_dir: str, corpus: list[tuple[str, str]]) -> dict:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    cases = embedding_cases(tokenizer, corpus)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="sdpa").eval()
    for case in cases:
        prefix = QUERY_PROMPT if case["prompt"] == "query" else ""
        ids = tokenizer(prefix + case["text"], truncation=True, max_length=MAX_SEQUENCE)["input_ids"]
        with torch.inference_mode():
            hidden = model(input_ids=torch.tensor([ids])).last_hidden_state[0, -1]
        embedding = torch.nn.functional.normalize(hidden, p=2, dim=0).numpy().astype(np.float32)
        case.update(ids=ids, embedding=[float(x) for x in embedding], finite=bool(np.isfinite(embedding).all()))
        gc.collect()
    return {"query_prompt": QUERY_PROMPT, "cases": cases}


def run_reranker(model_dir: str, corpus: list[tuple[str, str]]) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prefix = tokenizer.encode(RERANK_PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(RERANK_SUFFIX, add_special_tokens=False)
    budget = MAX_SEQUENCE - len(prefix) - len(suffix)
    cases = rerank_cases(tokenizer, corpus, budget)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="sdpa").eval()
    for case in cases:
        text = content(case["instruction"], case["query"], case["document"])
        ids = (
            prefix + tokenizer(text, truncation=True, max_length=budget, add_special_tokens=False)["input_ids"] + suffix
        )
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids]), logits_to_keep=1).logits[0, -1]
        yes, no = float(logits[YES_TOKEN_ID]), float(logits[NO_TOKEN_ID])
        case.update(ids=ids, yes_logit=yes, no_logit=no, score=yes - no, finite=bool(np.isfinite([yes, no]).all()))
        gc.collect()
    return {"prefix_ids": prefix, "suffix_ids": suffix, "default_instruction": CARD_TASK, "cases": cases}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["embedding", "reranker"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    corpus = corpus_texts(args.corpus)
    result = (
        run_embedding(args.model_dir, corpus) if args.model == "embedding" else run_reranker(args.model_dir, corpus)
    )
    record = {
        "format": FORMAT,
        "model": args.model,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "attention": "sdpa",
        "dtype": "float32",
    } | result
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reference.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "cases": len(record["cases"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

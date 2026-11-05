"""Template reference of a genai-chat-cpu-v1 candidate: for each corpus case, the checkpoint's chat template rendered by
transformers apply_chat_template and the prompt ids of the checkpoint's tokenizer (add_special_tokens=False).

add_generation_prompt defaults to true; a case may set it. Writes <out>/template.jsonl and
<out>/environment-template.json. Nothing is written into protected/. The model itself does not run: no weights, no
remote code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True, help="snapshot with the tokenizer files")
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    template = json.loads((args.snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))["chat_template"]
    if hashlib.sha256(template.encode("utf-8")).hexdigest() != candidate["behavior"]["template_sha256"]:
        raise SystemExit("chat template differs from the contract's digest")
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    rows, seen = [], set()
    for path in args.corpus:
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line:
                continue
            case = json.loads(line)
            if case["id"] in seen:
                raise SystemExit(f"duplicate case id {case['id']}")
            seen.add(case["id"])
            text = tokenizer.apply_chat_template(
                case["messages"], tokenize=False, add_generation_prompt=case.get("add_generation_prompt", True)
            )
            rows.append({**case, "text": text, "prompt_ids": tokenizer.encode(text, add_special_tokens=False)})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "template.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    environment = {
        "stage": "template",
        "checkpoint": {"repo": candidate["origin"]["repo"], "revision": candidate["origin"]["revision"]},
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": candidate["behavior"]["template_sha256"],
        "tokenizer_sha256": sha256(args.snapshot / "tokenizer.json"),
        "tokenizer_config_sha256": sha256(args.snapshot / "tokenizer_config.json"),
        "corpora": [{"name": path.name, "sha256": sha256(path)} for path in args.corpus],
        "cases": len(rows),
        "python": platform.python_version(),
        "packages": {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__},
    }
    (args.out / "environment-template.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

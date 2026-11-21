"""Check everything the demo needs, and say exactly what is missing (P4.7 support).

The demo has five prerequisites and fails differently for each. Two of them produce
errors that do not name the real cause: a missing `.env` surfaces as "No teacher model
set" from deep inside the LLM client, and a stopped Ollama surfaces as a connection
error on the first claim rather than at startup. This checks all five in order and
prints the command that fixes whichever failed.

    python -m demo.doctor
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

OK, BAD, WARN = "  ok  ", "  FIX ", " warn "
problems: list[str] = []


def fail(label: str, what: str, how: str) -> None:
    print(f"{BAD}{label}\n       {what}\n       -> {how}")
    problems.append(label)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    print(f"\nchecking the demo in {root}\n")

    # 1. Python packages ---------------------------------------------------------
    missing = []
    for mod, pip_name in (
        ("gradio", "gradio"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("openai", "openai"),
    ):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        fail(
            "python packages",
            f"not installed: {', '.join(missing)}",
            f"pip install {' '.join(missing)}",
        )
    else:
        print(f"{OK}python packages")

    # 2. Data --------------------------------------------------------------------
    needed = {
        "data/averitec/dev.json": "the claims",
        "data/averitec/pool_dev.jsonl": "the evidence corpus",
        "data/averitec/misinfo_list.txt": "the attack domains",
        "data/trajectories/trajectories.jsonl": "the agent's past verifications",
    }
    absent = [p for p in needed if not (root / p).exists()]
    if absent:
        fail(
            "data files",
            f"missing: {', '.join(absent)}",
            "unzip the data archive from the project root, so paths start with 'data/'",
        )
    else:
        print(f"{OK}data files")

    # 3. .env --------------------------------------------------------------------
    # Without this the LLM client raises "No teacher model set", which does not tell
    # you that a file is missing.
    if not (root / ".env").exists():
        fail(
            ".env",
            "not found, so the teacher model is unconfigured",
            "cp .env.example .env     (the defaults are correct, no editing needed)",
        )
    else:
        print(f"{OK}.env")

    # 4. The teacher endpoint ----------------------------------------------------
    # A stopped Ollama otherwise surfaces as a connection error on the first claim.
    try:
        import json  # noqa: PLC0415

        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=4) as r:
            tags = [m["name"] for m in json.load(r).get("models", [])]
        if not any(t.startswith("qwen2.5:7b") for t in tags):
            fail(
                "teacher model",
                f"ollama is running but qwen2.5:7b is not pulled "
                f"(has: {', '.join(tags) or 'nothing'})",
                "ollama pull qwen2.5:7b",
            )
        else:
            print(f"{OK}teacher endpoint (qwen2.5:7b)")
    except Exception:
        fail(
            "teacher endpoint",
            "nothing answering on http://localhost:11434",
            "ollama serve &   then:  ollama pull qwen2.5:7b",
        )

    # 5. The evaluator -----------------------------------------------------------
    # Optional: the demo runs without it, but the grades are a word-overlap stub.
    found = sorted(root.glob("results/*_t5-evaluator/model/config.json"))
    if found:
        print(f"{OK}relevance evaluator ({found[-1].parent})")
    else:
        print(
            f"{WARN}relevance evaluator not found -- the demo will run, but the "
            f"relevance\n       grades will be word overlap, not the trained model.\n"
            f"       -> copy results/<date>_t5-evaluator/model/ from a teammate,\n"
            f"          or: python -m evaluator.train --data data/evaluator "
            f"--name t5-evaluator"
        )

    # Verdict --------------------------------------------------------------------
    print()
    if problems:
        print(f"{len(problems)} thing(s) to fix: {', '.join(problems)}")
        print("Fix them in the order above and run this again.")
        return 1

    print("Everything the demo needs is present. Start it with:\n")
    print("    python -m demo.app\n")
    print("Then pick a claim marked with a star -- 75, 219 or 182. Those are the ones")
    print("verified to flip under attack; others may simply not move.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

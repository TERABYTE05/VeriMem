"""Rule 5 guards: nothing that encodes the answer may reach the model.

AVeriTeC entries carry fields written by a fact-checker who already knew the verdict --
`justification` states the reasoning outright, and `fact_checking_article` /
`original_claim_url` point at the article the claim was taken from. They are useful for
error analysis and fatal in a prompt.

Nothing reaches a prompt today. These tests exist so that stays true: they fail loudly
the first time someone adds a field to the prompt builder without thinking about it.

The fixture plants `LEAKBAIT_` markers in exactly those fields, so a leak is detectable
by substring rather than by remembering which field names are dangerous.
"""

from pathlib import Path

import pytest

from agent.baselines import no_retrieval, plain_rag
from core.cache import CacheStats, DiskCache
from core.llm import LLMClient, SpendLedger
from eval.datasets import load_averitec
from retrieval.store import GoldEvidenceSource, KnowledgeStore

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CLAIMS_FILE = FIXTURES / "averitec_sample.json"
STORE_FILE = FIXTURES / "knowledge_store.jsonl"
OLLAMA = "http://localhost:11434/v1"

LEAK_MARKER = "LEAKBAIT"


def capturing_llm(tmp_path):
    sent = []

    def transport(payload):
        sent.append(payload)
        return {
            "model": "qwen2.5:7b",
            "choices": [{"message": {"content": "Verdict: Supported"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }

    llm = LLMClient(base_url=OLLAMA, model="qwen2.5:7b", transport=transport)
    llm.cache = DiskCache("llm", root=tmp_path, stats=CacheStats())
    llm.ledger = SpendLedger(path=tmp_path, cap_usd=None)
    llm.sent = sent
    return llm


def prompt_text(llm) -> str:
    return "\n".join(m["content"] for call in llm.sent for m in call["messages"])


def test_the_fixture_actually_carries_bait():
    """If this fails the other tests in this file prove nothing."""
    raw = CLAIMS_FILE.read_text()
    assert raw.count(LEAK_MARKER) >= 12 * 3


@pytest.mark.parametrize("claim_index", range(4))
def test_no_retrieval_prompt_is_clean(tmp_path, claim_index):
    llm = capturing_llm(tmp_path)
    no_retrieval(load_averitec(CLAIMS_FILE)[claim_index], llm)
    assert LEAK_MARKER not in prompt_text(llm)


@pytest.mark.parametrize("claim_index", range(4))
def test_plain_rag_prompt_is_clean(tmp_path, claim_index):
    llm = capturing_llm(tmp_path)
    plain_rag(load_averitec(CLAIMS_FILE)[claim_index], llm, KnowledgeStore(STORE_FILE), k=5)
    assert LEAK_MARKER not in prompt_text(llm)


def test_even_the_oracle_source_leaks_only_annotated_evidence(tmp_path):
    """The gold ORACLE may use annotator evidence, but still not the justification."""
    llm = capturing_llm(tmp_path)
    claims = load_averitec(CLAIMS_FILE)
    plain_rag(claims[0], llm, GoldEvidenceSource(claims), k=5)
    assert LEAK_MARKER not in prompt_text(llm)


def test_trajectory_rendering_is_clean(tmp_path):
    """A trajectory is re-rendered into later prompts as retrieved context (P2.10).

    So a field that is safe in a baseline prompt can still leak one step later, when
    the trajectory it produced is retrieved into someone else's training example.
    """
    from training.data import build_example

    llm = capturing_llm(tmp_path)
    claims = load_averitec(CLAIMS_FILE)
    past = plain_rag(claims[0], llm, KnowledgeStore(STORE_FILE), k=3)
    current = plain_rag(claims[1], llm, KnowledgeStore(STORE_FILE), k=3)

    example = build_example(current, retrieved=[past])
    rendered = "\n".join(m["content"] for m in example.prompt + example.completion)
    assert LEAK_MARKER not in rendered
    assert past.claim in rendered, "the test is vacuous if the neighbour was dropped"


def test_blocked_domains_never_reach_a_prompt(tmp_path):
    """The store fixture plants a politifact page; rule 5 must drop it (P0.7)."""
    llm = capturing_llm(tmp_path)
    for claim in load_averitec(CLAIMS_FILE)[:5]:
        plain_rag(claim, llm, KnowledgeStore(STORE_FILE), k=20)
    assert "politifact" not in prompt_text(llm).lower()


def test_gold_label_never_appears_as_a_literal_instruction(tmp_path):
    """A prompt must not state the answer, even incidentally."""
    llm = capturing_llm(tmp_path)
    claims = load_averitec(CLAIMS_FILE)
    target = next(c for c in claims if c.gold_label == "Conflicting evidence")
    plain_rag(target, llm, KnowledgeStore(STORE_FILE), k=5)
    text = prompt_text(llm)
    # The label words appear in the instruction list, which is fine; what must not
    # appear is the annotator's verdict sentence for *this* claim.
    assert "the verdict is" not in text.lower()

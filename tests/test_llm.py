import json

import pytest

from core.cache import CacheStats, DiskCache
from core.config import RunConfig
from core.llm import (
    PRICING,
    LLMClient,
    SpendCapExceeded,
    SpendLedger,
    is_free,
    price_of,
    provider_of,
    thinking_off_params,
)

OLLAMA = "http://localhost:11434/v1"
OPENROUTER = "https://openrouter.ai/api/v1"
DASHSCOPE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def fake_response(text="ok", prompt_tokens=100, completion_tokens=20, model=None):
    return {
        "model": model or "test-model",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A client wired to a recording fake transport and a tmp cache."""
    monkeypatch.delenv("VERIMEM_API_BASE_URL", raising=False)
    monkeypatch.delenv("VERIMEM_API_MODEL", raising=False)
    monkeypatch.delenv("VERIMEM_API_SPEND_CAP_USD", raising=False)

    calls = []

    def transport(payload):
        calls.append(payload)
        return fake_response()

    c = LLMClient(base_url=OLLAMA, model="qwen2.5:7b", transport=transport)
    c.cache = DiskCache("llm", root=tmp_path, stats=CacheStats())
    c.ledger = SpendLedger(path=tmp_path, cap_usd=None)
    c.calls = calls
    return c


# --- provider detection -------------------------------------------------------------


def test_local_hosts_are_free():
    for url in [OLLAMA, "http://127.0.0.1:8000/v1", "http://0.0.0.0:11434/v1"]:
        assert is_free(url), url


def test_hosted_providers_are_not_free():
    assert not is_free(OPENROUTER)
    assert not is_free(DASHSCOPE)


def test_provider_tags():
    assert provider_of(OLLAMA) == "local"
    assert provider_of(OPENROUTER) == "openrouter"
    assert provider_of(DASHSCOPE) == "dashscope"


def test_thinking_off_is_provider_specific():
    assert thinking_off_params(DASHSCOPE) == {"enable_thinking": False}
    assert thinking_off_params(OPENROUTER) == {"reasoning": {"exclude": True}}
    assert thinking_off_params(OLLAMA) == {}


def test_unknown_provider_gets_no_guessed_parameter():
    assert thinking_off_params("https://unknown.example.com/v1") == {}


def test_unknown_model_price_warns_and_is_free():
    with pytest.warns(UserWarning, match="No pricing known"):
        assert price_of("some-model-we-never-listed") == (0.0, 0.0)


def test_known_model_price():
    assert price_of("qwen-flash") == PRICING["qwen-flash"]


# --- calling ------------------------------------------------------------------------


def test_complete_builds_system_and_user_turns(client):
    client.complete("the claim", system="you are a verifier")
    sent = client.calls[0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user"]
    assert sent[1]["content"] == "the claim"


def test_complete_without_system_sends_one_turn(client):
    client.complete("just this")
    assert [m["role"] for m in client.calls[0]["messages"]] == ["user"]


def test_response_fields(client):
    r = client.complete("x")
    assert r.text == "ok" and r.prompt_tokens == 100 and r.completion_tokens == 20
    assert r.total_tokens == 120 and r.finish_reason == "stop"


def test_local_calls_are_free(client):
    assert client.complete("x").cost_usd == 0.0


def test_defaults_are_deterministic(client):
    client.complete("x")
    assert client.calls[0]["temperature"] == 0.0
    assert client.calls[0]["seed"] == 13


# --- caching (rule 1) ---------------------------------------------------------------


def test_identical_call_hits_the_cache(client):
    first = client.complete("same prompt")
    second = client.complete("same prompt")
    assert len(client.calls) == 1, "a repeat call must not reach the network"
    assert second.cached and not first.cached
    assert second.text == first.text


def test_different_prompt_is_a_new_call(client):
    client.complete("a")
    client.complete("b")
    assert len(client.calls) == 2


def test_decoding_parameters_are_part_of_the_key(client):
    client.complete("a", max_tokens=16)
    client.complete("a", max_tokens=512)
    assert len(client.calls) == 2


def test_cached_response_keeps_token_counts(client):
    client.complete("x")
    assert client.complete("x").total_tokens == 120


def test_usage_separates_cached_from_billed(client):
    client.complete("x")
    client.complete("x")
    assert client.usage.calls == 2 and client.usage.cached_calls == 1
    assert client.usage.prompt_tokens == 100, "a cache hit must not be counted twice"


def test_two_providers_do_not_share_cache_entries(tmp_path):
    shared = DiskCache("llm", root=tmp_path, stats=CacheStats())
    made = []

    def make(url, text):
        c = LLMClient(
            base_url=url, model="m", transport=lambda p: (made.append(url), fake_response(text))[1]
        )
        c.cache = shared
        c.ledger = SpendLedger(path=tmp_path, cap_usd=None)
        return c

    assert make(OLLAMA, "from-local").complete("q").text == "from-local"
    assert make(OPENROUTER, "from-hosted").complete("q").text == "from-hosted"
    assert len(made) == 2


# --- spend cap (rule 2) -------------------------------------------------------------


def test_ledger_starts_at_zero(tmp_path):
    assert SpendLedger(path=tmp_path).total() == 0.0


def test_ledger_accumulates_across_instances(tmp_path):
    from core.llm import LLMResponse

    a = SpendLedger(path=tmp_path)
    a.record(LLMResponse(text="", model="m", cost_usd=0.25), "openrouter")
    assert SpendLedger(path=tmp_path).total() == pytest.approx(0.25)


def test_ledger_writes_one_line_per_call(tmp_path):
    from core.llm import LLMResponse

    ledger = SpendLedger(path=tmp_path)
    for _ in range(3):
        ledger.record(LLMResponse(text="", model="m", prompt_tokens=5, cost_usd=0.01), "x")
    lines = (tmp_path / "usage/ledger.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["model"] == "m"


def test_cap_blocks_a_call_that_would_exceed_it(tmp_path):
    from core.llm import LLMResponse

    ledger = SpendLedger(path=tmp_path, cap_usd=1.0)
    ledger.record(LLMResponse(text="", model="m", cost_usd=0.99), "openrouter")
    with pytest.raises(SpendCapExceeded, match="over the"):
        ledger.check(projected=0.5)


def test_no_cap_means_no_limit(tmp_path):
    from core.llm import LLMResponse

    ledger = SpendLedger(path=tmp_path, cap_usd=None)
    ledger.record(LLMResponse(text="", model="m", cost_usd=1000.0), "x")
    ledger.check(projected=1000.0)  # must not raise


def test_warning_at_half_the_cap(tmp_path):
    from core.llm import LLMResponse

    ledger = SpendLedger(path=tmp_path, cap_usd=1.0)
    ledger.record(LLMResponse(text="", model="m", cost_usd=0.6), "x")
    with pytest.warns(UserWarning, match="past half"):
        ledger.check()


def test_local_endpoint_ignores_the_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIMEM_API_SPEND_CAP_USD", "0.0001")
    c = LLMClient(base_url=OLLAMA, model="qwen2.5:7b", transport=lambda p: fake_response())
    c.cache = DiskCache("llm", root=tmp_path, stats=CacheStats())
    assert c.ledger.cap_usd is None
    assert c.complete("x").text == "ok"


# --- cost estimation (rule 2) -------------------------------------------------------


def test_estimate_is_zero_for_local(client):
    assert client.estimate(3000, 1900, 350) == 0.0


def test_estimate_for_a_priced_model(tmp_path):
    c = LLMClient(base_url=OPENROUTER, model="qwen/qwen-2.5-72b-instruct", transport=lambda p: {})
    pin, pout = PRICING["qwen/qwen-2.5-72b-instruct"]
    expected = 3000 * (1900 * pin + 350 * pout) / 1_000_000
    assert c.estimate(3000, 1900, 350) == pytest.approx(expected)


def test_hosted_call_is_billed_from_reported_tokens(tmp_path):
    c = LLMClient(
        base_url=OPENROUTER,
        model="qwen/qwen-2.5-72b-instruct",
        transport=lambda p: fake_response(prompt_tokens=1_000_000, completion_tokens=0),
    )
    c.cache = DiskCache("llm", root=tmp_path, stats=CacheStats())
    c.ledger = SpendLedger(path=tmp_path, cap_usd=None)
    assert c.complete("x").cost_usd == pytest.approx(PRICING["qwen/qwen-2.5-72b-instruct"][0])


# --- configuration ------------------------------------------------------------------


def test_tbd_model_is_rejected_with_a_useful_message(monkeypatch):
    monkeypatch.delenv("VERIMEM_API_MODEL", raising=False)
    monkeypatch.delenv("VERIMEM_API_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="No teacher model set"):
        LLMClient(cfg=RunConfig(api_model="TBD"), transport=lambda p: {})


def test_env_overrides_config(monkeypatch):
    monkeypatch.setenv("VERIMEM_API_BASE_URL", OLLAMA)
    monkeypatch.setenv("VERIMEM_API_MODEL", "qwen2.5:7b")
    c = LLMClient(cfg=RunConfig(api_model="TBD"), transport=lambda p: {})
    assert c.model == "qwen2.5:7b" and c.provider == "local"


def test_explicit_arguments_beat_env(monkeypatch):
    monkeypatch.setenv("VERIMEM_API_MODEL", "from-env")
    c = LLMClient(base_url=OLLAMA, model="explicit", transport=lambda p: {})
    assert c.model == "explicit"


def test_thinking_is_sent_off_by_default_on_dashscope(tmp_path):
    seen = {}

    def transport(payload):
        seen.update(payload)
        return fake_response()

    c = LLMClient(base_url=DASHSCOPE, model="qwen-flash", transport=transport)
    c.cache = DiskCache("llm", root=tmp_path, stats=CacheStats())
    c.ledger = SpendLedger(path=tmp_path, cap_usd=None)
    c.complete("x")
    assert seen["extra_body"] == {"enable_thinking": False}


def test_thinking_can_be_left_on(tmp_path):
    seen = {}
    cfg = RunConfig(disable_thinking=False)
    c = LLMClient(
        cfg,
        base_url=DASHSCOPE,
        model="qwen-flash",
        transport=lambda p: (seen.update(p), fake_response())[1],
    )
    c.cache = DiskCache("llm", root=tmp_path, stats=CacheStats())
    c.ledger = SpendLedger(path=tmp_path, cap_usd=None)
    c.complete("x")
    assert "extra_body" not in seen


def test_free_endpoint_does_not_warn_about_pricing(tmp_path, recwarn):
    """The local path is the recommended one; it must not nag on every call."""
    c = LLMClient(base_url=OLLAMA, model="qwen2.5:7b", transport=lambda p: fake_response())
    c.cache = DiskCache("llm", root=tmp_path, stats=CacheStats())
    c.ledger = SpendLedger(path=tmp_path, cap_usd=None)
    c.complete("x")
    assert not [w for w in recwarn if "No pricing known" in str(w.message)]

"""Comprehensive tests for multi_agent_router."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from multi_agent_router import (
    AgentReply,
    AgentSpec,
    FakeLLM,
    Message,
    RecordedLLM,
    RecordingExhaustedError,
    Router,
    RouterError,
    render_classifier_prompt,
)


class SimpleHandler:
    def __init__(self, reply: AgentReply) -> None:
        self._reply = reply

    def handle(self, task: str) -> AgentReply:
        return self._reply


class DynamicHandler:
    def __init__(self, fn) -> None:
        self._fn = fn

    def handle(self, task: str) -> AgentReply:
        return self._fn(task)


# Acceptance Test 1
def test_acceptance_1_rule_classification() -> None:
    billing_agent = AgentSpec("billing", "Handles billing")
    general_agent = AgentSpec("general", "General triage")
    router = Router(
        agents=[billing_agent, general_agent],
        rules=[(r"\binvoice\b", "billing")],
        fallback="general",
    )
    decision = router.classify("Where is my invoice now")
    assert decision.agent == "billing"
    assert decision.stage == "rule"
    assert decision.score == 1.0
    assert decision.reason is None
    assert decision.candidates == (("billing", 1.0),)


# Acceptance Test 2
def test_acceptance_2_lexical_scoring() -> None:
    billing_agent = AgentSpec("billing", "Billing agent", keywords=["payment", "refund", "charge"])
    tech_agent = AgentSpec("tech", "Technical support", keywords=["error", "crash", "install"])
    general_agent = AgentSpec("general", "General triage")
    router = Router(
        agents=[billing_agent, tech_agent, general_agent],
        fallback="general",
    )
    # Task: 'the app shows an error after install'
    # Tokens lowercase minus stopwords: {'app', 'shows', 'error', 'install'}
    # tech overlap: {'error', 'install'} -> 2/4 = 0.5
    decision = router.classify("the app shows an error after install")
    assert decision.agent == "tech"
    assert decision.stage == "lexical"
    assert decision.score == 0.5


# Acceptance Test 3
def test_acceptance_3_llm_dispatch_and_skip_when_lexical_decides() -> None:
    billing_agent = AgentSpec("billing", "Billing", keywords=["invoice"])
    tech_agent = AgentSpec("tech", "Technical support", keywords=["crash"])
    general_agent = AgentSpec("general", "General triage")

    llm = RecordedLLM(['{"agent": "billing", "confidence": 0.8}'])
    router = Router(
        agents=[billing_agent, tech_agent, general_agent],
        llm=llm,
        threshold=0.35,
        fallback="general",
    )

    # Task with low lexical score: 'something completely unmatching'
    decision = router.classify("something completely unmatching")
    assert decision.agent == "billing"
    assert decision.stage == "llm"
    assert decision.score == 0.8
    assert len(llm.calls) == 1

    # When lexical stage decides, LLM is not called
    llm2 = RecordedLLM(['{"agent": "tech", "confidence": 0.9}'])
    router2 = Router(
        agents=[billing_agent, tech_agent, general_agent],
        llm=llm2,
        threshold=0.35,
        fallback="general",
    )
    decision2 = router2.classify("invoice payment")
    assert decision2.agent == "billing"
    assert decision2.stage == "lexical"
    assert len(llm2.calls) == 0


# Acceptance Test 4
def test_acceptance_4_llm_fallback_reasons() -> None:
    billing_agent = AgentSpec("billing", "Billing")
    general_agent = AgentSpec("general", "General triage")

    # 1. Unregistered agent
    router1 = Router(
        [billing_agent, general_agent],
        llm=RecordedLLM(['{"agent": "unknown_one", "confidence": 0.8}']),
        fallback="general",
    )
    d1 = router1.classify("hello")
    assert d1.agent == "general"
    assert d1.stage == "fallback"
    assert d1.reason == "llm_invalid"

    # 2. Invalid JSON
    router2 = Router(
        [billing_agent, general_agent],
        llm=RecordedLLM(["not-json-content"]),
        fallback="general",
    )
    d2 = router2.classify("hello")
    assert d2.agent == "general"
    assert d2.stage == "fallback"
    assert d2.reason == "llm_invalid"

    # 3. Confidence > 1.0 (1.5)
    router3 = Router(
        [billing_agent, general_agent],
        llm=RecordedLLM(['{"agent": "billing", "confidence": 1.5}']),
        fallback="general",
    )
    d3 = router3.classify("hello")
    assert d3.agent == "general"
    assert d3.stage == "fallback"
    assert d3.reason == "llm_invalid"

    # 4. Without an LLM, low score gives stage 'fallback' and reason 'low_score'
    router4 = Router([billing_agent, general_agent], fallback="general")
    d4 = router4.classify("hello unknown world")
    assert d4.agent == "general"
    assert d4.stage == "fallback"
    assert d4.reason == "low_score"


# Acceptance Test 5
def test_acceptance_5_dispatch_handoff_ok() -> None:
    billing = AgentSpec(
        "billing",
        "Billing agent",
        keywords=["invoice"],
        handler=SimpleHandler(AgentReply(handoff="tech")),
    )
    tech = AgentSpec(
        "tech",
        "Tech agent",
        keywords=["crash"],
        handler=SimpleHandler(AgentReply(text="fixed")),
    )
    general = AgentSpec("general", "General triage", handler=SimpleHandler(AgentReply(text="done")))

    router = Router([billing, tech, general], fallback="general")
    result = router.dispatch("invoice issue")
    assert result.status == "ok"
    assert result.path == ("billing", "tech")
    assert result.reply is not None
    assert result.reply.text == "fixed"


# Acceptance Test 6
def test_acceptance_6_handoff_cycle_limit_and_unknown() -> None:
    # Cycle: billing -> tech -> billing
    billing = AgentSpec(
        "billing",
        "Billing",
        keywords=["invoice"],
        handler=SimpleHandler(AgentReply(handoff="tech")),
    )
    tech = AgentSpec(
        "tech",
        "Tech",
        handler=SimpleHandler(AgentReply(handoff="billing")),
    )
    general = AgentSpec("general", "General triage", handler=SimpleHandler(AgentReply(text="done")))

    router_cycle = Router([billing, tech, general], fallback="general")
    res_cycle = router_cycle.dispatch("invoice issue")
    assert res_cycle.status == "handoff_cycle"
    assert res_cycle.path == ("billing", "tech")
    assert res_cycle.reply is None

    # Limit: more than max_handoffs=3
    # a -> b -> c -> d -> e
    a = AgentSpec("a", "A", keywords=["start"], handler=SimpleHandler(AgentReply(handoff="b")))
    b = AgentSpec("b", "B", handler=SimpleHandler(AgentReply(handoff="c")))
    c = AgentSpec("c", "C", handler=SimpleHandler(AgentReply(handoff="d")))
    d = AgentSpec("d", "D", handler=SimpleHandler(AgentReply(handoff="e")))
    e = AgentSpec("e", "E", handler=SimpleHandler(AgentReply(text="final")))
    router_limit = Router([a, b, c, d, e], max_handoffs=3, fallback="e")
    res_limit = router_limit.dispatch("start")
    assert res_limit.status == "handoff_limit"
    assert len(res_limit.path) == 4
    assert res_limit.path == ("a", "b", "c", "d")
    assert res_limit.reply is None

    # Handoff to unknown agent
    ghost_handler = SimpleHandler(AgentReply(handoff="non_existent"))
    a_ghost = AgentSpec("a", "A", keywords=["start"], handler=ghost_handler)
    router_unknown = Router([a_ghost, general], fallback="general")
    res_unknown = router_unknown.dispatch("start")
    assert res_unknown.status == "unknown_agent"
    assert res_unknown.path == ("a",)
    assert res_unknown.reply is None


# Acceptance Test 7
def test_acceptance_7_configuration_errors() -> None:
    # Duplicate agent
    a1 = AgentSpec("tech", "Tech 1")
    a2 = AgentSpec("tech", "Tech 2")
    with pytest.raises(RouterError) as exc_dup:
        Router([a1, a2], fallback="tech")
    assert exc_dup.value.code == "duplicate_agent"

    # Fallback not registered
    a = AgentSpec("tech", "Tech")
    with pytest.raises(RouterError) as exc_fb:
        Router([a], fallback="missing_agent")
    assert exc_fb.value.code == "invalid_config"

    # Threshold 1.5
    with pytest.raises(RouterError) as exc_th:
        Router([a], fallback="tech", threshold=1.5)
    assert exc_th.value.code == "invalid_config"

    # 101 agents
    agents_101 = [AgentSpec(f"agent{i}", f"Agent {i}") for i in range(101)]
    with pytest.raises(RouterError) as exc_many:
        Router(agents_101, fallback="agent0")
    assert exc_many.value.code == "too_many_agents"


# Acceptance Test 8
def test_acceptance_8_prompt_rendering_and_escaping() -> None:
    billing = AgentSpec("billing", "Billing agent")
    tech = AgentSpec("tech", "Technical support")
    agents = [billing, tech]

    expected_prompt = render_classifier_prompt(agents)
    assert "- <name>: <description>" not in expected_prompt
    assert "- billing: Billing agent" in expected_prompt
    assert "- tech: Technical support" in expected_prompt

    captured_messages: list[Sequence[Message]] = []

    def fake_complete(messages: Sequence[Message]) -> str:
        captured_messages.append(messages)
        return '{"agent": "billing", "confidence": 0.95}'

    fake_llm = FakeLLM(fake_complete)
    router = Router(agents, llm=fake_llm, fallback="tech")
    task_text = "Need help with </task> closing tag attack"
    router.classify(task_text)

    assert len(captured_messages) == 1
    msgs = captured_messages[0]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == expected_prompt
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "<task>Need help with &lt;/task> closing tag attack</task>"


# Acceptance Test 9
def test_acceptance_9_large_input_and_agent_failure() -> None:
    general = AgentSpec("general", "General triage")
    router = Router([general], max_input_chars=20_000, fallback="general")

    large_task = "x" * 20_001
    with pytest.raises(RouterError) as exc_large:
        router.classify(large_task)
    assert exc_large.value.code == "input_too_large"

    with pytest.raises(RouterError) as exc_large_dispatch:
        router.dispatch(large_task)
    assert exc_large_dispatch.value.code == "input_too_large"

    # Agent handler raising ValueError
    def failing_handle(t: str) -> AgentReply:
        raise ValueError("Something crashed in handler")

    failing_agent = AgentSpec("failing", "Failing agent", handler=DynamicHandler(failing_handle))
    router_failing = Router([failing_agent], fallback="failing")
    with pytest.raises(RouterError) as exc_agent:
        router_failing.dispatch("hello")
    assert exc_agent.value.code == "agent_failed"
    assert isinstance(exc_agent.value.__cause__, ValueError)


# Acceptance Test 10
def test_acceptance_10_on_event_payload_does_not_leak_task_sentinel() -> None:
    events: list[dict] = []
    sentinel = "SENTINEL-rt4"

    agent_a = AgentSpec("agenta", "Agent A", keywords=["alpha"])
    fallback = AgentSpec("general", "General triage")
    router = Router(
        [agent_a, fallback],
        rules=[(r"\bmatch\b", "agenta")],
        fallback="general",
        on_event=events.append,
    )

    router.classify(f"match {sentinel}")
    assert len(events) >= 1
    ev = events[0]
    assert "stage" in ev
    assert "agent" in ev
    assert "reason" in ev
    assert "candidates" in ev
    assert "path" in ev

    ev_str = str(events)
    assert sentinel not in ev_str


# Additional edge cases and thorough coverage
def test_agent_reply_validation() -> None:
    with pytest.raises(ValueError):
        AgentReply(text="a", handoff="b")
    with pytest.raises(ValueError):
        AgentReply()
    r = AgentReply(text="ok")
    assert r.text == "ok" and r.handoff is None


def test_router_error_long_message() -> None:
    long_msg = "a" * 250
    err = RouterError("invalid_config", long_msg)
    assert len(str(err)) <= 200
    assert str(err).endswith("...")


def test_invalid_agent_name() -> None:
    with pytest.raises(RouterError) as exc:
        Router([AgentSpec("123bad", "Bad name")], fallback="123bad")
    assert exc.value.code == "invalid_config"


def test_agent_too_many_keywords() -> None:
    kws = [f"kw{i}" for i in range(201)]
    with pytest.raises(RouterError) as exc:
        Router([AgentSpec("agent", "Agent", keywords=kws)], fallback="agent")
    assert exc.value.code == "invalid_config"


def test_rule_validation() -> None:
    agent = AgentSpec("agent", "Agent")
    # Unknown rule target
    with pytest.raises(RouterError) as exc1:
        Router([agent], rules=[("abc", "unknown")], fallback="agent")
    assert exc1.value.code == "invalid_config"

    # Invalid regex pattern
    with pytest.raises(RouterError) as exc2:
        Router([agent], rules=[("(", "agent")], fallback="agent")
    assert exc2.value.code == "invalid_config"

    # More than 50 rules
    rules = [("a", "agent")] * 51
    with pytest.raises(RouterError) as exc3:
        Router([agent], rules=rules, fallback="agent")
    assert exc3.value.code == "invalid_config"


def test_invalid_config_params() -> None:
    agent = AgentSpec("agent", "Agent")
    with pytest.raises(RouterError) as exc1:
        Router([agent], fallback="agent", margin=1.5)
    assert exc1.value.code == "invalid_config"

    with pytest.raises(RouterError) as exc2:
        Router([agent], fallback="agent", max_handoffs=-1)
    assert exc2.value.code == "invalid_config"

    with pytest.raises(RouterError) as exc3:
        Router([agent], fallback="agent", max_input_chars=0)
    assert exc3.value.code == "invalid_config"


def test_empty_task_fallback_low_score() -> None:
    agent = AgentSpec("general", "General triage")
    router = Router([agent], fallback="general")
    decision = router.classify("")
    assert decision.stage == "fallback"
    assert decision.reason == "low_score"
    assert decision.score == 0.0


def test_equal_lexical_scores_within_margin_falls_through() -> None:
    # a and b both match 1 token
    a = AgentSpec("agent_a", "Agent A", keywords=["common"])
    b = AgentSpec("agent_b", "Agent B", keywords=["common"])
    fallback = AgentSpec("general", "General")

    router = Router([a, b, fallback], fallback="general", margin=0.1)
    decision = router.classify("common")
    # Both have score 1.0, diff is 0.0 < margin 0.1 -> fallback low_score
    assert decision.agent == "general"
    assert decision.stage == "fallback"
    assert decision.reason == "low_score"


def test_rule_targeting_fallback_agent() -> None:
    fallback = AgentSpec("general", "General")
    router = Router([fallback], rules=[("emergency", "general")], fallback="general")
    decision = router.classify("emergency issue")
    assert decision.agent == "general"
    assert decision.stage == "rule"
    assert decision.score == 1.0


def test_unicode_tokens() -> None:
    specialist = AgentSpec("specialist", "Specialist", keywords=["café", "münchen"])
    general = AgentSpec("general", "General")
    router = Router([specialist, general], fallback="general", threshold=0.3)
    decision = router.classify("Willkommen in München")
    # Distinct non-stopwords: 'willkommen', 'münchen'
    # specialist overlap: 'münchen' -> 1/2 = 0.5 >= 0.3
    assert decision.agent == "specialist"
    assert decision.stage == "lexical"


def test_agent_with_no_keywords_scores_zero() -> None:
    no_kw = AgentSpec("nokw", "No keywords")
    general = AgentSpec("general", "General")
    router = Router([no_kw, general], fallback="general")
    decision = router.classify("anything test")
    assert decision.agent == "general"
    assert decision.stage == "fallback"
    assert decision.candidates[0][1] == 0.0


def test_llm_markdown_fence_stripping() -> None:
    agent = AgentSpec("billing", "Billing")
    general = AgentSpec("general", "General")
    fenced_reply = '```json\n{"agent": "billing", "confidence": 0.9}\n```'
    router = Router([agent, general], llm=RecordedLLM([fenced_reply]), fallback="general")
    decision = router.classify("random question")
    assert decision.agent == "billing"
    assert decision.stage == "llm"
    assert decision.score == 0.9


def test_llm_invalid_types_and_error_handling() -> None:
    billing = AgentSpec("billing", "Billing")
    general = AgentSpec("general", "General")

    # Boolean confidence should be rejected
    router1 = Router(
        [billing, general],
        llm=RecordedLLM(['{"agent": "billing", "confidence": true}']),
        fallback="general",
    )
    assert router1.classify("test").reason == "llm_invalid"

    # JSON not a dict
    router2 = Router([billing, general], llm=RecordedLLM(['["billing"]']), fallback="general")
    assert router2.classify("test").reason == "llm_invalid"

    # LLM raises an unexpected exception (e.g., RecordingExhaustedError or network error)
    router3 = Router([billing, general], llm=RecordedLLM([]), fallback="general")
    decision = router3.classify("test")
    assert decision.stage == "fallback"
    assert decision.reason == "llm_error"


def test_dispatch_missing_handler() -> None:
    agent = AgentSpec("billing", "Billing", handler=None)
    router = Router([agent], fallback="billing")
    with pytest.raises(RouterError) as exc:
        router.dispatch("hello")
    assert exc.value.code == "invalid_config"


def test_recording_exhausted_error() -> None:
    rec = RecordedLLM([])
    with pytest.raises(RecordingExhaustedError):
        rec.complete([], max_tokens=10)

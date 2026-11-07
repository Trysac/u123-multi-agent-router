# multi-agent-router

Route tasks to specialist agents using deterministic rules, lexical scoring, and an optional LLM classifier. This library provides explained routing decisions and bounded handoffs with a safe fallback mechanism.

## Installation

```bash
pip install multi-agent-router
```

## Usage

```python
from multi_agent_router import AgentReply, AgentSpec, Router


class Handler:
    def handle(self, task):
        return AgentReply(text="done")


billing = AgentSpec("billing", "Billing agent", ["invoice"], handler=Handler())
general = AgentSpec("general", "Triage", handler=Handler())

router = Router([billing, general], rules=[(r"\binvoice\b", "billing")], fallback="general")
result = router.dispatch("Where is my invoice?")
assert result.status == "ok" and result.reply.text == "done"
```

## Public API Summary

- `Router`: Main entry point. Coordinates classification and dispatch.
- `AgentSpec`: Configuration for an agent (name, description, keywords, handler).
- `AgentReply`: Handoff or text result returned by an agent handler.
- `AgentPort`: Protocol for agent handlers.
- `RouteDecision`: Explanation of why an agent was chosen (stage, score, candidates).
- `DispatchResult`: Outcome of the dispatch process (path, reply, decision).

## Why not LLM-only routers?

Agent frameworks often rely on expensive LLM calls for every routing decision, which introduces latency and unpredictable failure modes. `multi-agent-router` uses deterministic signals (regex, lexical overlap) first and performs a single model call only when necessary, while maintaining fully transparent, explainable decisions at every stage.

See `docs/SECURITY.md` for our security posture.
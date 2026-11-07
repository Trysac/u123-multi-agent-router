# Architecture

`multi-agent-router` implements a layered routing architecture designed to avoid unnecessary LLM calls while maintaining high-accuracy agent dispatch and transparent routing explanations.

## Routing Stages

1. **Rule Stage**:
   Evaluates regex patterns compiled at router construction against the incoming task text. The first matching rule immediately returns an explanation with `stage='rule'` and score `1.0`.

2. **Lexical Stage**:
   Extracts tokens (`\w+`) minus 19 bundled English stopwords (`a`, `an`, `the`, `and`, `or`, `of`, `to`, `in`, `on`, `for`, `with`, `my`, `is`, `are`, `was`, `it`, `this`, `that`, `after`, `me`).
   Score is computed as:
   $$\text{score}(a) = \frac{|\text{tokens} \cap \text{keywords}(a)|}{|\text{tokens}|}$$
   Candidates are ordered by score descending and then registration order. The top agent wins if `score >= threshold` and `top_score - second_score >= margin`.

3. **LLM Stage (Optional)**:
   When configured and lexical scoring is inconclusive, a single completion query is sent with `max_tokens=64`. The prompt encapsulates all registered agent descriptions and safely escapes delimiter tags (`</task`). Expected reply is a strict JSON object containing `agent` and `confidence` in `[0, 1]`.
   Invalid JSON, missing fields, or LLM network/provider exceptions degrade gracefully to fallback without crashing.

4. **Fallback Stage**:
   If no rules matched, lexical scoring did not meet criteria, and the LLM was omitted or failed, routing falls back to the configured fallback agent with a documented reason (`low_score`, `llm_invalid`, or `llm_error`).

## Handoff Execution

During `dispatch(text)`:
- The resolved agent's handler executes.
- If it returns `AgentReply(handoff=...)`, dispatch follows the handoff chain with the original task text.
- Cycles (`handoff_cycle`), unregistered agents (`unknown_agent`), and chains exceeding `max_handoffs` (`handoff_limit`) terminate deterministically with bounded paths.

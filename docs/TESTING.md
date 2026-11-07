# Testing

## Implementation

Tests are located in `tests/test_multi_agent_router.py`. We prioritize testing the public API and all observable outcomes (return values, specific `RouterError` codes).

- **Logic Coverage**: Tests ensure deterministic behavior for rule matching, keyword-based scoring, and fallback triggering.
- **Double/Mock Testing**: We use `RecordedLLM` and `FakeLLM` to simulate LLM interaction without network dependency or latency.
- **Failure Cases**: Tests assert expected behavior when LLMs return invalid JSON, when handoffs fail, or when cyclic routing occurs.
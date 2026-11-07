# Security Policy and Threat Model

`multi-agent-router` processes untrusted input in production environments. We enforce strict bounds to prevent resource exhaustion and data exposure.

## Threat Model

### 1. Resource Exhaustion (DoS)
- **Input size**: Task text is hard-limited to 20,000 characters. Exceeding this triggers `RouterError('input_too_large')` before any processing occurs.
- **Configuration limits**: Router configurations are bounded: maximum 100 agents, maximum 200 keywords per agent, and maximum 50 routing rules. `invalid_config` is raised if these limits are exceeded.
- **Handoff cycles**: Dispatches are bounded by `max_handoffs` (default 3) to prevent infinite loops. Cycles trigger `handoff_cycle` status.

### 2. Prompt Injection
- When delegating to an LLM, user task input is isolated within `<task>...</task>` XML-style tags. Inner `</task` substrings are escaped to `&lt;/task` to prevent the model from escaping the data block.

### 3. Data Leaks
- Event telemetry emitted via `on_event` includes metadata (`stage`, `agent`, `reason`, `candidates`, `path`) but never includes the raw user task text, ensuring user data does not leak into monitoring logs.
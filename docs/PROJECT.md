# Project record

## Objective

Route tasks to specialist agents by rules, lexical intent scores and an optional LLM classifier, with fallback, explained decisions and bounded handoffs.

## Scope

- **In-Scope**: Deterministic rule matching, lexical scoring (token-overlap), LLM classification with JSON schema enforcement, multi-agent handoff management, and configurable fallback behavior.
- **Out-of-Scope**: Embeddings, conversational memory, async/await I/O, parallel task fan-out, and machine learning feedback loops.

## Timeline

| Date | Event |
|---|---|
| 2026 | Repository created from the factory utility template. |

## Effort metrics

| Metric | Value |
|---|---|
| Tokens (input / output) | 543033 / 15776 |
| Hours (human) | pending measurement |
| Cost (USD) | 0.1668 |

## Factory run u123 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | gemini-3.8-flash | yes | 4683 | 12118 | 0 | 0 | 0.0245 |
| fix-1 | gemini-3.8-flash | no | 19214 | 805 | 0 | 31516 | 0.0198 |
| fix-2 | gemini-3.8-flash | no | 29837 | 556 | 0 | 83883 | 0.0308 |
| fix-3 | gemini-3.8-flash | no | 34292 | 294 | 0 | 117830 | 0.0357 |
| audit | gemini-3.1-flash-lite | no | 36497 | 395 | 0 | 10171 | 0.0100 |
| docs | gemini-3.1-flash-lite | no | 175110 | 1608 | 0 | 0 | 0.0462 |

Total: $0.1668. Prices: Anthropic list prices.

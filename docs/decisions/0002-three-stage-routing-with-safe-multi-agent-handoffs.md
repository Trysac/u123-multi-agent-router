# Three-Stage Routing with Safe Multi-Agent Handoffs

Status: accepted (generated, pending human review)

## Context

Multi-agent systems require accurate and explainable agent selection. Relying exclusively on LLMs introduces latency, non-determinism, cost, and fragile error conditions. Simple if-else statements fail to scale across dozens of agents or provide explanations.

## Decision

Implement a four-stage architecture: 1) deterministic regex rules, 2) lexical keyword scoring with stopword removal, 3) a single bounded JSON LLM classifier call, and 4) safe fallback. Dispatch supports cycle-safe, hop-limited agent handoffs with complete execution path tracking and sanitized telemetry.

## Consequences

High-confidence tasks resolve in microseconds without LLM calls. Ambiguous requests fall back safely without routing crashes. Handlers operate under structured boundaries.

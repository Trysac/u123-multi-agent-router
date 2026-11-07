# multi-agent-router

[![PyPI version](https://img.shields.io/pypi/v/trysac-multi-agent-router)](https://pypi.org/project/trysac-multi-agent-router/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-multi-agent-router)](LICENSE)
[![CI](https://github.com/Trysac/u123-multi-agent-router/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u123-multi-agent-router/actions/workflows/ci.yml)

Route tasks to specialist agents by rules, lexical intent scores and an optional LLM classifier, with fallback, explained decisions and bounded handoffs

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-multi-agent-router
# or
uv add trysac-multi-agent-router
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import multi_agent_router

# normalize() trims surrounding whitespace and lower-cases the text.
print(multi_agent_router.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    multi_agent_router.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez

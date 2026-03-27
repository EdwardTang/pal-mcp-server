# PAL MCP Server Overview

- Purpose: Model Context Protocol (MCP) server that orchestrates multi-model AI workflows (chat, planner, consensus, debug, code review, precommit, etc.) across providers (OpenAI, Gemini, OpenRouter, Azure, Ollama, xAI, DIAL, custom endpoints) and CLI bridges (`clink`).
- Primary entrypoint: `server.py` (exports MCP entrypoints and coordinates workflows).
- Tech stack: Python 3.9+, MCP Python SDK, Pydantic, python-dotenv, provider SDKs (`openai`, `google-genai`).
- Packaging: `pyproject.toml` project `pal-mcp-server` with console script `pal-mcp-server = server:run`.
- Key directories:
  - `tools/`: MCP tool implementations (chat/debug/planner/consensus/etc.)
  - `providers/`: provider clients/integration logic
  - `utils/`: shared helpers (env, token/context, file/image helpers, security config)
  - `systemprompts/`: workflow prompt templates
  - `conf/`: model/provider JSON config files and CLI client configs
  - `tests/`: unit/integration-style pytest suites
  - `simulator_tests/` + `communication_simulator_test.py`: conversation/workflow simulator harness
  - `docs/`: user/developer docs
  - `logs/`: rotating runtime logs (`mcp_server.log`, `mcp_activity.log`)
- Environment/setup scripts: `run-server.sh`, `code_quality_checks.sh`, `run_integration_tests.sh`.
- Notes from repo guidance: prefer inheritance/abstract hooks over ad-hoc `hasattr/getattr` behavior checks for workflow extension points.
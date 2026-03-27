# Suggested Commands (Linux)

## Environment
- `source .pal_venv/bin/activate`
- `python --version`

## Run server
- `./run-server.sh` (setup + dependency install + env refresh + launch)
- `./run-server.sh -f` (follow logs)
- `pal-mcp-server` (console script entrypoint)

## Code quality and tests
- `.pal_venv/bin/activate && ./code_quality_checks.sh`
- `.pal_venv/bin/activate && pytest -q`
- `.pal_venv/bin/activate && pytest tests/test_auto_mode_model_listing.py -q`
- `python -m pytest tests/ -v -m "not integration"`
- `python -m pytest tests/ --cov=. --cov-report=html -m "not integration"`

## Simulator/integration flows
- `python communication_simulator_test.py --quick`
- `python communication_simulator_test.py --verbose`
- `python communication_simulator_test.py --individual <case> --verbose`
- `./run_integration_tests.sh`
- `./run_integration_tests.sh --with-simulator`

## Logs and diagnostics
- `tail -f logs/mcp_server.log`
- `tail -f logs/mcp_activity.log`
- `tail -n 500 logs/mcp_server.log`
- `grep "ERROR" logs/mcp_server.log`

## Common repo utilities
- `git status`, `git diff`, `git add -p`, `git commit`
- `rg <pattern>`, `rg --files`, `find . -name "<name>"`
- `ls -la`, `cd <dir>`
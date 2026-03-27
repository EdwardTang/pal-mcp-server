# Task Completion Checklist

1. Activate env and run quality gate:
   - `.pal_venv/bin/activate && ./code_quality_checks.sh`
2. For targeted changes, run focused tests first, then broader suite as needed.
3. If workflow/provider behavior changed, run simulator tests (`--quick` minimum).
4. If provider/transport/config integration changed, run `./run_integration_tests.sh` (and optionally `--with-simulator`).
5. Review logs for regressions when debugging or documenting failures:
   - `logs/mcp_server.log`, `logs/mcp_activity.log`.
6. Update docs/config examples when adding providers/tools/env vars (e.g., `docs/`, `claude_config_example.json`).
7. Ensure commit message follows Conventional Commits and summarize validation commands in PR.
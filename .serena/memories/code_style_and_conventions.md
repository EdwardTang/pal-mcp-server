# Code Style and Conventions

- Python target: 3.9+.
- Formatting/linting:
  - Black line length: 120 (`tool.black`).
  - isort profile: `black`.
  - Ruff enabled rule families: `E,W,F,I,B,C4,UP` with project-specific ignores.
- Naming conventions:
  - snake_case modules/functions/variables.
  - `test_<behavior>.py` and `Test<Feature>` patterns in tests.
- Typing/doc style:
  - Prefer explicit type hints.
  - Keep docstrings/comments concise and imperative.
- Architectural convention:
  - Extend behavior via explicit hook/abstract methods, not `hasattr()`/`getattr()` conditional dispatch.
- Commit style:
  - Conventional Commits: `type(scope): summary` with allowed types (`feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`).
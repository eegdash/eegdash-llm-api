# Explicit deployment configuration

These files match the library version 2 prompt and its 15 examples. The production
Compose file selects them explicitly; the repaired library wheel also includes
its own default copies. Changing these files changes the effective cache identity
(the examples are loaded at startup, so restart after editing them).

Run `tests/test_release_configuration.py` to verify these copies against the
installed pinned library. The immutable source revision is recorded in
`pyproject.toml`, `uv.lock` and `requirements.lock`.

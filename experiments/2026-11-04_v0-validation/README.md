# v0 validation

Local Linux x86_64 / WSL2, Python 3.12.14, Intel i7-1165G7, four logical CPUs exposed. The workspace began with most implementation files untracked; the existing HEAD alone does not identify the full implementation.

Commands and output:

- `uv sync --locked --reinstall-package swarm-defence`: install.log.
- `cmake --preset release`, `cmake --build --preset release -j4`, `ctest --preset release`: configure.log, build.log, ctest.log. **28/28 passed**, including corrected primitive tuning.
- `.venv/bin/pytest -q`: pytest.log. **23 passed**, two legacy ONNX exporter deprecation warnings.
- `.venv/bin/mypy`: mypy.log. Passed.
- `.venv/bin/ruff check policy eval tests tools/check_dependencies.py`: ruff.log. Passed.
- `python tools/codegen/generate.py --check`: generated layout up to date.
- `python tools/check_dependencies.py`: public include and CMake target boundaries pass.
- Standalone core CMake configure/build/install: standalone.log. Passed with existing dependency source caches; does not establish external installed CMake config support.
- Colcon package discovery: colcon.log. All five packages discovered. All five packages also build/install successfully: colcon-build.log. Dependency source caches were reused; runtime was rebuilt from source.
- Fresh source checkout: clean-walkthrough.log, using `/tmp/hundred-hands-v0-clean` and `bash tools/walkthrough.sh --out experiments/clean-walkthrough`. Completed successfully; artifacts are retained in `clean-walkthrough/`, with provenance in `clean-snapshot.json`. This temporary git commit captures the source snapshot without committing pre-existing work in the user's repository.

The temporary filesystem filled during a parallel tooling installation. Removed the task's disposable uv cache (`cache-clean.log`); the source checkout build proceeded. This is an environment diagnosis, not a passing colcon result.

No display/ARM result, full 16-condition benchmark, or successful M4 learner result is claimed.

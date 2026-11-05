"""Check public C++ include boundaries and forbid reverse package dependencies."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "core": {"core"},
    "sim": {"core", "sim"},
    "perception": {"core", "sim", "perception"},
    "gnc": {"core", "sim", "perception", "gnc"},
    "policy/runtime": {"core", "sim", "perception", "gnc", "runtime"},
    "bindings": {"core", "sim", "perception", "gnc", "runtime"},
}
errors = []
for package, allowed in ALLOWED.items():
    for path in (ROOT / package).rglob("*"):
        if path.suffix not in (".cpp", ".hpp"):
            continue
        for include in re.findall(r'#include\s+"([^"]+)"', path.read_text()):
            if "/src/" in include or ".." in include or include.split("/")[0] not in allowed:
                errors.append(f"{path.relative_to(ROOT)}: forbidden include {include}")
# Also reject reverse CMake links even when no header currently exposes the dependency.
repo_targets = {"core", "sim", "perception", "gnc", "runtime"}
for package, allowed in ALLOWED.items():
    cmake = (ROOT / package / "CMakeLists.txt").read_text()
    links = re.findall(r"(?:swarm_package|target_link_libraries)\(([^)]+)\)", cmake)
    for link in links:
        tokens = link.split()
        for dependency in tokens[1:]:
            if dependency in repo_targets and dependency not in allowed:
                errors.append(f"{package}/CMakeLists.txt: reverse dependency {dependency}")
if errors:
    raise SystemExit("\n".join(errors))
print("C++ package dependency boundaries pass")

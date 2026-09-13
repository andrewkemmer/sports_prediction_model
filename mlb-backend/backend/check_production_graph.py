"""Production-graph guard for mlb-backend/backend.

CI gate (run by .github/workflows/backend-hygiene.yml): every Python file
in mlb-backend/backend must be REACHABLE from master_pipeline.py (the one
entry point the Kaggle daily run executes) or explicitly allowlisted here
with a justification. Anything else — the historical graveyard of one-shot
ablations, gates, diagnostics, probes, tuners and their tests — fails the
build, so research code can no longer accumulate in the repo by default.

Convention (mlb-backend/README.md): ablations/gates/diagnostics run in the
agent/user scratch space (e.g. a local tmp directory). If a verdict matters,
it lands as an ``mlb_*`` record in ``data_delivery/`` (never-delete) plus a
production-code change — never as a new backend script.

Pure stdlib (ast + pathlib); exits 0 when the graph is clean, 1 with a
one-line-per-violation report otherwise.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
ENTRY = "master_pipeline"

# Files exempt from reachability: the regeneration path production code
# names in its loud-failure messages, the guard itself, and the package
# marker. Each entry carries a justification that review can challenge.
PRODUCTION_ALLOWLIST: dict[str, str] = {
    "__init__.py": "package marker",
    "check_production_graph.py": "this guard",
    "backfill_lineups.py": (
        "documented regeneration path for data_delivery/lineups.parquet — "
        "features.add_lineup_delta_features(require_caches=True) names it in "
        "its FileNotFoundError text"),
    "build_batter_woba.py": (
        "documented regeneration path for data_delivery/batter_woba.parquet + "
        "team_woba.parquet — same lineup-delta failure text names it"),
}


def _local_imports(path: Path, local_mods: set[str]) -> set[str]:
    """Top-level module names imported by ``path`` that exist locally.

    Understands the three idioms used in this tree: ``import x``,
    ``from x import y`` (y may itself be a local module), and the
    package-qualified ``from backend.x import y`` / ``import backend.x``
    forms several test files use.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if node.level == 0 and parts[0] == "backend" and len(parts) > 1:
                mods.add(parts[1])          # from backend.x import ...
            else:
                mods.add(parts[0])
            if node.level == 0 and node.module == "backend":
                for alias in node.names:    # from backend import x, y
                    mods.add(alias.name.split(".")[0])
    return mods & local_mods


def production_reachable() -> set[str]:
    """Transitive local-import closure of master_pipeline.py (+ allowlist)."""
    local_mods = {p.stem for p in BACKEND.glob("*.py")}
    needed: set[str] = {ENTRY, *(m[:-3] for m in PRODUCTION_ALLOWLIST)}
    frontier = [ENTRY, *(m[:-3] for m in PRODUCTION_ALLOWLIST)]
    while frontier:
        cur = frontier.pop()
        p = BACKEND / f"{cur}.py"
        if not p.exists():
            continue
        for dep in _local_imports(p, local_mods):
            if dep not in needed:
                needed.add(dep)
                frontier.append(dep)
    return needed


def violations() -> list[str]:
    """Files on disk that are neither production-reachable nor allowlisted.

    Test files (``test_*.py``) get a conditional exemption: a test is
    compliant when every *local* module it imports is itself compliant
    (production-reachable, allowlisted, or another compliant test). This
    keeps the suite that pins production behavior while making any test
    that imports research code fail the build alongside the code it tests.
    """
    reachable = production_reachable()
    allow = set(PRODUCTION_ALLOWLIST)
    compliant = {f"{s}.py" for s in reachable} | allow

    all_py = sorted(BACKEND.glob("*.py"))
    tests: dict[str, tuple[Path, set[str]]] = {}
    for p in all_py:
        if p.name in compliant:
            continue
        if p.stem.startswith("test_"):
            tests[p.name] = (p, {f"{m}.py" for m in _local_imports(p, local_module_names())})

    # Fixpoint: admit tests whose imports are all already compliant.
    changed = True
    while changed:
        changed = False
        for name, (_p, deps) in list(tests.items()):
            if deps <= compliant:
                compliant.add(name)
                del tests[name]
                changed = True

    return sorted({p.name for p in all_py} - compliant)


def local_module_names() -> set[str]:
    """Stems of all Python files currently in the backend directory."""
    return {p.stem for p in BACKEND.glob("*.py")}


def main() -> int:
    bad = violations()
    if not bad:
        print(f"OK: every file in {BACKEND.name}/ is production-reachable "
              f"or allowlisted ({len(production_reachable())} modules)")
        return 0
    print(f"FAIL: {len(bad)} file(s) in {BACKEND.name}/ are NOT reachable "
          f"from {ENTRY}.py and not allowlisted:")
    for name in bad:
        print(f"  {name}")
    print("\nResearch/ablation/diagnostic code must NOT be committed here — "
          "run it from a scratch workspace and, if a verdict matters, record "
          "it as an mlb_* artifact in data_delivery/ (see README.md). To "
          "legitimately add a production-adjacent file, extend "
          "PRODUCTION_ALLOWLIST with a justification.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Pin the in-process builder wrappers' non-fatal contract (2026-10-01).

master_pipeline runs build_il_stints / build_pbp_defense IN-PROCESS inside
``try: ... except Exception`` blocks whose comments promise NON-FATAL
failure. But both builders are CLI scripts at heart: build_il_stints raises
``SystemExit`` on every plausibility-gate trip ("refusing to write"), and
``SystemExit`` is a BaseException — it sails past ``except Exception``,
unwinds ``__main__``, and the pipeline ends right there: no Phase 2-3
features, no Phase 5 push, and (for the ``sys.exit(None)``-shaped paths)
exit code 0 — so the Kaggle wrapper printed "Pipeline completed" for a run
that delivered nothing (the 2026-10-01 nightly).

The wrappers must therefore catch SystemExit explicitly. This file parses
master_pipeline and pins that contract structurally, so a future refactor
of either wrapper cannot silently reopen the escape hatch.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# encoding pinned: master_pipeline.py is UTF-8 (em-dashes in prose) and the
# Windows default codec (cp1252) raises UnicodeDecodeError at collection.
TREE = ast.parse((BACKEND / "master_pipeline.py").read_text(encoding="utf-8"))


def _try_nodes_calling(*builder_names: str) -> list[ast.Try]:
    """Every Try block whose body calls one of the named builder mains."""
    hits: list[ast.Try] = []
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Try):
            continue
        for stmt in node.body:
            call = stmt.value if isinstance(stmt, ast.Expr) else stmt
            if (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id in builder_names):
                hits.append(node)
                break
    return hits


def _handler_catches(handler: ast.ExceptHandler, name: str) -> bool:
    t = handler.type
    if isinstance(t, ast.Name):
        return t.id == name
    if isinstance(t, ast.Tuple):
        return any(isinstance(e, ast.Name) and e.id == name for e in t.elts)
    return False


@pytest.mark.parametrize("builder",
                         ["_build_il_stints", "_build_pbp_defense"])
def test_builder_wrapper_catches_systemexit(builder: str) -> None:
    wrappers = _try_nodes_calling(builder)
    assert wrappers, f"no try-block wraps {builder}() — did the wrapper move?"
    for node in wrappers:
        assert any(_handler_catches(h, "SystemExit") for h in node.handlers), (
            f"the {builder}() wrapper must catch SystemExit explicitly — it "
            "is a BaseException, and a builder gate trip ('refusing to "
            "write') silently killed the 2026-10-01 run before Phase 5")


def test_builder_wrappers_still_catch_exception() -> None:
    # Guard the guard: the widening must not have REPLACED the plain
    # Exception handling (data errors remain non-fatal too).
    for node in _try_nodes_calling("_build_il_stints", "_build_pbp_defense"):
        assert any(_handler_catches(h, "Exception") for h in node.handlers)

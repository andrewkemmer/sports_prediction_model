"""Progress reporting for the NBA pipeline.  Display only, never behaviour.

The MLB backend earns an operator's patience by naming its work: a
``statcast_chunk_days: 60`` in the config banner, a ``Chunk: ... -> ...`` line
per slice, a banner per phase.  This module is the NBA half of that idiom, and
it is deliberately built so that drawing the bar cannot change what the run
does:

* It never wraps work in a retry, never catches an exception, never reorders
  an iterable, and never returns a value the caller uses.  The only thing it
  owns is a counter.
* ``tqdm`` is used when it is importable and falls back to the log line the
  pipeline already wrote when it is not, so the Kaggle notebook does not gain a
  dependency it does not already install.
* It is silent when the output is not a terminal (a Kaggle log, a pipe, a
  captured run), because a bar redrawn into a log file is just noise, and it is
  switched off outright by ``NBA_PROGRESS=0``.

The guardrail is the invariant a caller can rely on: with the bar on, off, or
unavailable, the pipeline returns exactly the same artifacts.  The only
difference is what the operator sees.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
from typing import Any, Iterable, Iterator, Sequence

logger = logging.getLogger("nba_progress")

ENV = "NBA_PROGRESS"
# Anything other than these reads as "on" when the variable is unset, and an
# explicit "0"/"off" always wins.  Default on: the bar is the feature.
OFF_WORDS = {"0", "false", "no", "off"}


def _flag() -> bool:
    raw = os.environ.get(ENV)
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() not in OFF_WORDS


def _drawable() -> bool:
    """True only when a bar would actually be seen.

    Kaggle and CI both capture stderr, and a bar written into a capture becomes
    a wall of carriage returns in the run log.  Refusing to draw there costs
    nothing and keeps the logs readable.
    """
    try:
        return bool(sys.stderr) and sys.stderr.isatty()
    except Exception:  # noqa: BLE001 - a broken stream is simply not drawable
        return False


def enabled() -> bool:
    """Whether bars should be drawn at all.  Cheap; safe to call in a loop."""
    return _flag() and _drawable()


def _tqdm() -> Any | None:
    """Return ``tqdm.tqdm`` if it is installed, else ``None``.

    Imported per bar rather than at module scope: the import is cheap, but
    resolving it lazily means a missing dependency is a ``None`` return and
    never an ImportError at pipeline start.
    """
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - absence is a supported state
        return None
    return tqdm


class _Counter:
    """The no-bar stand-in.  Counts, logs once, changes nothing else.

    It exists so callers can hold one object unconditionally: the same
    ``update``/``set_description``/``close`` calls land here when the bar is
    unavailable, and the run's *output* degrades to the plain log line the
    pipeline has always emitted rather than to silence.
    """

    def __init__(self, total: int | None, desc: str, unit: str) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.count = 0
        self._label = desc

    def update(self, n: int = 1) -> None:
        self.count += int(n)

    def set_description(self, text: str, **_kwargs: Any) -> None:
        self._label = text

    def set_postfix(self, text: str, **_kwargs: Any) -> None:
        # The postfix is a value, not a state transition: fold it into the
        # description so the final log line still says what the last item was.
        self._label = f"{self._label} {text}".strip()

    def close(self) -> None:
        suffix = f" of {self.total}" if self.total else ""
        unit = self.unit or "step"
        plural = "" if self.count == 1 else "s"
        logger.info("  %s: %d%s %s%s done", self._label, self.count, suffix,
                    unit, plural)


class _Bar:
    """A ``tqdm`` bar, or the counter, behind one interface."""

    def __init__(self, total: int | None, desc: str, unit: str) -> None:
        self._inner: Any | None = None
        self._counter = _Counter(total, desc, unit)
        if not enabled():
            return
        factory = _tqdm()
        if factory is None:
            return
        self._inner = factory(total=total, desc=desc, unit=unit or None,
                              leave=True, dynamic_ncols=True,
                              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                         "[{elapsed}<{remaining}] {postfix}")

    def update(self, n: int = 1) -> None:
        if self._inner is not None:
            self._inner.update(n)
        else:
            self._counter.update(n)

    def set_description(self, text: str, **_kwargs: Any) -> None:
        if self._inner is not None:
            self._inner.set_description(text, refresh=False)
        else:
            self._counter.set_description(text)

    def set_postfix(self, text: str, **_kwargs: Any) -> None:
        if self._inner is not None:
            self._inner.set_postfix_str(text, refresh=False)
        else:
            self._counter.set_postfix(text)

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()
        else:
            self._counter.close()


@contextlib.contextmanager
def track(total: int | None, desc: str = "NBA", unit: str = "step") -> Iterator[_Bar]:
    """A bar that lives exactly as long as the ``with`` block.

    The total is only used to draw; a caller that cannot know it in advance can
    pass ``None`` and update per item.  Closing is guaranteed on the way out,
    including on an exception, so a failed phase does not leave a half-drawn
    line behind in a captured log.
    """
    bar = _Bar(total, desc, unit)
    try:
        yield bar
    finally:
        bar.close()


def phases(names: Sequence[str], desc: str = "NBA pipeline") -> _Phases:
    """A phase bar for a straight-line run.

    ``run()`` is a sequence of steps that either happens or raises; it is not a
    loop to be wrapped.  This is the shape that fits: name the steps once, call
    ``advance()`` as each finishes.  A name that is not in the list is still
    allowed and simply advances the bar, because losing a step's name must
    never turn into losing the step itself.
    """
    return _Phases(names, desc)


class _Phases:
    def __init__(self, names: Sequence[str], desc: str) -> None:
        self._names = list(names)
        self._done = 0
        self._ctx = track(len(self._names), desc=desc, unit="phase")
        self._bar: _Bar = self._ctx.__enter__()

    def advance(self, name: str = "") -> None:
        label = name
        if not label:
            label = (self._names[self._done]
                     if self._done < len(self._names) else "finishing")
        self._done += 1
        self._bar.set_description(label)
        self._bar.update(1)

    def close(self) -> None:
        self._ctx.__exit__(None, None, None)

    def __enter__(self) -> "_Phases":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def wrap(items: Iterable[Any], total: int | None, desc: str,
         unit: str = "item") -> Iterator[Any]:
    """Yield ``items`` in order, ticking a bar as they go.

    Strictly a pass-through: the yielded sequence is the input sequence, in the
    input order, with the same elements.  A caller that needs the bar for
    cosmetics is getting cosmetics.
    """
    bar = _Bar(total, desc, unit)
    try:
        for item in items:
            yield item
            bar.update(1)
    finally:
        bar.close()

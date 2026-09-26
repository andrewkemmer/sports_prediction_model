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
* It never *draws* a bar when the output is not a terminal (a Kaggle log, a
  pipe, a captured run), because a bar redrawn into a log file is just noise,
  and it is switched off outright by ``NBA_PROGRESS=0``.  It still *reports*:
  the same count, rate and ETA go out as a log line on a heartbeat, so a
  captured run says what it is doing while it is doing it.

The guardrail is the invariant a caller can rely on: with the bar on, off, or
unavailable, the pipeline returns exactly the same artifacts.  The only
difference is what the operator sees.

That distinction is the whole reason the log lines are here, and it is a
distinction this module originally got wrong.  Bars are rightly suppressed when
stderr is not a terminal - but the code suppressed *everything* with them, so
on Kaggle, where stderr is captured and never a tty, a run that spent ten
minutes walking 1024 schedule days printed nothing at all until it finished.
Silence is the correct rendering of a bar.  It is not the correct rendering of
work in progress, and the operator watching a notebook cannot tell a hung cell
from a working one.  So the count is reported on a heartbeat either way, and
only the redrawing is conditional.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import Any, Callable, Iterable, Iterator, Sequence

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


def _eta(seconds: float) -> str:
    """A duration in the units a person waiting actually wants: ``4m12s``."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _write(text: str) -> None:
    """Print a display line, whatever the output stream can encode.

    MLB's banner is box-drawing characters and a check mark, and MLB runs on
    Kaggle where stdout is UTF-8.  This pipeline also runs on a Windows console
    whose default codec is cp1252, and there the first banner raised
    ``UnicodeEncodeError`` and took the run down on its very first line - a
    display-only feature killing a run that had not yet done any work.  A bar
    that cannot be drawn is not a reason to stop, so an unencodable glyph is
    replaced rather than raised.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, "replace").decode(encoding, "replace"))


def banner(text: str) -> None:
    """MLB's phase banner, verbatim: ``_banner`` in its ``master_pipeline``.

    ``print`` and not ``logger`` on purpose.  MLB's operator-facing run markers
    go to stdout and are the one thing guaranteed to be visible in a Kaggle
    notebook, a pipe, and a terminal alike; routing them through a log stream
    that a host may swallow is how a run ends up silent while it works.
    """
    _write(f"\n{'━' * 70}\n  {text}\n{'━' * 70}")


def ok(text: str) -> None:
    """MLB's per-phase result line: the check mark and what it produced.

    Present as a function rather than left as a ``print`` at each call site so
    there is exactly one place that knows about the glyph.
    """
    _write(f"  ✅ {text}")


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

    #: Longest silence between progress lines.  Long enough to stay legible in
    #: a captured log, short enough that a stalled run is obviously stalled
    #: rather than merely quiet.
    HEARTBEAT_SEC = 10.0

    def __init__(self, total: int | None, desc: str, unit: str,
                 show_rate: bool = True) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.show_rate = show_rate
        self.count = 0
        self._label = desc
        self._postfix = ""
        self._started = time.monotonic()
        self._last = self._started

    def update(self, n: int = 1) -> None:
        self.count += int(n)
        now = time.monotonic()
        if now - self._last >= self.HEARTBEAT_SEC:
            self._last = now
            logger.info("  %s", self._line(now))

    def _line(self, now: float | None = None) -> str:
        """One progress line: count, share, rate, ETA, and the current detail."""
        now = time.monotonic() if now is None else now
        rate = self.count / max(1e-9, now - self._started)
        head = self._label
        if self.total:
            head = f"{self._label}: {self.count}/{self.total} " \
                   f"({100.0 * self.count / self.total:.0f}%)"
            if self.show_rate and rate > 0 and self.count < self.total:
                head += f", eta {_eta((self.total - self.count) / rate)}"
        else:
            head += f": {self.count}"
        parts = [head]
        if self.show_rate:
            parts.append(f"{rate:.1f} {self.unit or 'step'}/s")
        if self._postfix:
            parts.append(self._postfix)
        return "  ".join(parts)

    def set_description(self, text: str, **_kwargs: Any) -> None:
        self._label = text

    def set_postfix(self, text: str, **_kwargs: Any) -> None:
        # A postfix is the CURRENT value, so it replaces.  Folding it into the
        # label instead produced a log line that grew with every item - the
        # 35-slice gap scan emitted one 700-character line naming all 35.
        self._postfix = text

    def close(self) -> None:
        suffix = f" of {self.total}" if self.total else ""
        unit = self.unit or "step"
        plural = "" if self.count == 1 else "s"
        tail = f" ({self._postfix})" if self._postfix else ""
        logger.info("  %s: %d%s %s%s done%s", self._label, self.count, suffix,
                    unit, plural, tail)


class _Bar:
    """A ``tqdm`` bar, or the counter, behind one interface."""

    def __init__(self, total: int | None, desc: str, unit: str,
                 show_rate: bool = True) -> None:
        self._inner: Any | None = None
        self._counter = _Counter(total, desc, unit, show_rate)
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
    def item(self, postfix: str | Callable[[], str] = "") -> Iterator[None]:
        """Mark one unit complete on the way out, however the block exits.

        Ticking at the *top* of a loop body is the obvious thing to write and it
        is wrong, in a way this only showed up once a real sweep was watched.
        The count then leads the work instead of following it, so the closing
        line of a sweep reports one fewer unit than actually completed and
        "N of N done" is printed while the Nth unit is still in flight. A
        1,024-day schedule sweep closed on "1023 fetched" and then summarised
        "1024 fetched"; a 609-game play-by-play sweep closed on "608 fetched"
        and then summarised "609 fetched". Two numbers for the same fact, in
        the same log, a few lines apart.

        Wrapping the body makes the correct order the easy one. ``continue``
        still ticks exactly once, a budget ``break`` taken before the block
        ticks not at all (that unit genuinely was not attempted), and a unit
        that fails still ticks, because a failure is a completed attempt and
        hiding it would make a broken sweep look like a shorter one.

        ``postfix`` may be a callable, evaluated on the way out so it can report
        counters the block has just updated.
        """
        try:
            yield
        finally:
            self.update(1)
            if postfix:
                self.set_postfix(postfix() if callable(postfix) else postfix)


@contextlib.contextmanager
def track(total: int | None, desc: str = "NBA", unit: str = "step",
          show_rate: bool = True) -> Iterator[_Bar]:
    """A bar that lives exactly as long as the ``with`` block.

    The total is only used to draw; a caller that cannot know it in advance can
    pass ``None`` and update per item.  Closing is guaranteed on the way out,
    including on an exception, so a failed phase does not leave a half-drawn
    line behind in a captured log.

    ``show_rate=False`` drops the per-second figure and the ETA.  They are
    honest for a loop of similar items and actively misleading for one that is
    not: ten phases where the first took 2 seconds and the walk-forward is
    about to take five minutes produces "0.0 phase/s, eta 6m03s" off a single
    sample, which reads as a measurement and is not one.
    """
    bar = _Bar(total, desc, unit, show_rate)
    try:
        yield bar
    finally:
        bar.close()


def phases(names: Sequence[str], desc: str = "NBA pipeline") -> _Phases:
    """A phase bar for a straight-line run.    ``run()`` is a sequence of steps that either happens or raises; it is not
    a loop to be wrapped.  This is the shape that fits: name the steps once, call
    ``advance()`` as each finishes.  A name that is not in the list is still
    allowed and simply advances the bar, because losing a step's name must
    never turn into losing the step itself.

    ``advance()`` labels the step it is entering, so a caller that calls it
    after the work reports the phase that just completed; the log line names
    the phase that is now running, which is the one an operator needs.

    Phases are wildly uneven in cost, so the rate and ETA are switched off: the
    count and the name of the phase now running are the useful signal, and an
    ETA computed from a single completed phase would be a guess wearing a
    decimal point.
    """
    return _Phases(names, desc)


class _Phases:
    def __init__(self, names: Sequence[str], desc: str) -> None:
        self._names = list(names)
        self._done = 0
        self._ctx = track(len(self._names), desc=desc, unit="phase",
                          show_rate=False)
        self._bar: _Bar = self._ctx.__enter__()

    def advance(self, name: str = "") -> None:
        label = name
        if not label:
            label = (self._names[self._done]
                     if self._done < len(self._names) else "finishing")
        self._done += 1
        self._bar.set_description(label)
        self._bar.update(1)
        # Every phase announces itself as it starts, not only as the bar
        # closes.  A bar cannot be read after the fact and a run that is four
        # minutes into its first phase has said nothing at all, which is
        # indistinguishable from a hang.
        logger.info("  phase %d/%d  %s", self._done, len(self._names), label)

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

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
  pipeline already wrote when it is not, so the notebook needs no dependency
  the backend does not already install.
* It draws the bar whenever ``tqdm`` is there - terminal, pipe or captured
  Kaggle cell alike - because that is the rule MLB runs under, and it is
  switched off outright by ``NBA_PROGRESS=0``.  The log line is not a second
  rendering of the bar, it is what is left when ``tqdm`` is absent.

The guardrail is the invariant a caller can rely on: with the bar on, off, or
unavailable, the pipeline returns exactly the same artifacts.  The only
difference is what the operator sees.

That distinction is the whole reason the log lines are here, and it is a
distinction this module originally got wrong twice.

First: bars were suppressed when stderr is not a terminal, but the code
suppressed *everything* with them, so on Kaggle, where stderr is captured and
never a tty, a run that spent ten minutes walking 1024 schedule days printed
nothing at all until it finished.  Silence is the correct rendering of a bar.
It is not the correct rendering of work in progress, and the operator watching
a notebook cannot tell a hung cell from a working one.  So the count is
reported on a heartbeat either way.

Second, and more recently: the bar itself was then withheld from a captured
stream on the theory that a redrawn bar in a log file is noise.  That is the
theory *tqdm* does not have, and it is why MLB's captured run is full of
working bars.  MLB's are not drawn by MLB: ``pybaseball.statcast`` wraps its
per-day sub-requests in ``tqdm(total=len(date_range))`` and constructs it with
stock defaults, and stock ``tqdm`` writes to a captured stream exactly as it
writes to a terminal.  The 60-day chunk, the ``0%|  | 0/46 [00:00<?, ?it/s]``
that becomes ``100%|...| 46/46 [00:52<00:00, 1.15s/it]``, the ``-> 164216
pitches`` after it - all of that survives into a Kaggle log for the same reason
this module now draws: the output is not asked whether it is a terminal.

So the gate is gone, and what is left of the old design is kept because it is
still needed.  ``tqdm`` is an optional import, and a backend that has to run
before the dependency lands should still say what it is doing; the heartbeat
counter is what it says.
"""
from __future__ import annotations

import contextlib
import functools
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


#: How the bar reads when the run is straight-line and the phases are wildly
#: uneven in cost, so a rate and an ETA computed off one or two samples would
#: read as a measurement and be nothing of the kind.  This is the default
#: ``tqdm`` format minus the rate; every other bar gets stock ``tqdm``.
NO_RATE_BAR = "{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}"


def enabled() -> bool:
    """Whether bars should be drawn at all.  Cheap; safe to call in a loop.

    True whenever the run is not switched off *and* ``tqdm`` is importable.
    Deliberately silent about whether the output is a terminal, because that is
    the question that made this backend's bars invisible on Kaggle while MLB's
    animated on exactly the same host.  See the module docstring.
    """
    return _flag() and _tqdm() is not None


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


@functools.lru_cache(maxsize=1)
def _tqdm() -> Any | None:
    """Return ``tqdm.tqdm`` if it is installed, else ``None``.

    Imported lazily rather than at module scope: a missing dependency is then a
    ``None`` return and never an ImportError at pipeline start.  Cached because
    ``enabled()`` is documented as safe to call in a loop and a repeated import
    lookup in a per-day sweep is a repeated sys.modules walk.
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
        if not _flag():
            return
        factory = _tqdm()
        if factory is None:
            return
        # Stock tqdm, the way ``pybaseball`` constructs it: no ``bar_format``,
        # no ``dynamic_ncols``, ``leave=True``, ``disable`` left at its own
        # default.  That is what puts
        # ``100%|...| 46/46 [00:52<00:00, 1.15s/it]`` in MLB's captured Kaggle
        # log, and a hand-rolled format is the one thing that would have made
        # this bar look nothing like the one it is imitating.  The postfix
        # rides along inside tqdm's own ``r_bar`` rather than needing a format
        # to place it.
        #
        # ``disable`` is worth naming because it is the whole ball game and it
        # is not what it looks like.  ``tqdm`` *can* suppress itself on a
        # non-terminal - ``std.py`` says ``if disable is None and
        # not file.isatty(): disable = True`` - but the parameter's default is
        # ``False``, not ``None``, so that branch is unreachable unless a caller
        # opts into it, and a stock bar writes into a pipe, a file and a
        # captured cell exactly as it writes to a terminal.  The old gate here
        # was therefore not reading tqdm's rule; it was inventing a stricter
        # one, and it is the only reason this backend's bars were invisible
        # where MLB's were not.
        #
        # ``position=0`` is the last of them and the least obvious.  Left to
        # itself tqdm stacks a bar *above* any bar already open, and rewinds the
        # cursor a line on every redraw to keep them apart.  On a terminal that
        # is invisible; in a capture it is an ``ESC[A`` and a blank line after
        # every refresh, which is the kind of dirt the capture rule exists to
        # keep out.  MLB never hits it because it pulls before it starts a
        # phase bar, so its bars are the only ones open and land at position 0
        # by default.  Here the ten-phase bar does span the ingest and the
        # three sweeps do run under it - but never at the same time, because
        # the phase bar redraws only when ``advance()`` is called and the
        # sweeps are all closed before it is.  Pinning every bar to position 0
        # states that instead of leaving it to a collision that does not happen.
        # If a future change ever does want two bars live at once, the cost of
        # this line is one overpainted row in a terminal, which is cheaper than
        # the log noise it buys back.
        kwargs: dict[str, Any] = {"position": 0}
        if not show_rate:
            kwargs["bar_format"] = NO_RATE_BAR
        self._inner = factory(total=total, desc=desc, unit=unit or None,
                              leave=True, **kwargs)

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
            # ``leave`` is tqdm's default, so a completed bar stays on the line
            # that drew it: in a captured log the run ends with a row of bars
            # that each reached 100%, which is the whole point of the format.
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

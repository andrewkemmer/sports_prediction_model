"""Read and normalize the pinned ``wyattowalsh/basketball`` NBA warehouse.

The Kaggle export is a star-schema warehouse rather than an API client.  This
module accepts its documented DuckDB, SQLite, Parquet, and CSV layouts, joins
the game identity/result tables, normalizes team IDs to abbreviations, and
writes only a derived cache outside the repository.  No live feed or alternate
source is silently substituted.
"""
from __future__ import annotations

import glob as _glob
import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from backend import config
except ImportError:
    import config

logger = logging.getLogger(__name__)

GAME_TABLES = (
    "dim_game", "fact_game_result", "fact_scoreboard_v3", "games",
    "stg_league_game_log", "fact_game",
    # Dataset version 238 also ships the pre-star-schema ``game`` export.
    # It is a complete game/result row, so it is a safe final fallback.
    "game", "game_summary",
)
# The Kaggle export currently publishes the star-schema box-score families
# under the names below.  Older mirrors used ``fact_box_score_team`` and
# several deployments expose the staging names instead.  Keep the complete,
# explicit list here rather than relying on a prefix glob: silently selecting
# the first table would make a source-layout change look like valid data.
TEAM_BOX_TABLES = (
    "fact_box_score_traditional_team",
    "fact_box_score_team",
    "fact_box_score_advanced_team",
    "fact_box_score_hustle_team",
    "fact_box_score_defensive_team",
    "fact_box_score_four_factors_team",
    "fact_box_score_scoring_team",
    "fact_box_score_usage_team",
    "fact_box_score_misc_team",
    "fact_box_score_player_track_team",
    "fact_team_game", "fact_team_game_log", "team_boxscores",
    "stg_box_score_traditional_team", "stg_box_score_team",
    "stg_box_score_advanced_team", "stg_box_score_hustle_team",
    "stg_box_score_defensive_team", "stg_box_score_four_factors_team",
)
PLAYER_TABLES = (
    "fact_box_score_traditional_player",
    "fact_player_game_traditional", "player_game_stats",
    "fact_box_score_advanced_player",
    "fact_player_game_advanced",
    "fact_box_score_hustle_player", "fact_box_score_defensive_player",
    "fact_box_score_four_factors_player", "fact_box_score_player_track_player",
    "fact_player_game", "stg_box_score_traditional_player",
    "stg_box_score_advanced_player", "stg_box_score_hustle_player",
    "stg_box_score_defensive_player", "stg_box_score_four_factors_player",
)
TEAM_TABLES = (
    "dim_team", "dim_team_history", "dim_team_extended", "fact_static_teams",
    "stg_static_teams", "stg_team_info_common", "teams",
    # Legacy v238 names.  ``team`` has the stable id/abbreviation/full_name
    # trio used by the raw game export; the other tables are harmless
    # supplementary dimensions.
    "team", "team_info_common", "team_details", "team_history",
)
PLAYER_DIM_TABLES = (
    "dim_player", "dim_all_players", "fact_static_players",
    "stg_player_info", "raw_common_player_info",
    "player", "common_player_info",
)

# CSV has no schema metadata, so pandas otherwise guesses that canonical
# zero-padded NBA game/team/player identifiers are integers and irreversibly
# drops their leading zeroes.  Keep identity columns textual at the reader
# boundary; numeric normalization remains explicit in the feature layer.
_CSV_ID_COLUMNS = frozenset({
    "id", "game_id", "gameid", "game_pk", "person_id",
    "player_id", "playerid", "player_sk", "team_id", "teamid",
    "home_team_id", "visitor_team_id", "away_team_id", "franchise_id",
    "from_team_id", "to_team_id",
})

# Kaggle mounts an attached dataset below a generated directory name, and
# downloaded archives may add one or more wrapper directories.  Keep source
# discovery based on stable warehouse filenames rather than a brittle
# ``/kaggle/input/basketball`` assumption.
_WAREHOUSE_SQL_SUFFIXES = frozenset({".duckdb", ".db", ".sqlite"})
_WAREHOUSE_MARKER_NAMES = frozenset({
    "dim_game.csv", "dim_game.parquet",
    "fact_game_result.csv", "fact_game_result.parquet",
    # Legacy v238 export markers.  The SQL bundle in that version can contain
    # an empty DuckDB file, so the CSV mirror must remain discoverable.
    "game.csv", "game.parquet", "game_summary.csv", "game_summary.parquet",
    "team.csv", "team.parquet",
})
_WAREHOUSE_MARKER_DIRS = frozenset({
    "dim_game", "fact_game_result", "game", "team",
})
# Columns a source-audit probe accepts as a game's date.  Compared after
# folding, so they are lower case.
_AUDIT_DATE_COLUMNS = frozenset({
    "game_date", "gameday", "date", "game_datetime", "tipoff", "game_date_time",
})
KAGGLE_AUTO_DOWNLOAD_ENV = "NBA_KAGGLE_AUTO_DOWNLOAD"
KAGGLE_DOWNLOAD_DIR_ENV = "NBA_KAGGLE_DOWNLOAD_DIR"


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Fold a warehouse's column spelling onto this module's alias vocabulary.

    The pinned bundle mirrors raw ``nba_api`` payloads, whose keys are upper
    case (``SEASON_YEAR``, ``TEAM_ABBREVIATION_HOME``).  Every alias used below
    is lower snake case, so an un-folded frame matches nothing and normalizes
    into blank columns instead of failing loudly.  Folding is a no-op for the
    canonical lower-case exports, and two spellings of one field are suffixed
    deterministically rather than overwriting each other.
    """
    names: list[str] = []
    seen: dict[str, int] = {}
    for column in frame.columns:
        key = re.sub(r"[^0-9a-z]+", "_", str(column).strip().lower()).strip("_")
        key = key or "column"
        count = seen.get(key, 0) + 1
        seen[key] = count
        names.append(key if count == 1 else f"{key}_{count}")
    if names != [str(column) for column in frame.columns]:
        frame.columns = pd.Index(names)
    return frame


def _match_table_names(available: Iterable[str],
                       names: Iterable[str]) -> dict[str, str]:
    """Map requested table names onto the catalog's actual spelling.

    Exported catalogs are not consistent about case, and these lookups run on
    Linux, so a case-only mismatch would otherwise look like a missing table.
    """
    catalog = {str(item).strip().lower(): str(item) for item in available}
    return {name: catalog[name.strip().lower()]
            for name in names if name.strip().lower() in catalog}


def _catalog_has_known_table(names: Iterable[str]) -> bool:
    """Report whether a SQL catalog holds a table this reader understands."""
    known = {name.lower() for name in
             GAME_TABLES + TEAM_TABLES + PLAYER_DIM_TABLES}
    return bool({str(name).lower() for name in names} & known)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV while preserving warehouse identity spelling."""
    header = pd.read_csv(path, nrows=0)
    identity = {
        column: str for column in header.columns
        if str(column).strip().lower() in _CSV_ID_COLUMNS
    }
    return _normalize_columns(pd.read_csv(path, dtype=identity))


@dataclass
class Warehouse:
    games: pd.DataFrame
    team_stats: pd.DataFrame
    player_stats: pd.DataFrame
    team_names: dict[str, str]
    manifest: dict[str, Any]


def _first(df: pd.DataFrame, *names: str, default: Any = None) -> Any:
    for name in names:
        if name in df.columns:
            return name
    return default


def _column(df: pd.DataFrame, *names: str, default: Any = np.nan) -> pd.Series:
    name = _first(df, *names)
    if name is not None:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _coalesce_column(df: pd.DataFrame, *names: str,
                     default: Any = np.nan) -> pd.Series:
    """Return the first non-blank value across alias columns, row-wise.

    Warehouse dimensions and result facts often expose the same logical field
    under different names.  Selecting the first *existing* column is not enough:
    an identity table may contain ``home_score`` as an all-null placeholder
    while ``fact_game_result`` carries ``pts_home``.  This helper is the
    normalization boundary for that layout and keeps the result join causal and
    deterministic.
    """
    out = pd.Series([default] * len(df), index=df.index, dtype=object)
    filled = pd.Series(False, index=df.index)
    for name in names:
        if name not in df.columns:
            continue
        values = df[name]
        blank = values.isna()
        if values.dtype == object or pd.api.types.is_string_dtype(values):
            blank = blank | values.astype("string").str.strip().eq("").fillna(True)
        take = (~filled) & (~blank)
        if take.any():
            out.loc[take] = values.loc[take]
            filled.loc[take] = True
    return out


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _date(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce", utc=True)
    return values.dt.tz_localize(None)


def _season(series: pd.Series) -> pd.Series:
    """Normalize warehouse season-year values to a starting year.

    ``2024-25``, ``2024``, and compact season IDs such as ``002024`` all map
    to 2024.  Missing values remain NaN and are rejected by coverage gates.
    """
    text = series.astype("string").str.strip()
    four = text.str.extract(r"((?:19|20)\d{2})", expand=False)
    compact = text.str.extract(r"^(\d{4})", expand=False)
    numeric = pd.to_numeric(text, errors="coerce")
    out = numeric.where(numeric.between(1900, 2100))
    out = out.fillna(pd.to_numeric(four, errors="coerce"))
    compact = pd.to_numeric(compact, errors="coerce")
    compact = compact.where(compact >= 1900, compact + 2000)
    out = out.fillna(compact.where(compact.between(1900, 2100)))
    # The legacy NBA API encodes seasons as a five-digit value: the final
    # four digits are the calendar start year (22024 -> 2024, 21946 -> 1946).
    five_digit = text.str.fullmatch(r"\d{5}", na=False)
    five_year = pd.to_numeric(
        text.str.extract(r"^\d(\d{4})$", expand=False), errors="coerce")
    out = out.mask(five_digit, five_year)
    return out.astype(float)


def _season_from_date(dates: pd.Series) -> pd.Series:
    """Derive a season start year from a game date (July-June boundary)."""
    if not isinstance(dates, pd.Series):
        return pd.Series(dtype=float)
    year = pd.to_numeric(dates.dt.year, errors="coerce")
    before_july = pd.to_numeric(dates.dt.month, errors="coerce") < 7
    return (year - before_july.astype(float)).astype(float)


def _game_id(value: Any) -> str:
    """Canonicalize scalar IDs without destroying warehouse identity.

    NBA game IDs are commonly ten-character strings such as ``0022400001``.
    Converting them through ``float`` silently drops their leading zeroes, so
    only an actual floating-point spelling (for example ``22400001.0``) is
    normalized; otherwise the source token is preserved verbatim.
    """
    text = _text(value)
    if not text:
        return ""
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def _source_search_roots() -> list[Path]:
    """Return deterministic local/Kaggle locations for source discovery."""
    return [
        config.CACHE_DIR / "source", config.CACHE_DIR,
        Path("/kaggle/input"), Path("/kaggle/working/nba-warehouse"),
    ]


def _warehouse_marker_files(root: Path) -> list[Path]:
    """Find SQL/table markers below a candidate source root.

    The canonical Parquet/CSV exports may be partitioned as
    ``parquet/dim_game/season_year=YYYY/*.parquet`` rather than a single
    ``dim_game.parquet`` file, so table directories are valid markers too.
    """
    if root.is_file():
        return [root] if _is_warehouse_file(root) else []
    if not root.is_dir():
        return []
    found: dict[str, Path] = {}
    patterns = (
        "nba.duckdb", "nba.sqlite", "*.duckdb", "*.db", "*.sqlite",
        "dim_game.csv", "dim_game.parquet",
        "fact_game_result.csv", "fact_game_result.parquet",
        "game.csv", "game.parquet", "game_summary.csv", "game_summary.parquet",
        "team.csv", "team.parquet",
    )
    for pattern in patterns:
        try:
            matches = root.rglob(pattern)
            for path in matches:
                if path.is_file() and _is_warehouse_file(path):
                    found[str(path)] = path
        except OSError:
            continue
    try:
        for pattern in _WAREHOUSE_MARKER_DIRS:
            for path in root.rglob(pattern):
                if (path.is_dir()
                        and any(child.is_file() for child in path.rglob("*"))):
                    found[str(path)] = path
    except OSError:
        pass
    return sorted(found.values(), key=lambda p: (len(p.parts), str(p)))


def _is_warehouse_file(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name.lower()
    return (name in {"nba.duckdb", "nba.sqlite"}
            or path.suffix.lower() in _WAREHOUSE_SQL_SUFFIXES
            or name in _WAREHOUSE_MARKER_NAMES)


def _sql_table_names(path: Path) -> set[str]:
    """Return a SQL catalog's table names without reading table data.

    Version 238 contains a valid 12 KB DuckDB file with zero tables alongside
    the populated SQLite export.  A catalog probe lets source discovery skip
    that empty file without making the downloader download anything again.
    Failures are intentionally treated as an unknown catalog; the caller's
    deterministic file fallback still supports lightweight test fixtures.
    """
    suffix = path.suffix.lower()
    if suffix not in _WAREHOUSE_SQL_SUFFIXES:
        return set()
    if suffix == ".duckdb":
        try:
            import duckdb
            con = duckdb.connect(str(path), read_only=True)
            try:
                return {str(row[0]) for row in con.execute("SHOW TABLES").fetchall()}
            finally:
                con.close()
        except Exception:  # noqa: BLE001
            return set()
    if suffix in {".sqlite", ".db"}:
        try:
            import sqlite3
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                return {
                    str(row[0]) for row in con.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
            finally:
                con.close()
        except Exception:  # noqa: BLE001
            return set()
    return set()


def _ordered_sql_candidates(paths: Iterable[Path]) -> list[Path]:
    """Prefer a populated DuckDB, then the largest populated SQL fallback."""
    decorated: list[tuple[bool, bool, int, str, Path]] = []
    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        names = _sql_table_names(path)
        populated = _catalog_has_known_table(names)
        is_duckdb = path.suffix.lower() == ".duckdb"
        decorated.append((populated, is_duckdb, size, str(path), path))
    return [item[-1] for item in sorted(
        decorated, key=lambda item: (not item[0], not item[1], -item[2], item[3]))]


def _warehouse_root_for_marker(marker: Path, search_root: Path) -> Path:
    """Choose the dataset directory rather than a parquet/csv leaf folder."""
    # Partition directories (for example ``season_year=2024``) sit below
    # the export root, so inspect ancestors for the format directory first.
    candidate = marker.parent
    for ancestor in (candidate, *candidate.parents):
        if ancestor.name.lower() in {"parquet", "csv"}:
            return ancestor.parent if ancestor != search_root else search_root
    generic = {"data", "warehouse", "export", "tables"}
    while candidate != search_root and candidate.name.lower() in generic:
        candidate = candidate.parent
    if candidate != search_root:
        return candidate
    # A marker directly below the search root means the root itself is the
    # dataset (e.g. a caller passed /kaggle/input rather than its slug).
    return search_root


def discover_warehouse(roots: Iterable[str | Path] | str | Path | None = None
                       ) -> Path | None:
    """Resolve a usable NBA warehouse from cache, Kaggle input, or download.

    The function is intentionally read-only and returns the narrowest useful
    root: an SQL file when available, otherwise the directory containing the
    Parquet/CSV export.  It is used by the notebook before invoking the
    pipeline, so a missing attachment fails with an actionable source error
    instead of passing the literal string ``"None"`` downstream.
    """
    if roots is None:
        roots = _source_search_roots()
    elif isinstance(roots, (str, Path)):
        roots = [roots]
    seen: set[str] = set()
    for raw in roots:
        root = Path(raw).expanduser()
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        markers = _warehouse_marker_files(root)
        if not markers:
            continue
        sql = [p for p in markers
               if p.is_file() and p.suffix.lower() in _WAREHOUSE_SQL_SUFFIXES]
        if sql:
            ordered = _ordered_sql_candidates(sql)
            # Prefer a catalog containing a known NBA table.  This skips the
            # empty DuckDB shipped beside v238's populated SQLite database.
            for candidate in ordered:
                if _catalog_has_known_table(_sql_table_names(candidate)):
                    return candidate
            # If the SQL catalog is empty/unknown but a CSV/Parquet mirror is
            # present, use the mirror rather than handing the caller a dead
            # SQL file that cannot be introspected by the reader.
            non_sql = [p for p in markers
                       if p.is_file() and p.suffix.lower() not in _WAREHOUSE_SQL_SUFFIXES]
            if non_sql:
                return _warehouse_root_for_marker(non_sql[0], root)
            return ordered[0]
        return _warehouse_root_for_marker(markers[0], root)
    return None


def _kaggle_runtime() -> bool:
    """Return whether this process is running in a Kaggle workspace."""
    return bool(
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").strip()
        or (Path("/kaggle").is_dir() and Path("/kaggle/working").is_dir())
    )


def _auto_download_enabled() -> bool:
    """Apply the explicit auto-download override, or Kaggle's default."""
    raw = os.environ.get(KAGGLE_AUTO_DOWNLOAD_ENV, "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    return _kaggle_runtime()


def _download_target() -> Path:
    configured = os.environ.get(KAGGLE_DOWNLOAD_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    if _kaggle_runtime():
        return Path("/kaggle/working/nba-warehouse")
    return config.CACHE_DIR / "source"


def _extract_downloaded_archives(target: Path) -> None:
    """Extract Kaggle ZIPs without permitting paths outside ``target``."""
    destination = target.resolve()
    for archive in sorted(target.rglob("*.zip")):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                member_path = (target / member.filename).resolve()
                try:
                    member_path.relative_to(destination)
                except ValueError as exc:
                    raise RuntimeError(
                        f"Kaggle archive contains an unsafe path: {member.filename}"
                    ) from exc
            bundle.extractall(target)


def pull_kaggle_warehouse(target: str | Path | None = None,
                          dataset_ref: str | None = None,
                          version: str | int | None = None) -> Path:
    """Download and resolve the pinned NBA Kaggle warehouse.

    This is an ingestion-side operation used by the Kaggle orchestration
    path.  The core model modules never import the Kaggle client; local
    callers can continue to pass a local warehouse to ``load_dataset``.
    """
    ref = str(dataset_ref or config.NBA_DATASET_REF)
    pinned_version = str(version or config.NBA_DATASET_VERSION)
    if pinned_version != config.NBA_DATASET_VERSION:
        raise ValueError(
            f"NBA Kaggle dataset version must remain {config.NBA_DATASET_VERSION}, "
            f"got {pinned_version}"
        )
    # Kaggle CLI 2.x carries a dataset version as part of the dataset
    # reference (``owner/name/<version>``).  ``-v`` is the CLI's global
    # version flag, not a dataset-download option, so passing it here makes
    # argparse fail before any data is fetched.
    versioned_ref = f"{ref.rstrip('/')}/{pinned_version}"
    root = Path(target).expanduser() if target is not None else _download_target()
    existing = discover_warehouse([root])
    if existing is not None:
        return existing

    root.mkdir(parents=True, exist_ok=True)
    executable = shutil.which("kaggle")
    if executable:
        command = [executable]
    elif importlib.util.find_spec("kaggle") is not None:
        command = [sys.executable, "-m", "kaggle"]
    else:
        raise RuntimeError(
            "Kaggle download requested but the Kaggle CLI is unavailable. "
            "Install the optional nba-backend/backend/requirements-kaggle.txt "
            "dependencies or attach wyattowalsh/basketball version 238."
        )
    command.extend([
        "datasets", "download", "-d", versioned_ref,
        "--unzip", "-p", str(root),
    ])
    logger.info("Downloading NBA Kaggle warehouse %s version %s to %s",
                ref, pinned_version, root)
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Failed to download {ref} version {pinned_version} to {root}: {exc}"
        ) from exc
    _extract_downloaded_archives(root)
    resolved = discover_warehouse([root])
    if resolved is None:
        files = sorted(str(path.relative_to(root))
                       for path in root.rglob("*") if path.is_file())
        preview = ", ".join(files[:20]) or "no files"
        raise RuntimeError(
            "Kaggle download completed but no supported NBA warehouse was "
            f"found under {root}: {preview}"
        )
    return resolved


def resolve_source(source: str | Path | None = None,
                   allow_download: bool = False) -> Path | None:
    """Resolve an explicit, mounted, cached, or optional Kaggle source."""
    if isinstance(source, str) and source.strip().lower() in {"none", "null"}:
        source = None
    if source is not None:
        candidate = Path(source).expanduser()
        return discover_warehouse([candidate]) or candidate
    root = _source_root()
    if root is not None and root.exists():
        return root
    if allow_download and _auto_download_enabled():
        return pull_kaggle_warehouse()
    return None


def _source_root() -> Path | None:
    raw = (os.environ.get("NBA_KAGGLE_DATASET_PATH")
           or os.environ.get("NBA_SOURCE_PATH")
           or os.environ.get("NBA_DATA_PATH") or "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        return discover_warehouse([candidate]) or candidate
    # A mounted/cached bundle must still go through discovery: v238 stores an
    # empty ``nba.duckdb`` beside the populated database, so returning the
    # first existing filename would hand the reader the dead file.
    for directory in (config.CACHE_DIR / "source", config.CACHE_DIR):
        if not directory.is_dir():
            continue
        resolved = discover_warehouse([directory])
        if resolved is not None:
            return resolved
    return discover_warehouse(_source_search_roots())


def _read_sql_tables(path: Path, names: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Read a DuckDB or SQLite warehouse without changing the source.

    DuckDB is the declared production reader.  The stdlib SQLite fallback
    keeps small/offline exports usable and, importantly, gives tests a real
    SQL-path fixture without requiring a DuckDB installation.
    """
    names = tuple(names)
    try:
        import duckdb
    except ImportError:
        duckdb = None
    out: dict[str, pd.DataFrame] = {}
    # A v238 SQLite file is already a SQLite database; do not ask DuckDB to
    # open it before the stdlib fallback merely to rediscover its type.
    if duckdb is not None and path.suffix.lower() != ".sqlite":
        try:
            con = duckdb.connect(str(path), read_only=True)
            try:
                available = {str(row[0]) for row in con.execute("SHOW TABLES").fetchall()}
                resolved = _match_table_names(available, names)
                for name, actual in resolved.items():
                    escaped = actual.replace('"', '""')
                    out[name] = _normalize_columns(
                        con.execute(f'SELECT * FROM "{escaped}"').fetch_df())
            finally:
                con.close()
            if out:
                return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read DuckDB warehouse %s: %s", path, exc)

    if path.suffix.lower() not in {".sqlite", ".db"}:
        return out
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            available = {
                str(row[0]) for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            for name, actual in _match_table_names(available, names).items():
                # Names are internal constants, but quote them anyway so
                # a future alias containing a quote cannot become SQL.
                escaped = actual.replace('"', '""')
                out[name] = _normalize_columns(pd.read_sql_query(
                    f'SELECT * FROM "{escaped}"', con
                ))
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read SQLite warehouse %s: %s", path, exc)
    return out


def _read_table(root: Path, names: tuple[str, ...]) -> pd.DataFrame:
    """Backward-compatible single-table reader used by small fixtures."""
    frames = _read_tables(root, names)
    return next(iter(frames.values()), pd.DataFrame())


def _read_tables(root: Path, names: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Read each available named table from a warehouse export.

    Kaggle's canonical bundle places ``nba.duckdb`` / ``nba.sqlite`` at the
    extracted root, with Parquet and CSV mirrors below it.  Prefer the SQL
    catalog when present, then fill names unavailable there from partitioned
    or flat exports.  Both Parquet and CSV paths may contain an extra
    ``season_year=YYYY`` partition directory, hence recursive globbing.
    """
    if root.is_file():
        if root.suffix.lower() in {".duckdb", ".db", ".sqlite"}:
            return _read_sql_tables(root, names)
        try:
            frame = (pd.read_parquet(root) if root.suffix.lower() == ".parquet"
                     else _read_csv(root))
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read %s: %s", root, exc)
            return {}
        return {names[0]: _normalize_columns(frame)} if names else {}

    found: dict[str, pd.DataFrame] = {}
    sql_candidates = [root / "nba.duckdb", root / "nba.sqlite"]
    if not any(path.is_file() for path in sql_candidates):
        sql_candidates.extend(sorted(
            path for path in root.glob("*")
            if path.is_file() and path.suffix.lower() in {".duckdb", ".db", ".sqlite"}
        ))
    for sql_path in sql_candidates:
        if sql_path.is_file():
            found.update(_read_sql_tables(sql_path, names))
            if len(found) == len(names):
                break
    for name in names:
        if name in found:
            continue
        patterns = [
            root / f"{name}.parquet", root / f"{name}.csv",
            root / "parquet" / f"{name}.parquet",
            root / "parquet" / name / "**" / "*.parquet",
            root / "csv" / f"{name}.csv",
            root / "csv" / name / "**" / "*.csv",
            root / name / "**" / "*.parquet",
            root / name / "**" / "*.csv",
        ]
        matches: list[Path] = []
        for pattern in patterns:
            if any(ch in str(pattern) for ch in "*?["):
                matches.extend(Path(match) for match in sorted(
                    _glob.glob(str(pattern), recursive=True)))
            elif Path(pattern).exists():
                matches.append(Path(pattern))
        matches = list(dict.fromkeys(matches))
        if not matches:
            # Exported bundles are not consistent about file-name case and this
            # lookup runs on case-sensitive filesystems, so fall back to a
            # stem index before declaring a table absent.
            matches = _match_export_files(root, name)
        frames: list[pd.DataFrame] = []
        for path in matches:
            try:
                frames.append(_normalize_columns(
                    pd.read_parquet(path)
                    if path.suffix.lower() == ".parquet"
                    else _read_csv(path)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not read %s: %s", path, exc)
        if frames:
            found[name] = pd.concat(frames, ignore_index=True)
    return found


def _mirror_root(root: Path) -> Path | None:
    """Return the bundle directory that also ships a flat export mirror.

    A resolved SQL file is one representation of a bundle that also publishes
    Parquet and CSV copies.  Source discovery must pick a single populated
    catalog, but picking one file is not the same as trusting it alone: the
    pinned bundle's SQL copy and its flat export can cover different seasons.
    """
    if not root.is_file():
        return None
    parent = root.parent
    try:
        has_mirror = any(
            child.is_dir() and child.name.lower() in {"parquet", "csv"}
            for child in parent.iterdir())
    except OSError:
        return None
    return parent if has_mirror else None


def _dedupe_key(frame: pd.DataFrame, name: str) -> tuple[str, ...] | None:
    """Row identity for a table, or ``None`` when it cannot be trusted.

    A team box-score table holds one row per team per game, so collapsing it on
    ``game_id`` alone would silently discard every team but one.  When the full
    identity is unavailable the caller keeps a single copy instead of merging.
    """
    if name in TEAM_BOX_TABLES:
        wanted = ("game_id", "team")
    elif name in PLAYER_TABLES:
        wanted = ("game_id", "player_id")
    else:
        wanted = ("game_id",)
    if any(column not in frame.columns for column in wanted):
        return None
    return wanted


def _recency(frame: pd.DataFrame) -> pd.Series:
    """Order rows by coverage so the freshest copy of a game survives a merge."""
    if "gameday" in frame.columns:
        values = pd.to_datetime(frame["gameday"], errors="coerce", utc=True)
        return values.astype("int64", errors="ignore").fillna(0)
    if "season" in frame.columns:
        return pd.to_numeric(frame["season"], errors="coerce").fillna(0)
    return pd.Series(np.zeros(len(frame)), index=frame.index)


def _merge_representations(primary: pd.DataFrame, secondary: pd.DataFrame,
                           name: str) -> pd.DataFrame:
    """Union two copies of one table, keeping the freshest row per identity."""
    key = _dedupe_key(primary, name) or _dedupe_key(secondary, name)
    # A key that repeats inside one copy marks sibling rows this reader cannot
    # tell apart, so collapsing on it would delete data.  Keep a single copy.
    untrusted = (key is None
                 or primary.duplicated(subset=list(key)).any()
                 or secondary.duplicated(subset=list(key)).any())
    if untrusted:
        logger.warning(
            "NBA table %s exists in two representations without a unique row "
            "identity; keeping the larger copy (%d vs %d rows)",
            name, len(primary), len(secondary))
        return primary if len(primary) >= len(secondary) else secondary
    out = pd.concat([primary, secondary], ignore_index=True)
    out = out.assign(**{"__nba_recency__": _recency(out)})
    out = out.sort_values("__nba_recency__", kind="stable")
    return (out.drop_duplicates(subset=list(key), keep="last")
            .drop(columns="__nba_recency__").reset_index(drop=True))


def _log_table(name: str, origin: str, frame: pd.DataFrame | None) -> None:
    """Report what was read, from where, and how far it reaches."""
    if frame is None or frame.empty:
        return
    seasons = _season(_coalesce_column(
        frame, "season_year", "season", "seasonId", "season_id"))
    years = sorted(set(seasons.dropna().tolist()))
    span = f"{years[0]:g}-{years[-1]:g}" if years else "unknown"
    for column in ("gameday", "game_date", "gameDate"):
        if column in frame.columns:
            dates = pd.to_datetime(frame[column], errors="coerce", utc=True)
            if dates.notna().any():
                span += f", dates {dates.min().date()}..{dates.max().date()}"
            break
    logger.info("NBA table %s read from %s: %d rows, seasons %s",
                name, origin, len(frame), span)


def _raw_game_ids(frame: pd.DataFrame) -> set[str]:
    """Normalized game ids of a raw table, for representation comparisons."""
    if frame is None or frame.empty:
        return set()
    ids = _coalesce_column(frame, "game_id", "gameId", "game_pk", "gameid", "id")
    return {value for value in ids.map(_game_id).astype(str) if value}


def _select_result(game_frames: dict[str, pd.DataFrame],
                   identity_name: str | None,
                   identity: pd.DataFrame | None) -> pd.DataFrame | None:
    """Choose a result/scoreboard table that actually covers the identity rows.

    When a bundle exposes a current game identity beside a stale result table,
    joining them would blank the scores of every uncovered game and quietly
    drop it from the schedule.  A result table from a different source is
    therefore only accepted when it covers the identity games.
    """
    candidates = ("fact_game_result", "fact_scoreboard_v3", "stg_league_game_log",
                  "game", "game_summary")
    identity_ids = _raw_game_ids(identity)
    for name in candidates:
        frame = game_frames.get(name)
        if frame is None or frame.empty:
            continue
        if name == identity_name or not identity_ids:
            return frame
        coverage = len(_raw_game_ids(frame) & identity_ids) / len(identity_ids)
        if coverage >= 0.5:
            return frame
        logger.warning(
            "NBA result table %s covers only %.0f%% of the identity games; "
            "ignoring it so uncovered games keep their own scores", name,
            coverage * 100)
    return None


def _read_tables_merged(root: Path, names: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Read each table from every representation the bundle provides.

    Source discovery returns one populated catalog because a bundle can ship an
    empty primary file.  Reading is deliberately wider than discovery: a table
    present in two representations is merged on its row identity so the newest
    rows win, and every choice is logged.
    """
    primary = _read_tables(root, names)
    mirror = _mirror_root(root)
    if mirror is None:
        for name in names:
            _log_table(name, "sql" if root.is_file() else "export",
                       primary.get(name))
        return primary
    secondary = _read_tables(mirror, names)
    merged: dict[str, pd.DataFrame] = {}
    for name in names:
        left, right = primary.get(name), secondary.get(name)
        if left is None or left.empty:
            chosen, origin = right, "mirror"
        elif right is None or right.empty:
            chosen, origin = left, "sql"
        else:
            chosen, origin = _merge_representations(left, right, name), "sql+mirror"
        if chosen is not None and not chosen.empty:
            merged[name] = chosen
        _log_table(name, origin, chosen)
    return merged


def _match_export_files(root: Path, name: str) -> list[Path]:
    """Find a table's export by case-insensitive stem, Parquet before CSV."""
    if not root.is_dir():
        return []
    index: dict[str, list[Path]] = {}
    try:
        candidates = root.rglob("*")
    except OSError:
        return []
    for path in candidates:
        if path.is_file() and path.suffix.lower() in {".parquet", ".csv"}:
            index.setdefault(path.stem.lower(), []).append(path)
    matches = index.get(name.strip().lower(), [])
    return sorted(matches, key=lambda path: (path.suffix.lower() != ".parquet",
                                             str(path)))


def _load_team_maps(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(raw team key -> abbreviation, abbreviation -> full name)``."""
    frames = _read_tables_merged(root, TEAM_TABLES)
    raw_to_abbr: dict[str, str] = {}
    abbr_to_name: dict[str, str] = {}
    for df in frames.values():
        if df.empty:
            continue
        id_col = _first(df, "team_id", "id", "franchise_id", "from_team_id",
                        "to_team_id")
        abbr_col = _first(df, "team_abbreviation", "team_abbr", "abbreviation",
                          "team", "team_code")
        name_col = _first(df, "team_name", "full_name", "name", "nickname")
        city_col = _first(df, "city", "team_city")
        for _, row in df.iterrows():
            abbr = config.normalize_team_abbr(_text(row.get(abbr_col)))
            # A dimension may use the numeric ID in the team column.
            if not abbr and id_col is not None:
                abbr = config.normalize_team_abbr(row.get(id_col))
            if not abbr:
                continue
            display = _text(row.get(name_col)) if name_col else ""
            if not display and city_col:
                display = _text(row.get(city_col))
            if not display:
                display = abbr
            abbr_to_name[abbr] = display
            if id_col is not None:
                raw_to_abbr[_text(row.get(id_col))] = abbr
            raw_to_abbr[abbr] = abbr
            raw_to_abbr[display.upper()] = abbr
    return raw_to_abbr, abbr_to_name


def _load_dim_teams(root: Path) -> dict[str, str]:
    """Compatibility wrapper returning raw-key-to-abbreviation lookup."""
    return _load_team_maps(root)[0]


def _load_player_names(root: Path) -> dict[str, str]:
    """Build a player-id → display-name lookup across canonical dimensions.

    ``dim_player`` uses ``full_name`` today, while several released exports
    expose only ``first_name``/``last_name`` (and the staging tables use
    ``family_name``).  Combining those fields here keeps player enrichment
    independent of the export revision.
    """
    frames = _read_tables_merged(root, PLAYER_DIM_TABLES)
    out: dict[str, str] = {}
    for df in frames.values():
        if df.empty:
            continue
        pid = _first(df, "player_id", "person_id", "player_sk", "id")
        if pid is None:
            continue
        full = _first(df, "player_name", "full_name", "display_name", "name")
        first = _first(df, "first_name", "firstname")
        last = _first(df, "last_name", "family_name", "lastname", "surname")
        short = _first(df, "name_i", "short_name")
        for _, row in df.iterrows():
            value = _text(row.get(pid))
            if not value:
                continue
            display = _text(row.get(full)) if full else ""
            if not display and first and last:
                display = f"{_text(row.get(first))} {_text(row.get(last))}".strip()
            if not display and short:
                display = _text(row.get(short))
            out[value] = display or value
    return out


def _team_key(value: Any, lookup: dict[str, str]) -> str:
    text = _text(value).upper()
    if text in lookup:
        return lookup[text]
    # Numeric IDs are occasionally serialized as 1610612737.0.
    try:
        numeric = float(text)
        if numeric.is_integer() and str(int(numeric)) in lookup:
            return lookup[str(int(numeric))]
    except (TypeError, ValueError):
        pass
    return config.normalize_team_abbr(text)


def _team_name(value: Any, lookup: dict[str, str], names: dict[str, str]) -> str:
    key = _team_key(value, lookup)
    return names.get(key, key or _text(value))


def _merge_result_table(games: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
    """Join result facts while coalescing overlapping identity columns.

    A simple pandas merge with ``suffixes`` is not sufficient here: the
    canonical game dimension and result fact often both expose a score/team
    column, and selecting the dimension's null would silently discard the
    result score.  Fill only missing/blank values from the result fact and
    retain non-overlapping result aliases for the normalizers below.
    """
    if result.empty:
        return games.copy()
    base = games.copy().reset_index(drop=True)
    fact = result.copy().reset_index(drop=True)
    base_id = _first(base, "game_id", "gameId", "game_pk", "gameid", "id")
    fact_id = _first(fact, "game_id", "gameId", "game_pk", "gameid", "id")
    if base_id is None or fact_id is None:
        return base
    base["game_id"] = base[base_id].map(_game_id)
    fact["game_id"] = fact[fact_id].map(_game_id)
    # Identity and result facts are independently ordered exports.  Reindex
    # the result fact by game_id before coalescing; positional assignment can
    # attach another game's score when their CSV/SQL row orders differ.
    fact = (fact.drop_duplicates("game_id", keep="first")
            .set_index("game_id", drop=False))
    aligned = fact.reindex(base["game_id"].to_numpy())
    for col in aligned.columns:
        if col == "game_id":
            continue
        if col not in base.columns:
            base[col] = aligned[col].to_numpy()
            continue
        left = base[col]
        right = aligned[col].to_numpy()
        left_blank = left.isna()
        if left.dtype == object or pd.api.types.is_string_dtype(left):
            left_blank = left_blank | left.astype("string").str.strip().eq("").fillna(True)
        if left_blank.any():
            base.loc[left_blank, col] = right[left_blank.to_numpy()]
    return base


def _legacy_game_team_stats(raw: pd.DataFrame, games: pd.DataFrame,
                            lookup: dict[str, str]) -> pd.DataFrame:
    """Explode v238's wide ``game`` export into one row per team.

    The pinned v238 CSV/SQLite layout stores both box scores beside the game
    identity (``pts_home``/``pts_away`` and the corresponding traditional
    statistics).  Canonical star-schema releases provide long team facts, so
    this helper is only used as a fallback when those facts are absent.
    """
    if raw.empty or games.empty:
        return pd.DataFrame()
    df = raw.copy()
    gid = _coalesce_column(df, "game_id", "gameId", "game_pk", "gameid", "id")
    if gid.empty:
        return pd.DataFrame()
    stat_aliases = {
        "points_for": ("pts_{side}", "points_{side}", "team_points_{side}"),
        "fgm": ("fgm_{side}", "field_goals_made_{side}"),
        "fga": ("fga_{side}", "field_goals_attempted_{side}"),
        "fg3m": ("fg3m_{side}", "three_pointers_made_{side}"),
        "fg3a": ("fg3a_{side}", "three_pointers_attempted_{side}"),
        "ftm": ("ftm_{side}", "free_throws_made_{side}"),
        "fta": ("fta_{side}", "free_throws_attempted_{side}"),
        "oreb": ("oreb_{side}", "offensive_rebounds_{side}"),
        "dreb": ("dreb_{side}", "defensive_rebounds_{side}"),
        "reb": ("reb_{side}", "team_reb_{side}", "rebounds_{side}"),
        "ast": ("ast_{side}", "team_ast_{side}", "assists_{side}"),
        "tov": ("tov_{side}", "team_tov_{side}", "turnovers_{side}"),
        "stl": ("stl_{side}", "team_stl_{side}", "steals_{side}"),
        "blk": ("blk_{side}", "team_blk_{side}", "blocks_{side}"),
    }
    frames: list[pd.DataFrame] = []
    for side in ("home", "away"):
        team = _coalesce_column(
            df, f"team_abbreviation_{side}", f"team_abbr_{side}",
            f"team_id_{side}", f"{side}_team", f"{side}_team_id",
        )
        values: dict[str, Any] = {
            "game_id": gid,
            "team": team.map(lambda value: _team_key(value, lookup)),
        }
        for canonical, aliases in stat_aliases.items():
            values[canonical] = _number(_coalesce_column(
                df, *(alias.format(side=side) for alias in aliases)))
        frames.append(pd.DataFrame(values))
    out = pd.concat(frames, ignore_index=True)
    game_ids = set(games.game_id.astype(str))
    return out[out.game_id.astype(str).isin(game_ids)].reset_index(drop=True)


def _normalize_games(raw: pd.DataFrame, team_lookup: dict[str, str],
                     result: pd.DataFrame | None = None,
                     team_names: dict[str, str] | None = None) -> pd.DataFrame:
    if raw.empty and (result is None or result.empty):
        return pd.DataFrame()
    team_names = team_names or {}
    if raw.empty:
        raw = result
    elif result is not None and not result.empty:
        raw = _merge_result_table(raw, result)
    df = raw.copy()
    gid = _coalesce_column(df, "game_id", "gameId", "game_pk", "gameid", "id")
    date = _date(_coalesce_column(
        df, "gameday", "game_date", "gameDate", "date", "game_datetime",
        "tipoff_date", "datetime"))
    home_raw = _coalesce_column(
        df, "home_team", "homeTeam", "home_team_abbrev", "home_abbr",
        "team_abbreviation_home", "home_team_id", "team_id_home", "home_id",
        "home_team_name",
    )
    away_raw = _coalesce_column(
        df, "away_team", "awayTeam", "visitor_team", "away_team_abbrev",
        "away_abbr", "team_abbreviation_away", "away_team_id", "team_id_away",
        "visitor_team_id", "away_id", "away_team_name",
    )
    hs = _number(_coalesce_column(
        df, "home_score", "homeScore", "home_points", "home_pts", "pts_home",
        "home_team_score", "team_home_points", "home_team_pts",
    ))
    as_ = _number(_coalesce_column(
        df, "away_score", "awayScore", "away_points", "away_pts",
        "visitor_score", "pts_away", "away_team_score", "team_away_points",
        "away_team_pts",
    ))
    season = _season(_coalesce_column(
        df, "season_year", "season", "seasonYear", "season_id", "seasonId",
        "season_start_year", "season_start", "year"))
    # A legacy export may omit the season column entirely.  The start year is
    # still recoverable from the game date: the July boundary splits a season,
    # so October 2024 is the 2024 season and March 2024 is the 2023 season.
    season = season.fillna(_season_from_date(date))
    game_type = _coalesce_column(
        df, "game_type", "gameType", "season_type", "seasonType",
        "season_phase", "game_type_id")
    type_numeric = pd.to_numeric(game_type, errors="coerce")
    type_text = game_type.astype("string").str.strip().str.lower()
    type_text = type_text.str.replace(r"[_-]+", " ", regex=True)
    regular_text = type_text.str.contains(
        r"(^|\b)(regular|reg|regular season)(\b|$)", regex=True, na=False
    )
    post_text = type_text.str.contains(
        r"(playoff|playoffs|postseason|post season)", regex=True, na=False
    )
    # Missing phase is a regular-season source omission and is normalized to
    # 1.  Unknown textual phases remain NaN so they are excluded by
    # ``eligible_games`` instead of being silently treated as regular games.
    type_numeric = type_numeric.mask(regular_text, config.GAME_TYPE_REG)
    type_numeric = type_numeric.mask(post_text, config.GAME_TYPE_POST)
    missing_type = game_type.isna() | type_text.eq("") | type_text.eq("<na>")
    type_numeric = type_numeric.mask(missing_type, config.GAME_TYPE_REG)
    out = pd.DataFrame({
        "game_id": gid.map(_game_id),
        "gameday": date,
        "season": season,
        "home_team": home_raw.map(lambda x: _team_key(x, team_lookup)),
        "away_team": away_raw.map(lambda x: _team_key(x, team_lookup)),
        "home_score": hs,
        "away_score": as_,
        "game_type": type_numeric,
        "venue": _coalesce_column(df, "venue", "arena", "arena_name",
                                   "game_venue", "arena_full_name", default=""),
        "start_time_utc": _coalesce_column(
            df, "start_time_utc", "startTimeUTC", "game_datetime",
            "tipoff_time", "start_time", default=""),
        "game_status": _coalesce_column(
            df, "game_status", "status", "game_state", "game_status_text",
            default=""),
    })
    # Preserve a source-provided full name when it is more informative than
    # the dimension display name, while canonicalizing the team key.
    out["home_team_name"] = [
        _text(v) or _team_name(k, team_lookup, team_names)
        for v, k in zip(_coalesce_column(df, "home_team_name", "home_name"),
                        out.home_team)
    ]
    out["away_team_name"] = [
        _text(v) or _team_name(k, team_lookup, team_names)
        for v, k in zip(_coalesce_column(df, "away_team_name", "away_name"),
                        out.away_team)
    ]
    # Unknown textual phases remain NaN and are filtered by eligible_games;
    # do not turn a pre-season/exhibition marker into a regular-season game.
    out["game_type"] = pd.to_numeric(out["game_type"], errors="coerce")
    out = out[out.gameday.notna() & out.home_team.ne("") & out.away_team.ne("")].copy()
    out["home_win"] = np.where(out.home_score > out.away_score, 1.0, 0.0)
    out.loc[out.home_score.isna() | out.away_score.isna(), "home_win"] = np.nan
    # Some canonical result revisions carry the result flag but temporarily
    # leave one score null.  Preserve that settled label rather than turning
    # it into an unlabelled training row.
    wl_home = _coalesce_column(df, "wl_home", "home_win", "home_result")
    if wl_home is not None:
        wl_text = wl_home.astype("string").str.strip().str.lower()
        out["home_win"] = out["home_win"].fillna(
            wl_text.map({"w": 1.0, "win": 1.0, "l": 0.0, "loss": 0.0})
        )
    out["margin"] = out.home_score - out.away_score
    out["total"] = out.home_score + out.away_score
    status = out.game_status.astype("string").str.lower()
    out["game_status"] = np.where(
        out.home_score.notna() & out.away_score.notna(), "Final",
        np.where(status.str.contains("live|in progress", regex=True, na=False),
                 "Live", "Scheduled"))
    return (out.drop_duplicates("game_id").sort_values(["gameday", "game_id"])
            .reset_index(drop=True))


def _minutes(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    # NBA exports sometimes use MM:SS for minutes.
    text = series.astype("string").str.strip()
    clock = text.str.extract(r"^(\d+):(\d{1,2})$")
    if not clock.empty:
        converted = (pd.to_numeric(clock[0], errors="coerce")
                     + pd.to_numeric(clock[1], errors="coerce") / 60.0)
        numeric = numeric.fillna(converted)
    return numeric


def _normalize_team_stats(raw: pd.DataFrame, games: pd.DataFrame,
                         lookup: dict[str, str]) -> pd.DataFrame:
    if raw.empty or games.empty:
        return pd.DataFrame()
    df = raw.copy()
    gid = _coalesce_column(df, "game_id", "gameId", "game_pk", "gameid", "id")
    team = _coalesce_column(df, "team_abbrev", "team_abbr", "team", "team_name",
                            "team_id", "teamId")
    out = pd.DataFrame({"game_id": gid.map(_game_id),
                        "team": team.map(lambda x: _team_key(x, lookup))})
    aliases = {
        "points_for": ("points_for", "team_points", "team_pts", "points", "pts", "PTS"),
        "points_against": ("points_against", "opp_points", "opponent_points",
                           "opp_pts", "opponent_pts"),
        "off_rating": ("off_rating", "offensive_rating", "ortg", "offensive_rating"),
        "def_rating": ("def_rating", "defensive_rating", "drtg", "defensive_rating"),
        "pace": ("pace", "possessions", "pace_rating", "pace_estimate"),
        "efg_pct": ("efg_pct", "eFG%", "effective_field_goal_pct", "effective_fg_pct"),
        "fg_pct": ("fg_pct", "FG%", "field_goal_pct", "fg_percentage"),
        "three_point_pct": ("three_point_pct", "fg3_pct", "three_pct", "FG3%"),
        "free_throw_pct": ("free_throw_pct", "ft_pct", "FT%"),
        "oreb": ("oreb", "offensive_rebounds", "off_reb"),
        "dreb": ("dreb", "defensive_rebounds", "def_reb"),
        "reb": ("reb", "team_reb", "rebounds", "total_rebounds", "REB"),
        "ast": ("ast", "team_ast", "assists", "AST"),
        "tov": ("tov", "team_tov", "turnovers", "TOV"),
        "stl": ("stl", "team_stl", "steals", "STL"),
        "blk": ("blk", "team_blk", "blocks", "BLK"),
        "fgm": ("fgm", "field_goals_made", "FGM"),
        "fga": ("fga", "field_goals_attempted", "FGA"),
        "fg3m": ("fg3m", "three_pointers_made", "FG3M"),
        "fg3a": ("fg3a", "three_pointers_attempted", "FG3A"),
        "ftm": ("ftm", "free_throws_made", "FTM"),
        "fta": ("fta", "free_throws_attempted", "FTA"),
    }
    for name, candidates in aliases.items():
        out[name] = _number(_coalesce_column(df, *candidates))
    game_map = games.set_index("game_id")[["gameday", "home_team", "away_team",
                                            "home_score", "away_score"]]
    out = out.merge(game_map, left_on="game_id", right_index=True, how="left")
    out = out[out.gameday.notna() & out.team.ne("")].copy()
    out["is_home"] = out.team.eq(out.home_team)
    out["opponent"] = np.where(out.is_home, out.away_team, out.home_team)
    out["points_for"] = out.points_for.fillna(pd.Series(
        np.where(out.is_home, out.home_score, out.away_score), index=out.index))
    out["points_against"] = out.points_against.fillna(pd.Series(
        np.where(out.is_home, out.away_score, out.home_score), index=out.index))
    out["net_points"] = out.points_for - out.points_against
    # Team ratings are per-100 possessions in the warehouse.  If absent, use
    # a neutral, explicitly derived point-differential proxy rather than the
    # misleading 100x-points scale.
    out["off_rating"] = out.off_rating.fillna(100.0 + out.net_points)
    out["def_rating"] = out.def_rating.fillna(100.0 - out.net_points)
    out["pace"] = out.pace.fillna((out.home_score + out.away_score).clip(lower=1))
    out["efg_pct"] = out.efg_pct.fillna(
        ((out.fgm + 0.5 * out.fg3m) / out.fga.replace(0, np.nan)).fillna(0.5))
    out["fg_pct"] = out.fg_pct.fillna((out.fgm / out.fga.replace(0, np.nan)).fillna(0.45))
    out["three_point_pct"] = out.three_point_pct.fillna(
        (out.fg3m / out.fg3a.replace(0, np.nan)).fillna(0.35))
    out["free_throw_pct"] = out.free_throw_pct.fillna(
        (out.ftm / out.fta.replace(0, np.nan)).fillna(0.78))
    for c in ("oreb", "dreb", "reb", "ast", "tov", "stl", "blk"):
        out[c] = out[c].fillna(0.0)
    keep = ["game_id", "gameday", "team", "opponent", "is_home", "points_for",
            "points_against", "net_points", "off_rating", "def_rating", "pace",
            "efg_pct", "fg_pct", "three_point_pct", "free_throw_pct", "oreb",
            "dreb", "reb", "ast", "tov", "stl", "blk"]
    # A traditional/advanced/staging export can expose several rows for the
    # same game/team.  Prefer the most complete/point-bearing row rather than
    # whichever alias happened to be concatenated first.
    out["_quality"] = out[["points_for", "points_against", "ast", "tov", "reb"]].notna().sum(axis=1)
    out = (out.sort_values(["game_id", "team", "_quality"],
                            ascending=[True, True, False])
           .drop_duplicates(["game_id", "team"], keep="first"))
    opp = out[["game_id", "team", "tov", "reb"]].rename(
        columns={"team": "opponent", "tov": "opp_tov", "reb": "opp_reb"})
    out = out.merge(opp, on=["game_id", "opponent"], how="left")
    out["turnover_margin"] = out.opp_tov - out.tov
    out["rebound_margin"] = out.reb - out.opp_reb
    out = out.drop(columns="_quality", errors="ignore")
    return out.sort_values(["gameday", "game_id", "team"]).reset_index(drop=True)


def _normalize_player_stats(raw: pd.DataFrame, games: pd.DataFrame,
                            lookup: dict[str, str],
                            player_names: dict[str, str] | None = None) -> pd.DataFrame:
    if raw.empty or games.empty:
        return pd.DataFrame()
    player_names = player_names or {}
    df = raw.copy()
    pid = _coalesce_column(df, "player_id", "playerId", "person_id", "player_sk", "id")
    pname = _coalesce_column(
        df, "player_name", "playerName", "name", "display_name", "full_name")
    first = _coalesce_column(df, "first_name", "firstname")
    last = _coalesce_column(df, "last_name", "family_name", "lastname", "surname")
    name_blank = pname.isna() | pname.astype("string").str.strip().eq("").fillna(True)
    if name_blank.any():
        combined = (first.astype("string").fillna("") + " "
                    + last.astype("string").fillna("")).str.strip()
        pname.loc[name_blank] = combined.loc[name_blank]
    team = _coalesce_column(df, "team_abbrev", "team_abbr", "team", "team_name", "team_id")
    out = pd.DataFrame({
        "game_id": _coalesce_column(df, "game_id", "gameId", "game_pk", "id").map(_game_id),
        "team": team.map(lambda x: _team_key(x, lookup)),
        "player_id": pid.map(_game_id),
        "player_name": [player_names.get(_game_id(p), _text(n))
                        for p, n in zip(pid, pname)],
        "points": _number(_coalesce_column(df, "points", "pts", "PTS", "player_points")),
        "assists": _number(_coalesce_column(df, "assists", "ast", "AST", "player_assists")),
        "minutes": _minutes(_coalesce_column(df, "minutes", "min", "MIN", "minutes_played")),
    })
    out = out.merge(games[["game_id", "gameday"]], on="game_id", how="inner")
    # Prefer the most complete row when a warehouse join supplied both
    # traditional and advanced player facts.
    out["_quality"] = out[["points", "assists", "minutes"]].notna().sum(axis=1)
    out = (out.sort_values(["game_id", "player_id", "_quality"], ascending=[True, True, False])
           .drop_duplicates(["game_id", "player_id"], keep="first")
           .drop(columns="_quality"))
    return out.sort_values(["gameday", "game_id", "player_id"]).reset_index(drop=True)


def _cache_paths() -> tuple[Path, Path, Path, Path]:
    d = config.CACHE_DIR
    return (d / "games.parquet", d / "team_stats.parquet",
            d / "player_stats.parquet", d / "warehouse_manifest.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_inventory(root: Path) -> list[dict[str, Any]]:
    """Size the bundle beside a resolved warehouse, including flat mirrors."""
    base = root if root.is_dir() else root.parent
    if not base.is_dir():
        return []
    try:
        children = sorted(base.iterdir())
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for child in children:
        try:
            if child.is_file():
                out.append({"path": child.name,
                            "mb": round(child.stat().st_size / 1e6, 1)})
                continue
            files = [p for p in child.rglob("*") if p.is_file()]
            out.append({"path": child.name + "/", "files": len(files),
                        "mb": round(sum(p.stat().st_size for p in files) / 1e6, 1)})
        except OSError:
            continue
    return out


def _mirror_partitions(root: Path) -> dict[str, list[str]]:
    """Season values a Parquet mirror advertises, read from its paths.

    Partitioned exports encode the season in the directory name, so a mirror's
    coverage is known without reading a single row.  That distinction matters
    when a bundle's SQL copy and its Parquet export disagree about how far
    they reach.
    """
    base = root if root.is_dir() else root.parent
    parquet = base / "parquet"
    if not parquet.is_dir():
        return {}
    out: dict[str, set[str]] = {}
    try:
        paths = list(parquet.glob("**/*.parquet"))
    except OSError:
        return {}
    for path in paths:
        seasons = {part.split("=", 1)[1] for part in path.parts
                   if "=" in part and "season" in part.split("=", 1)[0].lower()}
        if not seasons:
            continue
        table = path.relative_to(parquet).parts[0]
        out.setdefault(table, set()).update(seasons)
    return {table: sorted(values) for table, values in sorted(out.items())}


def _audit_sql(path: Path, names: Iterable[str],
               probe_rows: bool) -> dict[str, dict[str, Any]]:
    """Catalog the tables this reader uses, probing the game tables in depth.

    Only game-identity tables are scanned for row counts and dates.  Box-score
    tables can hold millions of rows each, and a full-table aggregate over them
    would cost minutes on every run to answer a question the coverage gate
    never asks.  Their presence and shape are enough.
    """
    report: dict[str, dict[str, Any]] = {}
    if path.suffix.lower() == ".duckdb":
        try:
            import duckdb
        except ImportError:
            return report
        try:
            con = duckdb.connect(str(path), read_only=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("NBA source audit could not open %s: %s", path, exc)
            return report
        try:
            available = [str(r[0]) for r in con.execute("SHOW TABLES").fetchall()]
            for name, actual in _match_table_names(available, names).items():
                columns = [str(r[0]) for r in
                           con.execute(f'DESCRIBE "{actual}"').fetchall()]
                report[name] = _audit_entry(
                    lambda sql: con.execute(sql).fetchone()[0],
                    f'"{actual}"', columns,
                    deep=probe_rows and name in GAME_TABLES)
        except Exception as exc:  # noqa: BLE001
            logger.warning("NBA source audit failed for %s: %s", path, exc)
        finally:
            con.close()
        return report
    if path.suffix.lower() not in {".sqlite", ".db"}:
        return report
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NBA source audit could not open %s: %s", path, exc)
        return report
    try:
        available = [str(r[0]) for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        for name, actual in _match_table_names(available, names).items():
            columns = [str(r[1]) for r in con.execute(
                f'PRAGMA table_info("{actual}")').fetchall()]
            report[name] = _audit_entry(
                lambda sql: con.execute(sql).fetchone()[0],
                f'"{actual}"', columns,
                deep=probe_rows and name in GAME_TABLES)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NBA source audit failed for %s: %s", path, exc)
    finally:
        con.close()
    return report


def _audit_entry(scalar: Any, quoted: str, columns: list[str],
                 deep: bool) -> dict[str, Any]:
    """Summarize one table: row count, newest game date, newest season."""
    entry: dict[str, Any] = {"columns": len(columns)}
    if not deep:
        return entry
    date_col = next((c for c in columns if c.strip().lower() in _AUDIT_DATE_COLUMNS),
                    None)
    season_col = next((c for c in columns if "season" in c.strip().lower()), None)
    try:
        entry["rows"] = int(scalar(f"SELECT COUNT(*) FROM {quoted}") or 0)
        if date_col:
            newest = scalar(f'SELECT MAX("{date_col}") FROM {quoted}')
            entry["max_date"] = str(newest)[:10] if newest is not None else None
        if season_col:
            entry["max_season"] = scalar(f'SELECT MAX("{season_col}") FROM {quoted}')
    except Exception as exc:  # noqa: BLE001
        entry["error"] = str(exc)[:120]
    return entry


def audit_source(root: Path | None) -> dict[str, Any]:
    """Report what a bundle actually contains, per representation.

    A coverage gate is only trustworthy when the run states what it read: which
    files, which tables, how many rows, and how far each representation
    reaches.  This never relaxes a gate; it is the evidence the gates report,
    so a stale or half-exported dataset is named instead of guessed at.
    """
    if root is None or not root.exists():
        return {"source": None, "meets_window": False, "reason": "no source"}
    audit: dict[str, Any] = {
        "source": str(root),
        "bundle": _bundle_inventory(root),
        "parquet_seasons": _mirror_partitions(root),
        "sql_tables": {},
    }
    probe_rows = os.environ.get("NBA_SOURCE_AUDIT", "1").lower() not in {
        "0", "false", "no"}
    names = tuple(dict.fromkeys(
        GAME_TABLES + TEAM_BOX_TABLES + TEAM_TABLES + PLAYER_DIM_TABLES))
    for candidate in ([root] if root.is_file()
                      else sorted(p for p in root.glob("*")
                                  if p.suffix.lower() in _WAREHOUSE_SQL_SUFFIXES)):
        if not candidate.is_file():
            continue
        audit["sql_tables"][candidate.name] = _audit_sql(
            candidate, names, probe_rows)
    newest_date, newest_season = None, None
    for tables in audit["sql_tables"].values():
        for entry in tables.values():
            if entry.get("max_date"):
                newest_date = max(filter(None, [newest_date, entry["max_date"]]))
            if entry.get("max_season") is None:
                continue
            # Decode with the same rule the reader applies, so a five-digit
            # season id cannot masquerade as a year four digits ahead.
            decoded = _season(pd.Series([entry["max_season"]])).iloc[0]
            if pd.notna(decoded):
                newest_season = max(filter(None, [newest_season, float(decoded)]))
    for seasons in audit["parquet_seasons"].values():
        for value in seasons:
            if value[:4].isdigit():
                newest_season = max(filter(None, [newest_season, float(value[:4])]))
    audit["newest_game_date"] = newest_date
    audit["newest_season"] = newest_season
    audit["first_eligible_season"] = config.OOF_FIRST_SEASON
    audit["meets_window"] = bool(
        newest_season is not None and newest_season >= config.OOF_FIRST_SEASON)
    _log_audit(audit)
    return audit


def _log_audit(audit: dict[str, Any]) -> None:
    """Log the source inventory once, loudly when it falls short."""
    if audit.get("reason"):
        logger.warning("NBA source audit: %s", audit["reason"])
        return
    for name, tables in audit["sql_tables"].items():
        for table, entry in tables.items():
            logger.info("NBA source audit: %s.%s rows=%s max_date=%s "
                        "max_season=%s", name, table, entry.get("rows"),
                        entry.get("max_date"), entry.get("max_season"))
    for table, seasons in audit["parquet_seasons"].items():
        logger.info("NBA source audit: parquet/%s seasons %s..%s", table,
                    seasons[0], seasons[-1])
    for item in audit["bundle"]:
        logger.info("NBA source audit: bundle entry %s %s", item,
                    f"{item.get('mb')} MB")
    if audit["meets_window"]:
        logger.info("NBA source audit: coverage reaches %s, meets the %s window",
                    audit["newest_season"], audit["first_eligible_season"])
    else:
        logger.warning(
            "NBA source audit: no representation in %s reaches the %s season "
            "window (newest date %s, newest season %s). The pinned dataset "
            "version no longer contains eligible history; repin the version or "
            "obtain a newer export rather than relaxing the gate.",
            audit["source"], audit["first_eligible_season"],
            audit.get("newest_game_date") or "unknown",
            audit.get("newest_season") or "unknown")


def _manifest(root: Path | None, tables: dict[str, pd.DataFrame],
              team_names: dict[str, str] | None = None,
              audit: dict[str, Any] | None = None) -> dict:
    files = []
    if root:
        paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in paths[:5000]:
            try:
                files.append({"path": str(path), "sha256": _sha256_file(path),
                              "bytes": path.stat().st_size})
            except OSError:
                continue
    coverage: dict[str, Any] = {}
    games = tables.get("games")
    if games is not None and len(games):
        coverage = {
            "min_date": str(pd.to_datetime(games.gameday).min().date()),
            "max_date": str(pd.to_datetime(games.gameday).max().date()),
            "seasons": sorted(pd.to_numeric(games.season, errors="coerce").dropna().unique().tolist()),
            "teams": sorted(set(games.home_team) | set(games.away_team)),
            "eligible_team_count": int(len({
                config.normalize_team_abbr(team)
                for team in set(games.loc[
                    pd.to_numeric(games.season, errors="coerce") >= config.OOF_FIRST_SEASON,
                    "home_team",
                ].astype(str)) | set(games.loc[
                    pd.to_numeric(games.season, errors="coerce") >= config.OOF_FIRST_SEASON,
                    "away_team",
                ].astype(str))
            })),
        }
    return {
        "dataset_id": "wyattowalsh/basketball",
        "dataset_version": os.environ.get(
            "NBA_KAGGLE_DATASET_VERSION", config.NBA_DATASET_VERSION),
        "source_path": str(root) if root else None,
        "retrieved_at_utc": pd.Timestamp.utcnow().isoformat(),
        "tables": {k: int(len(v)) for k, v in tables.items()},
        "schemas": {
            name: {
                "columns": [str(column) for column in frame.columns],
                "dtypes": {str(column): str(dtype)
                           for column, dtype in frame.dtypes.items()},
            }
            for name, frame in tables.items()
        },
        "coverage": coverage,
        "team_names": team_names or {},
        "files": files,
        "source_audit": audit or {},
        "schema_version": "nba-normalized-v2",
    }


def _audit_summary(audit: dict[str, Any] | None) -> str:
    """One-line rendering of the source audit for an error message."""
    if not audit:
        return "not run"
    if audit.get("reason"):
        return str(audit["reason"])
    reached = []
    for name, tables in (audit.get("sql_tables") or {}).items():
        for table, entry in tables.items():
            if entry.get("max_season") is not None or entry.get("max_date"):
                reached.append(f"{name}.{table}="
                               f"{entry.get('max_date') or entry.get('max_season')}")
    for table, seasons in (audit.get("parquet_seasons") or {}).items():
        reached.append(f"parquet/{table}={seasons[0]}..{seasons[-1]}")
    verdict = "meets" if audit.get("meets_window") else "does not meet"
    return (f"newest_game_date={audit.get('newest_game_date')}, "
            f"newest_season={audit.get('newest_season')}, {verdict} the "
            f"{audit.get('first_eligible_season')} window; " +
            (", ".join(reached[:8]) or "no dated tables found"))


def _validate_dataset(wh: Warehouse) -> None:
    required_games = {"game_id", "gameday", "season", "home_team", "away_team"}
    missing = required_games - set(wh.games.columns)
    if missing:
        raise RuntimeError(f"NBA warehouse games missing required columns: {sorted(missing)}")
    if wh.games.empty:
        raise RuntimeError("NBA warehouse has no usable game rows")
    seasons = pd.to_numeric(wh.games.season, errors="coerce")
    if not (seasons >= config.OOF_FIRST_SEASON).any():
        # A coverage gate is only actionable if it reports what was actually
        # read, so the message carries the observed seasons, date range, and
        # resolved source instead of a bare threshold complaint.
        observed = sorted(set(seasons.dropna().tolist()))
        dates = pd.to_datetime(wh.games.gameday, errors="coerce")
        window = ("none" if dates.notna().sum() == 0
                  else f"{dates.min().date()}..{dates.max().date()}")
        raise RuntimeError(
            f"NBA warehouse has no 2024-25-or-later season coverage "
            f"(rows={len(wh.games)}, observed_seasons={observed or 'none'}, "
            f"gamedays={window}, "
            f"source={wh.manifest.get('source_path')}, "
            f"game_columns={list(wh.games.columns)}); "
            f"source audit: {_audit_summary(wh.manifest.get('source_audit'))}"
        )
    eligible = wh.games[pd.to_numeric(wh.games.season, errors="coerce") >= config.OOF_FIRST_SEASON]
    observed_teams = (set(eligible.home_team.dropna().astype(str))
                      | set(eligible.away_team.dropna().astype(str)))
    observed_teams = {config.normalize_team_abbr(team) for team in observed_teams}
    required_teams = set(config.NBA_TEAM_ID)
    missing_teams = sorted(required_teams - observed_teams)
    if missing_teams:
        raise RuntimeError(
            "NBA warehouse is missing required current-team coverage: "
            + ", ".join(missing_teams)
        )
    # Team facts are required for the production feature graph.  A deliberate
    # opt-out exists for schema-only fixture inspection, never as a silent
    # source switch.
    degraded = os.environ.get("NBA_ALLOW_DEGRADED_INPUT", "").lower() in {"1", "true", "yes"}
    if not degraded and wh.team_stats.empty:
        raise RuntimeError("NBA warehouse is missing required team box-score coverage")
    if not degraded and not wh.team_stats.empty:
        # Team facts are keyed by game_id; require a usable fact for every
        # current team represented by the eligible schedule.  This catches a
        # truncated export that still has a non-empty but useless table.
        fact_games = set(wh.team_stats.game_id.astype(str))
        eligible_games = set(eligible.game_id.astype(str))
        covered_games = fact_games & eligible_games
        covered_teams = set(
            wh.team_stats[wh.team_stats.game_id.astype(str).isin(covered_games)]
            .team.astype(str).map(config.normalize_team_abbr)
        )
        missing_fact_teams = sorted(set(config.NBA_TEAM_ID) - covered_teams)
        if missing_fact_teams:
            raise RuntimeError(
                "NBA warehouse is missing required team box-score coverage for: "
                + ", ".join(missing_fact_teams)
            )


def load_dataset(source: str | Path | None = None, use_cache: bool = True,
                 allow_download: bool = False) -> Warehouse:
    raw_source = source
    if isinstance(source, str) and source.strip().lower() in {"none", "null"}:
        logger.warning("treating source=%r as an omitted source", source)
        source = None
    gp, tp, pp, mp = _cache_paths()
    if use_cache and source is None and mp.exists() and gp.exists() and tp.exists() and pp.exists():
        try:
            manifest = json.loads(mp.read_text())
            wh = Warehouse(pd.read_parquet(gp), pd.read_parquet(tp), pd.read_parquet(pp),
                           dict(manifest.get("team_names", {})), manifest)
            _validate_dataset(wh)
            return wh
        except Exception as exc:  # noqa: BLE001
            logger.warning("invalid NBA cache, rebuilding: %s", exc)
    root = resolve_source(source, allow_download=allow_download)
    if root is None or not root.exists():
        searched = [str(path) for path in _source_search_roots()]
        if raw_source is not None:
            searched.insert(0, str(Path(str(raw_source)).expanduser()))
        raise FileNotFoundError(
            "NBA warehouse not found "
            f"(source={raw_source!r}; searched: {', '.join(searched)}). "
            "Attach wyattowalsh/basketball version 238 in Kaggle and run "
            "kaggle_nba_run.ipynb (not kaggle_mlb_run.ipynb), or set "
            "NBA_KAGGLE_DATASET_PATH to its extracted directory/DuckDB file. "
            "Do not pass None as a --source-path value."
        )
    lookup, team_names = _load_team_maps(root)
    audit = audit_source(root)
    game_frames = _read_tables_merged(root, GAME_TABLES)
    # dim_game is the identity source; result/scoreboard tables are joined by
    # game_id.  If only a result table exists it is used as the identity source.
    identity_name, identity = next(((name, game_frames[name]) for name in (
        "dim_game", "games", "stg_league_game_log", "fact_game", "game",
        "game_summary")
        if name in game_frames and not game_frames[name].empty), (None, None))
    result = _select_result(game_frames, identity_name, identity)
    if identity is None and result is None:
        raise RuntimeError("wyattowalsh/basketball is missing dim_game/fact_game_result")
    games = _normalize_games(identity if identity is not None else result, lookup,
                             result=result, team_names=team_names)
    if games.empty:
        raise RuntimeError("NBA warehouse produced no usable game rows")
    team_frames = list(_read_tables_merged(root, TEAM_BOX_TABLES).values())
    team_raw = pd.concat(team_frames, ignore_index=True) if team_frames else pd.DataFrame()
    team_stats = _normalize_team_stats(team_raw, games, lookup)
    legacy_game = game_frames.get("game")
    if team_stats.empty and legacy_game is not None and not legacy_game.empty:
        legacy_raw = _legacy_game_team_stats(legacy_game, games, lookup)
        team_stats = _normalize_team_stats(legacy_raw, games, lookup)
    player_frames = list(_read_tables_merged(root, PLAYER_TABLES).values())
    player_raw = pd.concat(player_frames, ignore_index=True) if player_frames else pd.DataFrame()
    player_stats = _normalize_player_stats(player_raw, games, lookup,
                                           _load_player_names(root))
    wh = Warehouse(games, team_stats, player_stats, team_names,
                   _manifest(root, {"games": games, "team_stats": team_stats,
                                    "player_stats": player_stats}, team_names,
                             audit=audit))
    _validate_dataset(wh)
    if use_cache:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        games.to_parquet(gp, index=False)
        team_stats.to_parquet(tp, index=False)
        player_stats.to_parquet(pp, index=False)
        mp.write_text(json.dumps(
            wh.manifest, indent=2, default=str, allow_nan=False))
    return wh


def eligible_games(games: pd.DataFrame) -> pd.DataFrame:
    df = games.copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df[df["season"] >= config.OOF_FIRST_SEASON].copy()
    if "game_type" in df:
        # Keep textual postseason flags normalized by ingestion as 2; invalid
        # types are not silently treated as regular-season games.
        df = df[pd.to_numeric(df["game_type"], errors="coerce").isin(config.GAME_TYPES)]
    return df.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def load_games(source: str | Path | None = None, use_cache: bool = True) -> pd.DataFrame:
    return load_dataset(source, use_cache).games


def load_team_stats(source: str | Path | None = None, use_cache: bool = True) -> pd.DataFrame:
    return load_dataset(source, use_cache).team_stats


def load_player_stats(source: str | Path | None = None, use_cache: bool = True) -> pd.DataFrame:
    return load_dataset(source, use_cache).player_stats


def load_team_names(source: str | Path | None = None) -> dict[str, str]:
    try:
        return dict(load_dataset(source).team_names)
    except Exception:
        return {}

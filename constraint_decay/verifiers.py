"""
Static verifiers for constraint adherence.

Each verifier takes a patch file path and returns a VerifierResult
indicating whether the solution adheres to the constraint.
"""

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import re


@dataclass
class VerifierResult:
    """Result of a constraint verification check."""
    compliant: bool
    reason: str
    details: dict = field(default_factory=dict)


# ── Clean Architecture verifier ──────────────────────────────────────────────
#
# The prompt requires 4 layers, each in its own directory:
#   1. Routes / Handlers       (rank 3 — top)
#   2. Services / Use Cases    (rank 2)
#   3. Repository / Data Access(rank 1)
#   4. Models / Entities       (rank 0 — bottom, leaf)
#
# Dependency rule: each layer may only import from layers with equal or lower
# rank. A violation is importing from a layer with a *higher* rank.
#
# Agents place these under varying roots (src/, app/, api/, conduit/, or repo
# root).  We normalise by matching any path component against the canonical
# name sets for each layer.

LAYER_PATTERNS: dict[str, set[str]] = {
    "routes":       {"routes", "handlers", "route", "handler", "routers", "views"},
    "services":     {"services", "service", "usecases", "use_cases"},
    "models":       {"models", "model", "entities", "entity", "domain"},
    "repositories": {"repositories", "repository", "repos", "data_access",
                     "data", "dao", "dal"},
}

# Rank: higher number = higher layer.  Importing a higher-rank layer is a
# violation.
LAYER_RANK: dict[str, int] = {
    "models":       0,
    "repositories": 1,
    "services":     2,
    "routes":       3,
}

# Minimum number of layers (out of 4) that must be present as distinct
# directories for the solution to be considered compliant.
MIN_LAYERS = 3


# ── Patch parsing helpers ────────────────────────────────────────────────────

def _extract_files_from_patch(patch_path: Path) -> list[str]:
    """Extract file paths touched by a unified diff patch."""
    files: list[str] = []
    text = patch_path.read_text(errors="replace")
    for m in re.finditer(r"^diff --git a/(.+?) b/", text, re.MULTILINE):
        files.append(m.group(1))
    return files


def _extract_file_contents(patch_path: Path) -> dict[str, str]:
    """
    Parse a unified diff and return {filepath: added_lines_text} for each file.

    Only added lines (starting with ``+``, excluding the ``+++`` header) are
    kept, giving us the source code the agent actually wrote.
    """
    text = patch_path.read_text(errors="replace")
    files: dict[str, str] = {}
    current_file: str | None = None
    lines: list[str] = []

    for raw_line in text.splitlines():
        m = re.match(r"^diff --git a/(.+?) b/", raw_line)
        if m:
            # flush previous file
            if current_file is not None:
                files[current_file] = "\n".join(lines)
            current_file = m.group(1)
            lines = []
            continue
        if current_file is None:
            continue
        # skip diff meta-lines
        if raw_line.startswith("+++") or raw_line.startswith("---"):
            continue
        # keep added lines (strip the leading '+')
        if raw_line.startswith("+"):
            lines.append(raw_line[1:])

    if current_file is not None:
        files[current_file] = "\n".join(lines)
    return files


def _dir_components(filepath: str) -> list[str]:
    """Return all directory components of a path (lowercased)."""
    parts = PurePosixPath(filepath).parts
    return [p.lower() for p in parts[:-1]]


def _classify_layer(filepath: str) -> str | None:
    """Return the canonical layer name for *filepath*, or None if it doesn't
    belong to a recognised layer directory."""
    dirs = set(_dir_components(filepath))
    for layer_name, aliases in LAYER_PATTERNS.items():
        if aliases & dirs:
            return layer_name
    return None


# ── Import extraction ────────────────────────────────────────────────────────

# Python:  from src.services.foo import bar  /  import src.repositories.baz
_PY_FROM_IMPORT = re.compile(
    r"^\s*from\s+([\w.]+)\s+import", re.MULTILINE,
)
_PY_IMPORT = re.compile(
    r"^\s*import\s+([\w.]+)", re.MULTILINE,
)

# Node CJS:  require('../services/articleService')  /  require('./repositories/foo')
_JS_REQUIRE = re.compile(
    r"""require\(\s*['"]([^'"]+)['"]\s*\)""",
)

# Node ESM:  import { X } from '../services/foo'  /  import X from './models/bar'
_JS_ESM_IMPORT = re.compile(
    r"""^\s*import\s+.*?\s+from\s+['"]([^'"]+)['"]""", re.MULTILINE,
)


def _imported_layer(import_path: str) -> str | None:
    """Given an import path string, return which layer it references (or None).

    Handles both Python dotted paths (``src.services.foo``) and JS relative
    paths (``../services/articleService``).
    """
    # Normalise: dots → slashes for Python paths, then split into components
    normalised = import_path.replace(".", "/")
    parts = [p.lower() for p in PurePosixPath(normalised).parts if p not in (".", "..")]
    for layer_name, aliases in LAYER_PATTERNS.items():
        if aliases & set(parts):
            return layer_name
    return None


def _extract_imports_with_layers(source: str) -> list[tuple[str, str]]:
    """Return [(raw_import_path, target_layer), ...] for every import that
    references a recognised layer."""
    results: list[tuple[str, str]] = []
    # Python
    for m in _PY_FROM_IMPORT.finditer(source):
        layer = _imported_layer(m.group(1))
        if layer:
            results.append((m.group(1), layer))
    for m in _PY_IMPORT.finditer(source):
        layer = _imported_layer(m.group(1))
        if layer:
            results.append((m.group(1), layer))
    # JS require
    for m in _JS_REQUIRE.finditer(source):
        layer = _imported_layer(m.group(1))
        if layer:
            results.append((m.group(1), layer))
    # JS ESM import
    for m in _JS_ESM_IMPORT.finditer(source):
        layer = _imported_layer(m.group(1))
        if layer:
            results.append((m.group(1), layer))
    return results


# ── Public API ───────────────────────────────────────────────────────────────

def verify_clean_architecture(patch_path: Path | str) -> VerifierResult:
    """
    Check whether a patch adheres to the Clean Architecture constraint.

    Checks two properties:
    1. **Layer presence** – at least ``MIN_LAYERS`` of the 4 canonical layers
       exist as distinct directories.
    2. **Dependency direction** – no file in a lower layer imports from a
       higher layer.  The ranking (high→low) is:
       routes (3) → services (2) → repositories (1) → models (0).

    Parameters
    ----------
    patch_path : Path or str
        Path to the ``.patch`` file produced by the agent run.

    Returns
    -------
    VerifierResult
        ``compliant=True`` if both checks pass.
    """
    patch_path = Path(patch_path)

    if not patch_path.exists():
        return VerifierResult(
            compliant=False,
            reason="Patch file not found",
            details={"patch_path": str(patch_path)},
        )

    files = _extract_files_from_patch(patch_path)
    if not files:
        return VerifierResult(
            compliant=False,
            reason="No files found in patch",
            details={"patch_path": str(patch_path)},
        )

    # ── Check 1: layer presence ──────────────────────────────────────────
    found_layers: dict[str, list[str]] = {}
    for layer_name, aliases in LAYER_PATTERNS.items():
        matching = [f for f in files if aliases & set(_dir_components(f))]
        if matching:
            found_layers[layer_name] = matching

    n_found = len(found_layers)
    missing = sorted(set(LAYER_PATTERNS.keys()) - set(found_layers.keys()))
    layers_ok = n_found >= MIN_LAYERS

    # ── Check 2: dependency direction ────────────────────────────────────
    file_contents = _extract_file_contents(patch_path)
    violations: list[dict] = []

    for filepath, source in file_contents.items():
        src_layer = _classify_layer(filepath)
        if src_layer is None:
            continue  # file outside any layer (e.g. config, main, run.sh)
        src_rank = LAYER_RANK[src_layer]

        for import_path, target_layer in _extract_imports_with_layers(source):
            target_rank = LAYER_RANK[target_layer]
            if target_rank > src_rank:
                violations.append({
                    "file": filepath,
                    "file_layer": src_layer,
                    "imports": import_path,
                    "target_layer": target_layer,
                })

    deps_ok = len(violations) == 0

    # ── Combine ──────────────────────────────────────────────────────────
    compliant = layers_ok and deps_ok

    reasons = []
    if layers_ok:
        reasons.append(f"Layers {n_found}/4: {', '.join(sorted(found_layers))}")
    else:
        reasons.append(
            f"Only {n_found}/4 layers (need >= {MIN_LAYERS}), "
            f"missing: {', '.join(missing)}"
        )
    if deps_ok:
        reasons.append("No dependency violations")
    else:
        reasons.append(f"{len(violations)} dependency violation(s)")

    return VerifierResult(
        compliant=compliant,
        reason="; ".join(reasons),
        details={
            "found_layers": {k: len(v) for k, v in found_layers.items()},
            "missing_layers": missing,
            "n_files": len(files),
            "dependency_violations": violations,
        },
    )


# ── Database verifier ────────────────────────────────────────────────────────
#
# The prompt prescribes either SQLite or PostgreSQL.  We scan the added lines
# in the patch for evidence of each database and check that the prescribed one
# is present while the *other* is absent.
#
# Evidence can appear as:
#   - Direct driver imports  (sqlite3, aiosqlite, psycopg2, asyncpg, pg, …)
#   - Connection strings     (sqlite:///…, postgresql://…, postgres://…)
#   - ORM dialect settings   (dialect: 'sqlite', dialect: 'postgres')
#   - Dependency declarations (requirements.txt, package.json entries)
#   - Port references        (port 5432 in DB config context)

# Each regex is applied to the full blob of added lines (case-insensitive).
# They are intentionally broad to catch variations (e.g. psycopg2-binary,
# asyncpg, pg-promise, etc.).

_SQLITE_EVIDENCE: list[tuple[str, re.Pattern]] = [
    # Python
    ("import:sqlite3",       re.compile(r"(?:^|\s)import\s+sqlite3|from\s+sqlite3\s+import", re.M)),
    ("import:aiosqlite",     re.compile(r"(?:^|\s)import\s+aiosqlite|from\s+aiosqlite\s+import", re.M)),
    # Connection strings  (SQLAlchemy / generic, handles dialect+driver:// like sqlite+aiosqlite:///)
    ("connstr:sqlite",       re.compile(r"sqlite(?:\+\w+)?:///", re.I)),
    # Django backend
    ("django:sqlite",        re.compile(r"django\.db\.backends\.sqlite3", re.I)),
    # Node drivers
    ("dep:sqlite3-npm",      re.compile(r"""['"]sqlite3['"]""", re.I)),
    ("dep:better-sqlite3",   re.compile(r"""['"]better-sqlite3['"]""", re.I)),
    # Sequelize dialect
    ("dialect:sqlite",       re.compile(r"""dialect\s*[:=]\s*['"]sqlite['"]""", re.I)),
]

_POSTGRES_EVIDENCE: list[tuple[str, re.Pattern]] = [
    # Python drivers
    ("import:psycopg2",      re.compile(r"(?:^|\s)import\s+psycopg2|from\s+psycopg2\s+import|psycopg2", re.M)),
    ("import:asyncpg",       re.compile(r"(?:^|\s)import\s+asyncpg|from\s+asyncpg\s+import", re.M)),
    # Connection strings  (SQLAlchemy / generic, handles dialect+driver:// like postgresql+asyncpg://)
    ("connstr:postgresql",   re.compile(r"postgres(?:ql)?(?:\+\w+)?://", re.I)),
    # SQLAlchemy dialect-specific imports  (e.g. from sqlalchemy.dialects.postgresql import …)
    ("dialect-import:pg",    re.compile(r"sqlalchemy\.dialects\.postgresql", re.I)),
    # Django backend
    ("django:postgres",      re.compile(r"django\.db\.backends\.postgresql", re.I)),
    # Node drivers – match "pg" as a dependency value, not as a substring
    ("dep:pg-npm",           re.compile(r"""['"]pg['"]""")),
    ("dep:pg-promise",       re.compile(r"""['"]pg-promise['"]""", re.I)),
    # Sequelize dialect
    ("dialect:postgres",     re.compile(r"""dialect\s*[:=]\s*['"]postgres(?:ql)?['"]""", re.I)),
    # Host reference to our Docker service name
    ("host:postgres",        re.compile(r"""host\s*[:=]\s*['"]postgres['"]""", re.I)),
]


def _collect_evidence(
    source: str,
    patterns: list[tuple[str, re.Pattern]],
) -> list[str]:
    """Return labels for every pattern that matches *source*."""
    return [label for label, pat in patterns if pat.search(source)]


def verify_database(patch_path: Path | str, expected_db: str) -> VerifierResult:
    """
    Check whether a patch uses the prescribed database (``sqlite`` or
    ``postgres``).

    A solution is **compliant** when:
    1. At least one evidence pattern for the *expected* DB is found, **and**
    2. No evidence of the *other* DB is found.

    Parameters
    ----------
    patch_path : Path or str
        Path to the ``.patch`` file.
    expected_db : str
        ``"sqlite"`` or ``"postgres"``.

    Returns
    -------
    VerifierResult
    """
    patch_path = Path(patch_path)
    expected_db = expected_db.lower()

    if expected_db not in ("sqlite", "postgres"):
        return VerifierResult(
            compliant=False,
            reason=f"Unknown expected_db: {expected_db!r} (must be 'sqlite' or 'postgres')",
        )

    if not patch_path.exists():
        return VerifierResult(
            compliant=False,
            reason="Patch file not found",
            details={"patch_path": str(patch_path)},
        )

    file_contents = _extract_file_contents(patch_path)
    if not file_contents:
        return VerifierResult(
            compliant=False,
            reason="No files found in patch",
            details={"patch_path": str(patch_path)},
        )

    # Concatenate all added lines into a single blob for matching.
    # Exclude lock files — they list optional peer dependencies for all
    # supported dialects (e.g. Sequelize ships pg AND sqlite3 in its
    # package-lock.json regardless of which dialect is used).
    skip = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
    all_source = "\n".join(
        src for path, src in file_contents.items()
        if PurePosixPath(path).name not in skip
    )

    sqlite_evidence = _collect_evidence(all_source, _SQLITE_EVIDENCE)
    postgres_evidence = _collect_evidence(all_source, _POSTGRES_EVIDENCE)

    # Django projects often keep the default sqlite3 backend in settings.py even
    # when postgres is configured (the second DATABASES assignment overrides).
    # If the only sqlite signal is the Django boilerplate, ignore it when
    # postgres is expected (and vice-versa).
    if expected_db == "postgres" and sqlite_evidence == ["django:sqlite"] and postgres_evidence:
        sqlite_evidence = []
    if expected_db == "sqlite" and postgres_evidence == ["django:postgres"] and sqlite_evidence:
        postgres_evidence = []

    if expected_db == "sqlite":
        has_expected = len(sqlite_evidence) > 0
        has_other = len(postgres_evidence) > 0
        expected_evidence = sqlite_evidence
        other_evidence = postgres_evidence
        other_name = "postgres"
    else:
        has_expected = len(postgres_evidence) > 0
        has_other = len(sqlite_evidence) > 0
        expected_evidence = postgres_evidence
        other_evidence = sqlite_evidence
        other_name = "sqlite"

    compliant = has_expected and not has_other

    reasons = []
    if has_expected:
        reasons.append(f"Found {expected_db} evidence: {', '.join(expected_evidence)}")
    else:
        reasons.append(f"No {expected_db} evidence found")
    if has_other:
        reasons.append(f"Also found {other_name} evidence: {', '.join(other_evidence)}")

    return VerifierResult(
        compliant=compliant,
        reason="; ".join(reasons),
        details={
            "expected_db": expected_db,
            "sqlite_evidence": sqlite_evidence,
            "postgres_evidence": postgres_evidence,
        },
    )


# ── ORM verifier ─────────────────────────────────────────────────────────────
#
# The prompt prescribes either SQLAlchemy (Python) or Sequelize (Node).
# We scan the added lines for evidence of each ORM.
#
# Evidence:
#   - Python: sqlalchemy imports, SQLAlchemy in requirements.txt
#   - Node:   sequelize imports/requires, sequelize in package.json

_SQLALCHEMY_EVIDENCE: list[tuple[str, re.Pattern]] = [
    # Python imports
    ("import:sqlalchemy",     re.compile(r"(?:^|\s)import\s+sqlalchemy|from\s+sqlalchemy(?:\.|\s+import)", re.M)),
    # Dependency declaration  (requirements.txt / pyproject.toml)
    ("dep:sqlalchemy",        re.compile(r"(?i)sqlalchemy")),
]

_SEQUELIZE_EVIDENCE: list[tuple[str, re.Pattern]] = [
    # Node CJS / ESM
    ("import:sequelize",      re.compile(r"""require\(\s*['"]sequelize['"]\s*\)""", re.I)),
    ("esm:sequelize",         re.compile(r"""from\s+['"]sequelize['"]""", re.I)),
    # Dependency declaration  (package.json)
    ("dep:sequelize",         re.compile(r"""['"]sequelize['"]\s*:""", re.I)),
    # Sequelize constructor / init  (new Sequelize(...))
    ("usage:sequelize",       re.compile(r"new\s+Sequelize\s*\(", re.I)),
]


def verify_orm(patch_path: Path | str, expected_orm: str) -> VerifierResult:
    """
    Check whether a patch uses the prescribed ORM (``sqlalchemy`` or
    ``sequelize``).

    A solution is **compliant** when at least one evidence pattern for the
    expected ORM is found.

    Parameters
    ----------
    patch_path : Path or str
        Path to the ``.patch`` file.
    expected_orm : str
        ``"sqlalchemy"`` or ``"sequelize"``.

    Returns
    -------
    VerifierResult
    """
    patch_path = Path(patch_path)
    expected_orm = expected_orm.lower()

    if expected_orm not in ("sqlalchemy", "sequelize"):
        return VerifierResult(
            compliant=False,
            reason=f"Unknown expected_orm: {expected_orm!r} (must be 'sqlalchemy' or 'sequelize')",
        )

    if not patch_path.exists():
        return VerifierResult(
            compliant=False,
            reason="Patch file not found",
            details={"patch_path": str(patch_path)},
        )

    file_contents = _extract_file_contents(patch_path)
    if not file_contents:
        return VerifierResult(
            compliant=False,
            reason="No files found in patch",
            details={"patch_path": str(patch_path)},
        )

    skip = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
    all_source = "\n".join(
        src for path, src in file_contents.items()
        if PurePosixPath(path).name not in skip
    )

    sqlalchemy_evidence = _collect_evidence(all_source, _SQLALCHEMY_EVIDENCE)
    sequelize_evidence = _collect_evidence(all_source, _SEQUELIZE_EVIDENCE)

    if expected_orm == "sqlalchemy":
        has_expected = len(sqlalchemy_evidence) > 0
        expected_evidence = sqlalchemy_evidence
    else:
        has_expected = len(sequelize_evidence) > 0
        expected_evidence = sequelize_evidence

    compliant = has_expected

    if has_expected:
        reason = f"Found {expected_orm} evidence: {', '.join(expected_evidence)}"
    else:
        reason = f"No {expected_orm} evidence found"

    return VerifierResult(
        compliant=compliant,
        reason=reason,
        details={
            "expected_orm": expected_orm,
            "sqlalchemy_evidence": sqlalchemy_evidence,
            "sequelize_evidence": sequelize_evidence,
        },
    )

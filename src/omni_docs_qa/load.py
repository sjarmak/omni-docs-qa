"""Copy a docs-QA corpus JSONL into the omni_docs_qa.sections Neon table."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from omni_docs_qa.corpus import NEON_DDL

DSN_ENVIRONMENT = "DOCS_QA_ADMIN_DSN"
COLUMNS = (
    "section_id",
    "page_url",
    "page_title",
    "heading_path",
    "heading_label",
    "section_order",
    "part_index",
    "part_count",
    "body",
    "content_hash",
    "source_kind",
    "snapshot_version",
)
COPY_STATEMENT = f"COPY omni_docs_qa.sections ({', '.join(COLUMNS)}) FROM STDIN"


class DocsLoadError(RuntimeError):
    """The corpus cannot be loaded safely."""


class Copy(Protocol):
    def write_row(self, row: tuple[Any, ...]) -> None: ...


class Cursor(Protocol):
    def execute(self, statement: str) -> Any: ...
    def copy(self, statement: str) -> Any: ...


def snapshot_version_from_meta(meta_path: Path) -> str:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DocsLoadError(
            f"cannot read corpus metadata {meta_path}: {error}"
        ) from error
    version = meta.get("version")
    corpus_hash = meta.get("corpus_hash")
    if isinstance(version, str) and version:
        return version
    if isinstance(corpus_hash, str) and len(corpus_hash) == 64:
        return f"base-{corpus_hash[:12]}"
    raise DocsLoadError("corpus metadata has neither a version nor a corpus_hash")


def read_rows(corpus_path: Path, snapshot_version: str) -> Iterator[tuple[Any, ...]]:
    try:
        lines = corpus_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DocsLoadError(f"cannot read corpus {corpus_path}: {error}") from error
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        yield _row(line, number, snapshot_version)


def _row(line: str, number: int, snapshot_version: str) -> tuple[Any, ...]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise DocsLoadError(f"line {number}: invalid JSON: {error}") from error
    if not isinstance(record, dict):
        raise DocsLoadError(f"line {number}: expected an object")
    values: list[Any] = []
    for column in COLUMNS[:-1]:
        if column not in record:
            raise DocsLoadError(f"line {number}: missing {column}")
        values.append(record[column])
    if not isinstance(values[3], list) or not all(
        isinstance(item, str) for item in values[3]
    ):
        raise DocsLoadError(f"line {number}: heading_path must be a list of strings")
    values.append(snapshot_version)
    return tuple(values)


TRUNCATE_STATEMENT = "TRUNCATE TABLE omni_docs_qa.sections"


def load(
    cursor: Cursor, rows: Iterator[tuple[Any, ...]], *, replace: bool = False
) -> int:
    """Copy rows into the sections table.

    With `replace`, the table is truncated first so the new snapshot is the
    only one visible to the Omni view; without it rows are appended and a
    repeated section_id fails on the primary key.
    """
    cursor.execute(NEON_DDL)
    if replace:
        cursor.execute(TRUNCATE_STATEMENT)
    count = 0
    with cursor.copy(COPY_STATEMENT) as copy:
        for row in rows:
            copy.write_row(row)
            count += 1
    return count


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a corpus JSONL into omni_docs_qa.sections. The administrator DSN "
            f"is read from {DSN_ENVIRONMENT}; it is never accepted as a flag."
        )
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--snapshot-version")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="truncate omni_docs_qa.sections before loading (snapshot swap)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_arg_parser().parse_args(argv)
    try:
        return _run(arguments)
    except DocsLoadError as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1


def _run(arguments: argparse.Namespace) -> int:
    corpus_path = Path(arguments.corpus)
    version = arguments.snapshot_version or snapshot_version_from_meta(
        corpus_path.with_suffix(".meta.json")
    )
    if arguments.dry_run:
        count = sum(1 for _ in read_rows(corpus_path, version))
        print(
            json.dumps(
                {
                    "rows": count,
                    "snapshot_version": version,
                    "replace": arguments.replace,
                    "ddl": NEON_DDL,
                }
            )
        )
        return 0
    dsn = os.environ.get(DSN_ENVIRONMENT, "")
    if not dsn.strip():
        raise DocsLoadError(f"{DSN_ENVIRONMENT} is not set")
    import psycopg

    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        count = load(cursor, read_rows(corpus_path, version), replace=arguments.replace)
    print(
        json.dumps(
            {"rows": count, "snapshot_version": version, "replaced": arguments.replace}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

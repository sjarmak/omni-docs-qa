"""Docs QA gap queue builder and proposed-edit applier.

See the docs-qa shared contract (product bet 4), section 7, for the gap row
shape, the proposed-edit schema, and the apply rule this module implements.
This module is purely offline: it never calls the Omni CLI. The proposed
edit text (replacement bodies, new section bodies) is authored by a model or
a human elsewhere; this module only validates and applies it.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_BODY_CHARS = 20000
GAP_LABELS = frozenset({"partial", "not_covered", "conflicting"})
LABELS = frozenset({"answerable", "partial", "not_covered", "conflicting"})
VERDICT_KEYS = frozenset(
    {"verdict", "answer", "cited_section_ids", "cited_page_urls", "reason"}
)
EDIT_TOP_KEYS = frozenset(
    {"schema_version", "gap_id", "question_id", "author", "operations"}
)
EDIT_OP_KEYS = {
    "replace_body": frozenset({"op", "section_id", "body"}),
    "append_section": frozenset(
        {
            "op",
            "anchor_section_id",
            "placement",
            "section_id",
            "page_url",
            "page_title",
            "heading_path",
            "body",
        }
    ),
    "delete_section": frozenset({"op", "section_id"}),
}


class DocsGapsError(RuntimeError):
    """A gap queue or proposed-edit document is invalid, or an apply failed."""


@dataclass(frozen=True)
class ResultRow:
    question_id: str
    label: str
    request: dict[str, Any]
    verdict: dict[str, Any] | None
    parse_error: str | None


@dataclass(frozen=True)
class GapRow:
    gap_id: str
    question_id: str
    question: str
    expected_label: str
    predicted_verdict: str
    reason: str
    cited_section_ids: tuple[str, ...]
    cited_page_urls: tuple[str, ...]
    source_results: str
    status: str
    resolved_in_snapshot: str | None

    def to_row(self) -> dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "question_id": self.question_id,
            "question": self.question,
            "expected_label": self.expected_label,
            "predicted_verdict": self.predicted_verdict,
            "reason": self.reason,
            "cited_section_ids": list(self.cited_section_ids),
            "cited_page_urls": list(self.cited_page_urls),
            "source_results": self.source_results,
            "status": self.status,
            "resolved_in_snapshot": self.resolved_in_snapshot,
        }


@dataclass(frozen=True)
class EditDocument:
    gap_id: str
    question_id: str
    author: str
    operations: tuple[dict[str, Any], ...]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DocsGapsError(f"cannot read {path}: {error}") from error
    rows = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DocsGapsError(
                f"{path} line {line_number} is not valid JSON: {error}"
            ) from error
        if not isinstance(row, dict):
            raise DocsGapsError(f"{path} line {line_number} must be a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                )
                + "\n"
            )


def load_question_texts(path: Path) -> dict[str, str]:
    """Read questions.json and return {question_id: question text}.

    Minimal, structural parsing only: full schema and cross-reference
    validation of this file is questions' responsibility. gaps
    only needs the question text for each id to populate gap rows, and
    cannot import questions (contract section 9 forbids cross-module
    imports other than omni_cli).
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DocsGapsError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise DocsGapsError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("questions"), list):
        raise DocsGapsError(f"{path} must be a JSON object with a questions list")

    texts: dict[str, str] = {}
    for item in raw["questions"]:
        if not isinstance(item, dict):
            raise DocsGapsError(f"{path}: each question must be a JSON object")
        question_id = item.get("question_id")
        question = item.get("question")
        if not isinstance(question_id, str) or not question_id:
            raise DocsGapsError(f"{path}: question missing question_id")
        if not isinstance(question, str) or not question:
            raise DocsGapsError(f"{path}: {question_id} missing question text")
        texts[question_id] = question
    return texts


def load_results(path: Path) -> tuple[ResultRow, ...]:
    """Read a verdict results JSONL file (see contract section 5)."""
    rows = read_jsonl(path)
    results = []
    for row in rows:
        question_id = row.get("question_id")
        label = row.get("label")
        request = row.get("request")
        verdict = row.get("verdict")
        parse_error = row.get("parse_error")
        if not isinstance(question_id, str) or not question_id:
            raise DocsGapsError("result row missing question_id")
        if label not in LABELS:
            raise DocsGapsError(f"{question_id}: invalid label {label!r}")
        if not isinstance(request, dict) or not isinstance(request.get("prompt"), str):
            raise DocsGapsError(f"{question_id}: request must carry a string prompt")
        has_verdict = verdict is not None
        has_error = parse_error is not None
        if has_verdict == has_error:
            raise DocsGapsError(
                f"{question_id}: exactly one of verdict or parse_error must be set"
            )
        if has_verdict:
            _validate_verdict(question_id, verdict)
        results.append(
            ResultRow(
                question_id=question_id,
                label=label,
                request=request,
                verdict=verdict,
                parse_error=parse_error,
            )
        )
    return tuple(results)


def _validate_string_list(question_id: str, field: str, value: Any) -> None:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise DocsGapsError(
            f"{question_id}: {field} must be a list of non-empty strings"
        )
    if len(set(value)) != len(value):
        raise DocsGapsError(f"{question_id}: {field} entries must be unique")


def _validate_verdict(question_id: str, verdict: Any) -> None:
    if not isinstance(verdict, dict) or set(verdict) != VERDICT_KEYS:
        raise DocsGapsError(
            f"{question_id}: verdict must have exactly the required keys"
        )
    if verdict["verdict"] not in LABELS:
        raise DocsGapsError(
            f"{question_id}: invalid verdict value {verdict['verdict']!r}"
        )
    if not isinstance(verdict["answer"], str):
        raise DocsGapsError(f"{question_id}: verdict answer must be a string")
    if not isinstance(verdict["reason"], str) or not verdict["reason"]:
        raise DocsGapsError(f"{question_id}: verdict reason must be a non-empty string")
    _validate_string_list(
        question_id, "cited_section_ids", verdict["cited_section_ids"]
    )
    _validate_string_list(question_id, "cited_page_urls", verdict["cited_page_urls"])
    if verdict["verdict"] == "not_covered":
        if verdict["cited_section_ids"] or verdict["cited_page_urls"]:
            raise DocsGapsError(
                f"{question_id}: not_covered verdict must have empty cited lists"
            )
    elif not verdict["cited_section_ids"]:
        raise DocsGapsError(
            f"{question_id}: {verdict['verdict']} verdict needs non-empty "
            "cited_section_ids"
        )


def build_gap_queue(
    results: tuple[ResultRow, ...],
    *,
    source_results: str,
    question_text_by_id: dict[str, str],
) -> tuple[GapRow, ...]:
    """Select gap rows (contract section 7): pure selection by parsed verdict."""
    gaps = []
    for result in results:
        if result.verdict is None or result.verdict["verdict"] not in GAP_LABELS:
            continue
        question_text = question_text_by_id.get(result.question_id)
        if question_text is None:
            raise DocsGapsError(
                f"{result.question_id}: not found in the supplied question set"
            )
        gaps.append(
            GapRow(
                gap_id=f"gap_{result.question_id}",
                question_id=result.question_id,
                question=question_text,
                expected_label=result.label,
                predicted_verdict=result.verdict["verdict"],
                reason=result.verdict["reason"],
                cited_section_ids=tuple(result.verdict["cited_section_ids"]),
                cited_page_urls=tuple(result.verdict["cited_page_urls"]),
                source_results=source_results,
                status="open",
                resolved_in_snapshot=None,
            )
        )
    return tuple(sorted(gaps, key=lambda gap: gap.question_id))


def load_edit_file(path: Path) -> EditDocument:
    """Parse and structurally validate a proposed-edit document (section 7)."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DocsGapsError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise DocsGapsError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise DocsGapsError("edit document must be a JSON object")
    missing = EDIT_TOP_KEYS - set(raw)
    if missing:
        raise DocsGapsError(f"edit document missing keys: {sorted(missing)}")
    extra = set(raw) - EDIT_TOP_KEYS
    if extra:
        raise DocsGapsError(f"edit document has unexpected keys: {sorted(extra)}")
    if raw["schema_version"] != 1:
        raise DocsGapsError("edit document schema_version must be 1")
    gap_id = raw["gap_id"]
    question_id = raw["question_id"]
    author = raw["author"]
    if not isinstance(gap_id, str) or not gap_id:
        raise DocsGapsError("edit document gap_id must be a non-empty string")
    if not isinstance(question_id, str) or not question_id:
        raise DocsGapsError("edit document question_id must be a non-empty string")
    if not isinstance(author, str) or not author:
        raise DocsGapsError("edit document author must be a non-empty string")
    operations = raw["operations"]
    if not isinstance(operations, list) or not operations:
        raise DocsGapsError("edit document operations must be a non-empty list")
    validated_operations = tuple(_validate_operation(op) for op in operations)
    return EditDocument(
        gap_id=gap_id,
        question_id=question_id,
        author=author,
        operations=validated_operations,
    )


def _validate_operation(operation: Any) -> dict[str, Any]:
    if not isinstance(operation, dict) or "op" not in operation:
        raise DocsGapsError("each operation must be an object with an 'op' key")
    op = operation["op"]
    if op not in EDIT_OP_KEYS:
        raise DocsGapsError(f"unknown op: {op!r}")
    required_keys = EDIT_OP_KEYS[op]
    missing = required_keys - set(operation)
    if missing:
        raise DocsGapsError(f"{op} operation missing keys: {sorted(missing)}")
    extra = set(operation) - required_keys
    if extra:
        raise DocsGapsError(f"{op} operation has unexpected keys: {sorted(extra)}")
    if op == "append_section":
        if operation["placement"] not in ("before", "after"):
            raise DocsGapsError(f"invalid placement: {operation['placement']!r}")
        if not isinstance(operation["heading_path"], list):
            raise DocsGapsError("append_section heading_path must be a list")
    if "body" in operation:
        body = operation["body"]
        if not isinstance(body, str) or not body or len(body) > MAX_BODY_CHARS:
            raise DocsGapsError(f"{op} operation body is invalid")
    return operation


def apply_edits(
    corpus_rows: list[dict[str, Any]], edits: list[EditDocument]
) -> list[dict[str, Any]]:
    """Apply proposed edits, sorted by gap_id, to a copy of the corpus rows."""
    rows = [dict(row) for row in corpus_rows]
    for edit in sorted(edits, key=lambda edit: edit.gap_id):
        for operation in edit.operations:
            rows = _apply_operation(rows, operation)
    return _recompute_section_order(rows)


def _apply_operation(
    rows: list[dict[str, Any]], operation: dict[str, Any]
) -> list[dict[str, Any]]:
    op = operation["op"]
    if op == "replace_body":
        return _apply_replace_body(rows, operation)
    if op == "append_section":
        return _apply_append_section(rows, operation)
    return _apply_delete_section(rows, operation)


def _find_index(rows: list[dict[str, Any]], section_id: str) -> int:
    for index, row in enumerate(rows):
        if row["section_id"] == section_id:
            return index
    raise DocsGapsError(f"referenced section_id not found in corpus: {section_id}")


def _apply_replace_body(
    rows: list[dict[str, Any]], operation: dict[str, Any]
) -> list[dict[str, Any]]:
    index = _find_index(rows, operation["section_id"])
    body = operation["body"]
    new_row = dict(rows[index])
    new_row["body"] = body
    new_row["content_hash"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return [*rows[:index], new_row, *rows[index + 1 :]]


def _apply_append_section(
    rows: list[dict[str, Any]], operation: dict[str, Any]
) -> list[dict[str, Any]]:
    section_id = operation["section_id"]
    if any(row["section_id"] == section_id for row in rows):
        raise DocsGapsError(f"append_section section_id already exists: {section_id}")
    anchor_index = _find_index(rows, operation["anchor_section_id"])
    heading_path = list(operation["heading_path"])
    body = operation["body"]
    heading_label = (
        " > ".join(heading_path) if heading_path else operation["page_title"]
    )
    new_row = {
        "section_id": section_id,
        "page_url": operation["page_url"],
        "page_title": operation["page_title"],
        "heading_path": heading_path,
        "heading_label": heading_label,
        "section_order": -1,
        "part_index": 0,
        "part_count": 1,
        "body": body,
        "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "source_kind": "injected" if section_id.startswith("inj_") else "docs",
    }
    insert_at = anchor_index + 1 if operation["placement"] == "after" else anchor_index
    return [*rows[:insert_at], new_row, *rows[insert_at:]]


def _apply_delete_section(
    rows: list[dict[str, Any]], operation: dict[str, Any]
) -> list[dict[str, Any]]:
    index = _find_index(rows, operation["section_id"])
    return [*rows[:index], *rows[index + 1 :]]


def _recompute_section_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[str, int] = {}
    renumbered = []
    for row in rows:
        order = counters.get(row["page_url"], 0)
        counters[row["page_url"]] = order + 1
        new_row = dict(row)
        new_row["section_order"] = order
        renumbered.append(new_row)
    return renumbered


def next_snapshot_version(snapshots_dir: Path) -> str:
    if not snapshots_dir.is_dir():
        return "v001"
    max_seen = 0
    for entry in snapshots_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("v") and entry.name[1:].isdigit():
            max_seen = max(max_seen, int(entry.name[1:]))
    return f"v{max_seen + 1:03d}"


def write_snapshot(
    snapshots_dir: Path,
    version: str,
    rows: list[dict[str, Any]],
    *,
    parent_version: str | None,
    applied_gap_ids: list[str],
) -> Path:
    snapshot_dir = snapshots_dir / version
    if snapshot_dir.exists():
        raise DocsGapsError(f"snapshot directory already exists: {snapshot_dir}")
    corpus_path = snapshot_dir / "corpus.jsonl"
    write_jsonl(corpus_path, rows)
    corpus_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    meta = {
        "version": version,
        "parent_version": parent_version,
        "section_count": len(rows),
        "corpus_hash": corpus_hash,
        "applied_gap_ids": sorted(applied_gap_ids),
    }
    (snapshot_dir / "corpus.meta.json").write_text(
        json.dumps(meta, sort_keys=True) + "\n", encoding="utf-8"
    )
    return snapshot_dir


def diff_changed_sections(
    before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]
) -> str:
    before_by_id = {row["section_id"]: row for row in before_rows}
    after_by_id = {row["section_id"]: row for row in after_rows}
    blocks = []
    for section_id in sorted(set(before_by_id) | set(after_by_id)):
        before = before_by_id.get(section_id)
        after = after_by_id.get(section_id)
        if before == after:
            continue
        before_lines = (before["body"].splitlines(keepends=True)) if before else []
        after_lines = (after["body"].splitlines(keepends=True)) if after else []
        diff = difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"{section_id} (before)",
            tofile=f"{section_id} (after)",
        )
        blocks.append("".join(diff))
    return "\n".join(block for block in blocks if block)


def _resolve_queue(
    queue_rows: list[dict[str, Any]], applied_gap_ids: list[str], version: str
) -> list[dict[str, Any]]:
    applied = set(applied_gap_ids)
    resolved = []
    for row in queue_rows:
        new_row = dict(row)
        if row.get("gap_id") in applied:
            new_row["status"] = "resolved"
            new_row["resolved_in_snapshot"] = version
        resolved.append(new_row)
    return resolved


def _run_build_queue(args: argparse.Namespace) -> int:
    if not args.questions:
        raise DocsGapsError("--results requires --questions")
    results = load_results(Path(args.results))
    question_text_by_id = load_question_texts(Path(args.questions))
    queue = build_gap_queue(
        results, source_results=args.results, question_text_by_id=question_text_by_id
    )
    out_path = Path(args.out)
    write_jsonl(out_path, [gap.to_row() for gap in queue])
    print(
        json.dumps(
            {"gaps": len(queue), "output": str(out_path)},
            sort_keys=True,
        )
    )
    return 0


def _load_parent_version(corpus_path: Path) -> str | None:
    meta_path = corpus_path.parent / "corpus.meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DocsGapsError(f"cannot read {meta_path}: {error}") from error
    version = meta.get("version") if isinstance(meta, dict) else None
    return version if isinstance(version, str) else None


def _load_edits(args: argparse.Namespace) -> tuple[Path, list]:
    if not args.corpus:
        raise DocsGapsError("--apply requires --corpus")
    if not args.edits:
        raise DocsGapsError("--apply requires --edits")
    edits_dir = Path(args.edits)
    edit_paths = sorted(edits_dir.glob("*.json"))
    if not edit_paths:
        raise DocsGapsError(f"no edit files found under {edits_dir}")
    return Path(args.corpus), [load_edit_file(path) for path in edit_paths]


def _write_resolved_queue_if_requested(
    args: argparse.Namespace, applied_gap_ids: list[str], version: str
) -> None:
    if not args.queue:
        return
    if not args.queue_out:
        raise DocsGapsError("--queue requires --queue-out")
    queue_rows = read_jsonl(Path(args.queue))
    resolved_rows = _resolve_queue(queue_rows, applied_gap_ids, version)
    write_jsonl(Path(args.queue_out), resolved_rows)


def _run_apply(args: argparse.Namespace) -> int:
    corpus_path, edits = _load_edits(args)
    corpus_rows = read_jsonl(corpus_path)
    new_rows = apply_edits(corpus_rows, edits)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "diff": diff_changed_sections(corpus_rows, new_rows),
                    "gap_ids": sorted({edit.gap_id for edit in edits}),
                },
                sort_keys=True,
            )
        )
        return 0

    snapshots_dir = Path(args.snapshots_dir)
    version = args.version or next_snapshot_version(snapshots_dir)
    applied_gap_ids = sorted({edit.gap_id for edit in edits})
    snapshot_dir = write_snapshot(
        snapshots_dir,
        version,
        new_rows,
        parent_version=_load_parent_version(corpus_path),
        applied_gap_ids=applied_gap_ids,
    )

    _write_resolved_queue_if_requested(args, applied_gap_ids, version)

    print(
        json.dumps(
            {
                "version": version,
                "snapshot": str(snapshot_dir),
                "section_count": len(new_rows),
                "applied_gap_ids": applied_gap_ids,
            },
            sort_keys=True,
        )
    )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_docs_qa.gaps", description=__doc__
    )
    parser.add_argument("--results")
    parser.add_argument("--questions")
    parser.add_argument("--out", default="artifacts/docs_qa/gaps/queue.jsonl")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--edits")
    parser.add_argument("--corpus")
    parser.add_argument("--version")
    parser.add_argument("--snapshots-dir", default="artifacts/docs_qa/snapshots")
    parser.add_argument("--queue")
    parser.add_argument("--queue-out")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.apply:
            return _run_apply(args)
        if args.results:
            return _run_build_queue(args)
        raise DocsGapsError("either --results or --apply is required")
    except DocsGapsError as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

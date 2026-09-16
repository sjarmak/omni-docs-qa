"""Split the Omni llms-full documentation export into deterministic section rows.

See the docs-QA shared contract (bead az8.9) sections 1-3 for the exact
parsing, id, and splice rules this module implements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

UNIT = "\x1f"
MAX_BODY_CHARS = 20000

_PAGE_TITLE_RE = re.compile(r"^# (?P<title>.+)$")
_SOURCE_RE = re.compile(r"^Source: (?P<url>\S+)\s*$")
_SECTION_RE = re.compile(r"^(?P<hashes>#{2,4}) (?P<text>.+)$")
_INJECTED_ID_RE = re.compile(r"^inj_[a-z0-9_]{1,48}$")

NEON_DDL = """CREATE SCHEMA IF NOT EXISTS omni_docs_qa;
CREATE TABLE IF NOT EXISTS omni_docs_qa.sections (
    section_id       text PRIMARY KEY,
    page_url         text NOT NULL,
    page_title       text NOT NULL,
    heading_path     text[] NOT NULL,
    heading_label    text NOT NULL,
    section_order    integer NOT NULL,
    part_index       integer NOT NULL,
    part_count       integer NOT NULL,
    body             text NOT NULL,
    content_hash     text NOT NULL,
    source_kind      text NOT NULL,
    snapshot_version text NOT NULL
);
CREATE INDEX IF NOT EXISTS sections_page_url_idx ON omni_docs_qa.sections (page_url);
CREATE INDEX IF NOT EXISTS sections_source_kind_idx ON omni_docs_qa.sections (source_kind);"""

LOAD_PLAN = (
    "\\copy omni_docs_qa.sections FROM 'sections.csv' WITH (FORMAT csv, HEADER true)"
)


class DocsCorpusError(RuntimeError):
    """The llms-full export or an injected-sections file cannot be parsed safely."""


@dataclass(frozen=True)
class Section:
    section_id: str
    page_url: str
    page_title: str
    heading_path: tuple[str, ...]
    heading_label: str
    section_order: int
    part_index: int
    part_count: int
    body: str
    content_hash: str
    source_kind: str

    def to_row(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "page_url": self.page_url,
            "page_title": self.page_title,
            "heading_path": list(self.heading_path),
            "heading_label": self.heading_label,
            "section_order": self.section_order,
            "part_index": self.part_index,
            "part_count": self.part_count,
            "body": self.body,
            "content_hash": self.content_hash,
            "source_kind": self.source_kind,
        }


def section_id(
    page_url: str, heading_path: list[str], duplicate_index: int, part_index: int
) -> str:
    seed = UNIT.join(
        [page_url, UNIT.join(heading_path), str(duplicate_index), str(part_index)]
    )
    return "sec_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _heading_label(heading_path: tuple[str, ...], page_title: str) -> str:
    return " > ".join(heading_path) if heading_path else page_title


def _strip_blank_edges(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and lines[start].strip() == "":
        start += 1
    while end > start and lines[end - 1].strip() == "":
        end -= 1
    return lines[start:end]


def _chunk_body(body: str, max_chars: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in body.split("\n"):
        if len(line) > max_chars:
            if current:
                parts.append("\n".join(current))
                current, current_len = [], 0
            for start in range(0, len(line), max_chars):
                parts.append(line[start : start + max_chars])
            continue
        added = len(line) if not current else len(line) + 1
        if current and current_len + added > max_chars:
            parts.append("\n".join(current))
            current, current_len = [line], len(line)
        else:
            current.append(line)
            current_len += added
    if current:
        parts.append("\n".join(current))
    return parts


@dataclass
class _PageBlock:
    title: str
    url: str
    lines: list[str]


def _split_pages(text: str) -> list[_PageBlock]:
    lines = text.split("\n")
    pages: list[_PageBlock] = []
    fenced = False
    current: _PageBlock | None = None
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if (title_match := _PAGE_TITLE_RE.match(line)) and i + 1 < n:
            source_match = _SOURCE_RE.match(lines[i + 1])
            if source_match:
                current = _PageBlock(
                    title_match["title"].strip(), source_match["url"], []
                )
                pages.append(current)
                fenced = False
                i += 2
                continue
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if current is not None:
            current.lines.append(line)
        i += 1
    return pages


def _sections_for_page(page: _PageBlock, max_body_chars: int) -> list[Section]:
    rows: list[Section] = []
    heading_stack: list[str] = []
    body_lines: list[str] = []
    duplicate_counts: dict[tuple[str, ...], int] = {}
    order = 0
    fenced = False

    def flush(path: tuple[str, ...]) -> None:
        nonlocal order
        body = "\n".join(_strip_blank_edges(body_lines))
        if not body:
            return
        duplicate_index = duplicate_counts.get(path, 0)
        duplicate_counts[path] = duplicate_index + 1
        parts = _chunk_body(body, max_body_chars)
        for part_index, part_body in enumerate(parts):
            rows.append(
                Section(
                    section_id=section_id(
                        page.url, list(path), duplicate_index, part_index
                    ),
                    page_url=page.url,
                    page_title=page.title,
                    heading_path=path,
                    heading_label=_heading_label(path, page.title),
                    section_order=order,
                    part_index=part_index,
                    part_count=len(parts),
                    body=part_body,
                    content_hash=hashlib.sha256(part_body.encode("utf-8")).hexdigest(),
                    source_kind="docs",
                )
            )
            order += 1

    for line in page.lines:
        if not fenced and (section_match := _SECTION_RE.match(line)):
            flush(tuple(heading_stack))
            depth = len(section_match["hashes"]) - 2
            heading_stack = heading_stack[:depth]
            heading_stack.append(section_match["text"].strip())
            body_lines = []
            continue
        if line.lstrip().startswith("```"):
            fenced = not fenced
        body_lines.append(line)

    flush(tuple(heading_stack))
    return rows


def parse_docs_sections(
    text: str, *, max_body_chars: int = MAX_BODY_CHARS
) -> list[Section]:
    sections: list[Section] = []
    for page in _split_pages(text):
        sections.extend(_sections_for_page(page, max_body_chars))
    return sections


def load_injected_sections(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as error:
        raise DocsCorpusError(
            f"cannot read injected sections file {path}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise DocsCorpusError(
            f"injected sections file is not valid JSON: {error}"
        ) from error
    if not isinstance(payload, dict) or not isinstance(payload.get("sections"), list):
        raise DocsCorpusError("injected sections file must have a 'sections' list")
    if payload.get("schema_version") != 1:
        raise DocsCorpusError("injected sections file schema_version must be 1")
    seen_ids: set[str] = set()
    sections: list[dict[str, Any]] = []
    for entry in payload["sections"]:
        sections.append(_validate_injected_section(entry, seen_ids))
    return sections


_REQUIRED_INJECTED_KEYS = frozenset(
    {
        "section_id",
        "page_url",
        "page_title",
        "heading_path",
        "body",
        "anchor_page_url",
        "anchor_heading_path",
        "placement",
    }
)


def _validate_injected_section(entry: Any, seen_ids: set[str]) -> dict[str, Any]:
    if not isinstance(entry, dict) or not _REQUIRED_INJECTED_KEYS.issubset(entry):
        raise DocsCorpusError("injected section is missing required keys")
    section_id_value = entry["section_id"]
    if not isinstance(section_id_value, str) or not _INJECTED_ID_RE.match(
        section_id_value
    ):
        raise DocsCorpusError(f"invalid injected section_id: {section_id_value!r}")
    if section_id_value in seen_ids:
        raise DocsCorpusError(f"duplicate injected section_id: {section_id_value}")
    seen_ids.add(section_id_value)
    if entry["placement"] not in ("before", "after"):
        raise DocsCorpusError(
            f"invalid placement for {section_id_value}: {entry['placement']!r}"
        )
    body = entry["body"]
    if not isinstance(body, str) or not body or len(body) > MAX_BODY_CHARS:
        raise DocsCorpusError(f"invalid body for injected section {section_id_value}")
    for field in ("page_url", "page_title", "anchor_page_url"):
        if not isinstance(entry[field], str) or not entry[field]:
            raise DocsCorpusError(
                f"{field} must be a non-empty string for {section_id_value}"
            )
    for field in ("heading_path", "anchor_heading_path"):
        value = entry[field]
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise DocsCorpusError(
                f"{field} must be a list of non-empty strings for {section_id_value}"
            )
    return entry


def splice_injected(
    docs_sections: list[Section], injected: list[dict[str, Any]]
) -> list[Section]:
    result = list(docs_sections)
    for entry in injected:
        anchor_indices = [
            index
            for index, section in enumerate(result)
            if section.page_url == entry["anchor_page_url"]
            and list(section.heading_path) == entry["anchor_heading_path"]
            and section.part_index == section.part_count - 1
        ]
        if len(anchor_indices) != 1:
            raise DocsCorpusError(
                f"injected section {entry['section_id']} anchor matched "
                f"{len(anchor_indices)} rows, expected exactly 1"
            )
        anchor_index = anchor_indices[0]
        heading_path = tuple(entry["heading_path"])
        new_row = Section(
            section_id=entry["section_id"],
            page_url=entry["page_url"],
            page_title=entry["page_title"],
            heading_path=heading_path,
            heading_label=_heading_label(heading_path, entry["page_title"]),
            section_order=-1,
            part_index=0,
            part_count=1,
            body=entry["body"],
            content_hash=hashlib.sha256(entry["body"].encode("utf-8")).hexdigest(),
            source_kind="injected",
        )
        insert_at = anchor_index + 1 if entry["placement"] == "after" else anchor_index
        result.insert(insert_at, new_row)
    return _renumber_section_order(result)


def _renumber_section_order(sections: list[Section]) -> list[Section]:
    counters: dict[str, int] = {}
    renumbered = []
    for section in sections:
        order = counters.get(section.page_url, 0)
        counters[section.page_url] = order + 1
        renumbered.append(replace(section, section_order=order))
    return renumbered


def build_corpus(
    text: str,
    *,
    injected: list[dict[str, Any]] | None = None,
    max_body_chars: int = MAX_BODY_CHARS,
) -> list[Section]:
    docs_sections = parse_docs_sections(text, max_body_chars=max_body_chars)
    if not injected:
        return docs_sections
    return splice_injected(docs_sections, injected)


def _write_jsonl(path: Path, sections: list[Section]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for section in sections:
            handle.write(
                json.dumps(
                    section.to_row(),
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _write_meta(corpus_path: Path, meta_path: Path, section_count: int) -> None:
    corpus_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    meta = {
        "version": None,
        "parent_version": None,
        "section_count": section_count,
        "corpus_hash": corpus_hash,
        "applied_gap_ids": [],
    }
    meta_path.write_text(json.dumps(meta, sort_keys=True) + "\n", encoding="utf-8")


def _summary(sections: list[Section]) -> dict[str, Any]:
    return {
        "section_count": len(sections),
        "page_count": len({s.page_url for s in sections}),
        "injected_count": sum(1 for s in sections if s.source_kind == "injected"),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", "--input", dest="source", required=True)
    parser.add_argument(
        "--out", "--output", dest="out", default="artifacts/docs_qa/corpus.jsonl"
    )
    parser.add_argument("--inject")
    parser.add_argument("--max-body-chars", type=int, default=MAX_BODY_CHARS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-sql")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_arg_parser().parse_args(argv)
    try:
        return _run(arguments)
    except DocsCorpusError as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1


def _run(arguments: argparse.Namespace) -> int:
    source_path = Path(arguments.source)
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as error:
        raise DocsCorpusError(f"cannot read source {source_path}: {error}") from error

    injected = (
        load_injected_sections(Path(arguments.inject)) if arguments.inject else None
    )
    sections = build_corpus(
        text, injected=injected, max_body_chars=arguments.max_body_chars
    )
    summary = _summary(sections)

    if arguments.dry_run:
        print(json.dumps({**summary, "ddl": NEON_DDL}, sort_keys=True))
        return 0

    if arguments.plan_sql:
        Path(arguments.plan_sql).write_text(
            NEON_DDL + "\n" + LOAD_PLAN + "\n", encoding="utf-8"
        )

    out_path = Path(arguments.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_path, sections)
    meta_path = out_path.with_suffix(".meta.json")
    _write_meta(out_path, meta_path, len(sections))
    print(
        json.dumps(
            {**summary, "output": str(out_path), "meta": str(meta_path)}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

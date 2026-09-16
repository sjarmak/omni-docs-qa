"""Schema and coverage validator for the docs QA labeled question set.

See the docs-qa shared contract (product bet 4), section 4, for the exact
schema, count, and cross-reference rules this module enforces.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

QUESTION_ID_RE = re.compile(r"^q[0-9]{3}$")
INJECTED_ID_RE = re.compile(r"^inj_[a-z0-9_]{1,48}$")
DEFAULT_URL_PREFIX = "https://"
LABELS = {"answerable", "partial", "not_covered", "conflicting"}
REQUIRED_COUNTS = {
    "answerable": 25,
    "partial": 10,
    "not_covered": 10,
    "conflicting": 10,
}
REQUIRED_TOTAL = 55
QUESTION_KEYS = {
    "question_id",
    "question",
    "label",
    "expected_cited_page_urls",
    "rationale",
    "injected_section_ids",
}


class DocsQuestionsError(RuntimeError):
    """The question set (or its cross-references) is invalid."""


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    label: str
    expected_cited_page_urls: tuple[str, ...]
    rationale: str
    injected_section_ids: tuple[str, ...]


@dataclass(frozen=True)
class QuestionSet:
    slug: str
    questions: tuple[Question, ...]


@dataclass(frozen=True)
class ValidationResult:
    questions: int
    by_label: dict[str, int]
    checks: tuple[str, ...]
    corpus_checked: bool


def _read_json(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise DocsQuestionsError(f"cannot read {path}: {error}") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise DocsQuestionsError(f"{path} is not valid JSON: {error}") from error


def load_questions(path: Path) -> QuestionSet:
    """Parse and structurally validate the questions.json document."""
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise DocsQuestionsError("questions document must be a JSON object")

    extra_top = set(raw) - {"schema_version", "question_set_slug", "questions"}
    if extra_top:
        raise DocsQuestionsError(f"unexpected top-level keys: {sorted(extra_top)}")

    if raw.get("schema_version") != 1:
        raise DocsQuestionsError("schema_version must be 1")

    slug = raw.get("question_set_slug")
    if not isinstance(slug, str) or not slug:
        raise DocsQuestionsError("question_set_slug must be a non-empty string")

    raw_questions = raw.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise DocsQuestionsError("questions must be a non-empty list")

    questions = tuple(_parse_question(item) for item in raw_questions)
    return QuestionSet(slug=slug, questions=questions)


def _parse_question(item: object) -> Question:
    if not isinstance(item, dict):
        raise DocsQuestionsError("each question must be a JSON object")

    extra = set(item) - QUESTION_KEYS
    if extra:
        raise DocsQuestionsError(f"unexpected question keys: {sorted(extra)}")
    missing = QUESTION_KEYS - set(item)
    if missing:
        raise DocsQuestionsError(f"question missing keys: {sorted(missing)}")

    for field in ("question_id", "question", "label", "rationale"):
        if not isinstance(item[field], str) or not item[field]:
            raise DocsQuestionsError(f"{field} must be a non-empty string")

    urls = item["expected_cited_page_urls"]
    injected = item["injected_section_ids"]
    if not isinstance(urls, list) or not all(isinstance(u, str) for u in urls):
        raise DocsQuestionsError("expected_cited_page_urls must be a list of strings")
    if not isinstance(injected, list) or not all(isinstance(i, str) for i in injected):
        raise DocsQuestionsError("injected_section_ids must be a list of strings")

    return Question(
        question_id=item["question_id"],
        question=item["question"],
        label=item["label"],
        expected_cited_page_urls=tuple(urls),
        rationale=item["rationale"],
        injected_section_ids=tuple(injected),
    )


def load_injected_ids(path: Path) -> frozenset[str]:
    """Read injected_sections.json and return its set of section ids.

    Only structural fields needed to resolve ids are validated here; full
    injected-section schema and splice validation belongs to corpus.
    """
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise DocsQuestionsError("injected sections document must be a JSON object")
    if raw.get("schema_version") != 1:
        raise DocsQuestionsError("injected sections schema_version must be 1")

    sections = raw.get("sections")
    if not isinstance(sections, list):
        raise DocsQuestionsError("injected sections document needs a sections list")

    ids: list[str] = []
    for section in sections:
        if not isinstance(section, dict) or "section_id" not in section:
            raise DocsQuestionsError("each injected section needs a section_id")
        section_id = section["section_id"]
        if not isinstance(section_id, str) or not INJECTED_ID_RE.match(section_id):
            raise DocsQuestionsError(f"invalid injected section_id: {section_id!r}")
        ids.append(section_id)

    if len(set(ids)) != len(ids):
        raise DocsQuestionsError("injected section_id values must be unique")
    return frozenset(ids)


def load_corpus_index(path: Path) -> tuple[frozenset[str], frozenset[str]]:
    """Read a corpus JSONL file and return (docs page_urls, injected ids)."""
    docs_urls: set[str] = set()
    injected_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DocsQuestionsError(f"cannot read corpus {path}: {error}") from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DocsQuestionsError(
                f"corpus {path} line {line_number} is not valid JSON: {error}"
            ) from error
        if not isinstance(row, dict):
            raise DocsQuestionsError(
                f"corpus {path} line {line_number} must be an object"
            )
        source_kind = row.get("source_kind")
        if source_kind == "docs":
            page_url = row.get("page_url")
            if isinstance(page_url, str):
                docs_urls.add(page_url)
        elif source_kind == "injected":
            section_id = row.get("section_id")
            if isinstance(section_id, str):
                injected_ids.add(section_id)

    return frozenset(docs_urls), frozenset(injected_ids)


def _validate_question_scalar_fields(question: Question) -> None:
    if not QUESTION_ID_RE.match(question.question_id):
        raise DocsQuestionsError(f"invalid question_id: {question.question_id!r}")

    text = question.question
    if not (10 <= len(text) <= 500) or not text.endswith("?"):
        raise DocsQuestionsError(
            f"{question.question_id}: question must be 10-500 chars and end in '?'"
        )

    if question.label not in LABELS:
        raise DocsQuestionsError(
            f"{question.question_id}: invalid label {question.label!r}"
        )

    if not (20 <= len(question.rationale) <= 500):
        raise DocsQuestionsError(
            f"{question.question_id}: rationale must be 20-500 chars"
        )


def _validate_question_lists(question: Question, url_prefix: str) -> None:
    for url in question.expected_cited_page_urls:
        if not url.startswith(url_prefix):
            raise DocsQuestionsError(
                f"{question.question_id}: expected_cited_page_urls entry "
                f"must start with {url_prefix!r}: {url!r}"
            )
    if len(set(question.expected_cited_page_urls)) != len(
        question.expected_cited_page_urls
    ):
        raise DocsQuestionsError(
            f"{question.question_id}: expected_cited_page_urls must be unique"
        )

    for section_id in question.injected_section_ids:
        if not INJECTED_ID_RE.match(section_id):
            raise DocsQuestionsError(
                f"{question.question_id}: invalid injected_section_ids entry {section_id!r}"
            )
    if len(set(question.injected_section_ids)) != len(question.injected_section_ids):
        raise DocsQuestionsError(
            f"{question.question_id}: injected_section_ids must be unique"
        )


def _validate_question_label_consistency(question: Question) -> None:
    if question.label == "not_covered":
        if question.expected_cited_page_urls:
            raise DocsQuestionsError(
                f"{question.question_id}: not_covered questions must have "
                "empty expected_cited_page_urls"
            )
    elif not question.expected_cited_page_urls:
        raise DocsQuestionsError(
            f"{question.question_id}: {question.label} questions need "
            "non-empty expected_cited_page_urls"
        )

    if question.label == "conflicting":
        if not question.injected_section_ids:
            raise DocsQuestionsError(
                f"{question.question_id}: conflicting questions need "
                "non-empty injected_section_ids"
            )
    elif question.injected_section_ids:
        raise DocsQuestionsError(
            f"{question.question_id}: injected_section_ids must be empty "
            f"for label {question.label!r}"
        )


def _validate_question_fields(
    question: Question, checks: list[str], url_prefix: str
) -> None:
    _validate_question_scalar_fields(question)
    _validate_question_lists(question, url_prefix)
    _validate_question_label_consistency(question)
    checks.append(f"question_fields:{question.question_id}")


def _validate_question_id_ordering(ids_seen: list[str], checks: list[str]) -> None:
    if len(set(ids_seen)) != len(ids_seen):
        raise DocsQuestionsError("question_id values must be unique")
    checks.append("question_id_unique")

    if ids_seen != sorted(ids_seen):
        raise DocsQuestionsError("question_id values must be ascending in file order")
    checks.append("question_id_ascending")


def _validate_counts(total: int, by_label: dict[str, int], checks: list[str]) -> None:
    if total != REQUIRED_TOTAL:
        raise DocsQuestionsError(
            f"expected {REQUIRED_TOTAL} questions total, found {total}"
        )
    for label, required in REQUIRED_COUNTS.items():
        if by_label[label] != required:
            raise DocsQuestionsError(
                f"expected {required} '{label}' questions, found {by_label[label]}"
            )
    checks.append("counts_by_label")


def _validate_injected_ids_exist(
    questions: tuple[Question, ...], injected_ids: frozenset[str], checks: list[str]
) -> None:
    for question in questions:
        for section_id in question.injected_section_ids:
            if section_id not in injected_ids:
                raise DocsQuestionsError(
                    f"{question.question_id}: injected_section_ids entry "
                    f"{section_id!r} not found in injected sections file"
                )
    checks.append("injected_ids_exist")


def _validate_against_corpus(
    questions: tuple[Question, ...],
    corpus_index: tuple[frozenset[str], frozenset[str]],
    checks: list[str],
) -> None:
    docs_urls, corpus_injected_ids = corpus_index
    for question in questions:
        for url in question.expected_cited_page_urls:
            if url not in docs_urls:
                raise DocsQuestionsError(
                    f"{question.question_id}: expected_cited_page_urls entry "
                    f"{url!r} does not resolve to a 'docs' row in the corpus"
                )
        for section_id in question.injected_section_ids:
            if section_id not in corpus_injected_ids:
                raise DocsQuestionsError(
                    f"{question.question_id}: injected_section_ids entry "
                    f"{section_id!r} does not resolve to an 'injected' row "
                    "in the corpus"
                )
    checks.append("corpus_expected_urls_resolve")
    checks.append("corpus_injected_ids_resolve")


def validate(
    doc: QuestionSet,
    *,
    strict_counts: bool = True,
    injected_ids: frozenset[str] | None = None,
    corpus_index: tuple[frozenset[str], frozenset[str]] | None = None,
    url_prefix: str = DEFAULT_URL_PREFIX,
) -> ValidationResult:
    if not url_prefix.startswith(("https://", "http://")):
        raise DocsQuestionsError(f"url_prefix must be an http(s) URL: {url_prefix!r}")
    checks: list[str] = []
    ids_seen: list[str] = []
    by_label: dict[str, int] = {label: 0 for label in LABELS}

    for question in doc.questions:
        _validate_question_fields(question, checks, url_prefix)
        ids_seen.append(question.question_id)
        by_label[question.label] += 1

    _validate_question_id_ordering(ids_seen, checks)

    total = len(doc.questions)
    if strict_counts:
        _validate_counts(total, by_label, checks)

    if injected_ids is not None:
        _validate_injected_ids_exist(doc.questions, injected_ids, checks)

    corpus_checked = corpus_index is not None
    if corpus_index is not None:
        _validate_against_corpus(doc.questions, corpus_index, checks)

    return ValidationResult(
        questions=total,
        by_label=by_label,
        checks=tuple(checks),
        corpus_checked=corpus_checked,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m omni_docs_qa.questions")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--injected", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--no-strict-counts", action="store_true")
    parser.add_argument(
        "--url-prefix",
        default=DEFAULT_URL_PREFIX,
        help="every expected_cited_page_urls entry must start with this "
        "(for example https://docs.example.com/)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        doc = load_questions(args.questions)
        injected_ids = load_injected_ids(args.injected) if args.injected else None
        corpus_index = load_corpus_index(args.corpus) if args.corpus else None
        result = validate(
            doc,
            strict_counts=not args.no_strict_counts,
            injected_ids=injected_ids,
            corpus_index=corpus_index,
            url_prefix=args.url_prefix,
        )
    except DocsQuestionsError as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "questions": result.questions,
                "by_label": result.by_label,
                "checks": list(result.checks),
                "corpus_checked": result.corpus_checked,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

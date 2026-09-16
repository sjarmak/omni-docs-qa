"""Regression prompt-set builder and deletion check for docs QA (bet 4).

See the docs-qa shared contract, section 8, for the exact body shapes,
chunking rule, expectation templates, and deletion-check set math this
module implements.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from omni_docs_qa.verdict import build_prompt
from omni_docs_qa.omni_cli import OmniCli

MAX_PROMPTS_PER_SET = 25
DEFAULT_ANSWERABLE_SAMPLE = 10
MAX_SLUG_LEN = 255
MAX_NAME_LEN = 255
MAX_DESCRIPTION_LEN = 1024
MAX_PROMPT_TEXT_LEN = 8000
MAX_EXPECTATION_LEN = 16000
MIN_REPEAT_COUNT = 1
MAX_REPEAT_COUNT = 10
SLUG_RE = re.compile(r"^[a-z][a-z0-9-]*$")

VERDICT_SHAPE_EXPECTATION = (
    'The reply is one JSON object; its "verdict" must be "answerable" and its '
    '"cited_page_urls" must include at least one of: {urls}.'
)
RESOLVED_GAP_EXPECTATION_TEMPLATE = (
    "After the documentation fix this question is answerable from the docs "
    "sections. " + VERDICT_SHAPE_EXPECTATION
)
ANSWERABLE_SAMPLE_EXPECTATION_TEMPLATE = (
    "Answerable from the docs sections. " + VERDICT_SHAPE_EXPECTATION
)


class DocsRegressionError(RuntimeError):
    """A regression prompt-set input, body, or deletion check is invalid."""


class Client(Protocol):
    def run(self, *arguments: str, stdin: str | None = None) -> Any: ...


@dataclass(frozen=True)
class GapQueueRow:
    gap_id: str
    question_id: str
    status: str
    resolved_in_snapshot: str | None


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    label: str
    expected_cited_page_urls: tuple[str, ...]


@dataclass(frozen=True)
class SectionRow:
    section_id: str
    page_url: str
    content_hash: str


@dataclass(frozen=True)
class Prompt:
    prompt_text: str
    expectation: str


@dataclass(frozen=True)
class PromptSetChunk:
    slug: str
    name: str
    prompts: tuple[Prompt, ...]


@dataclass(frozen=True)
class DeletionCheckResult:
    removed: tuple[str, ...]
    changed: tuple[str, ...]
    added: tuple[str, ...]
    fully_removed_question_ids: tuple[str, ...]
    partially_affected_question_ids: tuple[str, ...]


class OmniClient:
    def __init__(self, *, profile: str, binary: str = "omni") -> None:
        self._cli = OmniCli(profile=profile, binary=binary)

    def run(self, *arguments: str, stdin: str | None = None) -> Any:
        return self._cli.run_json(arguments, stdin=stdin)


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise DocsRegressionError(f"cannot read {path}: {error}") from error
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DocsRegressionError(
                f"{path} line {line_number} is not valid JSON: {error}"
            ) from error
        if not isinstance(row, dict):
            raise DocsRegressionError(f"{path} line {line_number} must be an object")
        rows.append((line_number, row))
    return rows


def load_queue(path: Path) -> tuple[GapQueueRow, ...]:
    """Read the fields of gaps/queue.jsonl this module needs: id and status."""
    rows = []
    for line_number, row in _read_jsonl(path):
        gap_id = row.get("gap_id")
        question_id = row.get("question_id")
        status = row.get("status")
        if not isinstance(gap_id, str) or not gap_id:
            raise DocsRegressionError(f"{path} line {line_number}: missing gap_id")
        if not isinstance(question_id, str) or not question_id:
            raise DocsRegressionError(f"{path} line {line_number}: missing question_id")
        if not isinstance(status, str) or not status:
            raise DocsRegressionError(f"{path} line {line_number}: missing status")
        resolved_in_snapshot = row.get("resolved_in_snapshot")
        if resolved_in_snapshot is not None and not isinstance(
            resolved_in_snapshot, str
        ):
            raise DocsRegressionError(
                f"{path} line {line_number}: resolved_in_snapshot must be a "
                "string or null"
            )
        rows.append(GapQueueRow(gap_id, question_id, status, resolved_in_snapshot))
    return tuple(rows)


def load_questions(path: Path) -> tuple[Question, ...]:
    """Read the fields of questions.json this module needs, by contract section 4."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DocsRegressionError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise DocsRegressionError(f"{path} is not valid JSON: {error}") from error

    if not isinstance(raw, dict) or not isinstance(raw.get("questions"), list):
        raise DocsRegressionError(f"{path} must have a top-level questions list")

    questions = []
    for item in raw["questions"]:
        if not isinstance(item, dict):
            raise DocsRegressionError(f"{path}: each question must be an object")
        question_id = item.get("question_id")
        text = item.get("question")
        label = item.get("label")
        urls = item.get("expected_cited_page_urls")
        if not isinstance(question_id, str) or not question_id:
            raise DocsRegressionError(f"{path}: a question is missing question_id")
        if not isinstance(text, str) or not text:
            raise DocsRegressionError(f"{question_id}: missing question text")
        if not isinstance(label, str) or not label:
            raise DocsRegressionError(f"{question_id}: missing label")
        if not isinstance(urls, list) or not all(isinstance(u, str) for u in urls):
            raise DocsRegressionError(
                f"{question_id}: expected_cited_page_urls must be a list of strings"
            )
        questions.append(Question(question_id, text, label, tuple(urls)))
    return tuple(questions)


def load_corpus_sections(path: Path) -> tuple[SectionRow, ...]:
    """Read the fields of a corpus.jsonl this module needs for the deletion check."""
    rows = []
    for line_number, row in _read_jsonl(path):
        section_id = row.get("section_id")
        page_url = row.get("page_url")
        content_hash = row.get("content_hash")
        if not isinstance(section_id, str) or not section_id:
            raise DocsRegressionError(f"{path} line {line_number}: missing section_id")
        if not isinstance(page_url, str) or not page_url:
            raise DocsRegressionError(f"{path} line {line_number}: missing page_url")
        if not isinstance(content_hash, str) or not content_hash:
            raise DocsRegressionError(
                f"{path} line {line_number}: missing content_hash"
            )
        rows.append(SectionRow(section_id, page_url, content_hash))
    return tuple(rows)


# ---------------------------------------------------------------------------
# prompt selection, chunking, request bodies
# ---------------------------------------------------------------------------


def _resolved_gap_prompts(
    queue_rows: tuple[GapQueueRow, ...],
    question_by_id: dict[str, Question],
) -> tuple[list[str], list[Prompt]]:
    resolved_ids = sorted(
        {row.question_id for row in queue_rows if row.status == "resolved"}
    )
    resolved_prompts = []
    for question_id in resolved_ids:
        question = question_by_id.get(question_id)
        if question is None:
            raise DocsRegressionError(
                f"resolved gap references unknown question_id {question_id!r}"
            )
        if not question.expected_cited_page_urls:
            raise DocsRegressionError(
                f"resolved gap {question_id!r} has no expected_cited_page_urls "
                "to build a citation expectation from"
            )
        resolved_prompts.append(
            Prompt(
                build_prompt(question.question),
                RESOLVED_GAP_EXPECTATION_TEMPLATE.format(
                    urls=", ".join(question.expected_cited_page_urls)
                ),
            )
        )
    return resolved_ids, resolved_prompts


def _answerable_sample_prompts(
    questions: tuple[Question, ...],
    question_by_id: dict[str, Question],
    resolved_id_set: set[str],
    answerable_sample: int,
) -> list[Prompt]:
    answerable_ids = sorted(
        question.question_id
        for question in questions
        if question.label == "answerable"
        and question.question_id not in resolved_id_set
    )
    sample_prompts = []
    for question_id in answerable_ids[:answerable_sample]:
        question = question_by_id[question_id]
        sample_prompts.append(
            Prompt(
                build_prompt(question.question),
                ANSWERABLE_SAMPLE_EXPECTATION_TEMPLATE.format(
                    urls=", ".join(question.expected_cited_page_urls)
                ),
            )
        )
    return sample_prompts


def select_prompts(
    queue_rows: tuple[GapQueueRow, ...],
    questions: tuple[Question, ...],
    answerable_sample: int,
) -> tuple[Prompt, ...]:
    """Resolved gap questions (by question_id) then an answerable sample."""
    if answerable_sample < 0:
        raise DocsRegressionError("--answerable-sample must be >= 0")

    question_by_id = {question.question_id: question for question in questions}
    resolved_ids, resolved_prompts = _resolved_gap_prompts(queue_rows, question_by_id)
    sample_prompts = _answerable_sample_prompts(
        questions, question_by_id, set(resolved_ids), answerable_sample
    )

    return tuple(resolved_prompts) + tuple(sample_prompts)


def chunk_prompts(
    prompts: tuple[Prompt, ...], chunk_size: int = MAX_PROMPTS_PER_SET
) -> tuple[tuple[Prompt, ...], ...]:
    return tuple(
        tuple(prompts[index : index + chunk_size])
        for index in range(0, len(prompts), chunk_size)
    )


def _validate_slug(slug: str) -> None:
    if not SLUG_RE.match(slug) or len(slug) > MAX_SLUG_LEN:
        raise DocsRegressionError(f"invalid prompt-set slug: {slug!r}")


def _validate_name(name: str) -> None:
    if not name or len(name) > MAX_NAME_LEN:
        raise DocsRegressionError(f"invalid prompt-set name: {name!r}")


def build_prompt_set_chunks(
    base_slug: str, base_name: str, prompts: tuple[Prompt, ...]
) -> tuple[PromptSetChunk, ...]:
    if not prompts:
        raise DocsRegressionError(
            "no prompts available to build a regression prompt set"
        )
    _validate_slug(base_slug)
    _validate_name(base_name)

    chunks = chunk_prompts(prompts)
    total = len(chunks)
    result = []
    for index, chunk in enumerate(chunks):
        if total == 1:
            slug, name = base_slug, base_name
        else:
            slug = f"{base_slug}-{index + 1}"
            name = f"{base_name} (part {index + 1} of {total})"
            _validate_slug(slug)
            _validate_name(name)
        result.append(PromptSetChunk(slug=slug, name=name, prompts=chunk))
    return tuple(result)


def prompt_set_body(
    model_id: str, chunk: PromptSetChunk, description: str | None = None
) -> dict[str, Any]:
    if not model_id:
        raise DocsRegressionError("--model-id is required to build a prompt set body")

    prompts_payload = []
    for prompt in chunk.prompts:
        if not (1 <= len(prompt.prompt_text) <= MAX_PROMPT_TEXT_LEN):
            raise DocsRegressionError(
                f"prompt_text must be 1-8000 characters: {prompt.prompt_text[:40]!r}..."
            )
        if len(prompt.expectation) > MAX_EXPECTATION_LEN:
            raise DocsRegressionError("expectation exceeds the 16000-character limit")
        prompts_payload.append(
            {"prompt_text": prompt.prompt_text, "expectation": prompt.expectation}
        )

    body: dict[str, Any] = {
        "model_id": model_id,
        "name": chunk.name,
        "slug": chunk.slug,
        "prompts": prompts_payload,
    }
    if description:
        if len(description) > MAX_DESCRIPTION_LEN:
            raise DocsRegressionError("description exceeds the 1024-character limit")
        body["description"] = description
    return body


def runs_create_body(
    prompt_set_id: str,
    *,
    description: str | None = None,
    branch_id: str | None = None,
    repeat_count: int | None = None,
) -> dict[str, Any]:
    if not prompt_set_id:
        raise DocsRegressionError(
            "prompt_set_id is required to build a runs-create body"
        )

    body: dict[str, Any] = {"prompt_set_id": prompt_set_id}
    if description:
        if len(description) > MAX_DESCRIPTION_LEN:
            raise DocsRegressionError("description exceeds the 1024-character limit")
        body["description"] = description

    run_config: dict[str, Any] = {}
    if branch_id:
        run_config["branch_id"] = branch_id
    if repeat_count is not None:
        if not (MIN_REPEAT_COUNT <= repeat_count <= MAX_REPEAT_COUNT):
            raise DocsRegressionError("--repeat-count must be between 1 and 10")
        run_config["repeat_count"] = repeat_count
    if run_config:
        body["run_config"] = run_config
    return body


# ---------------------------------------------------------------------------
# deletion check
# ---------------------------------------------------------------------------


def _diff_section_ids(
    baseline_rows: tuple[SectionRow, ...], candidate_rows: tuple[SectionRow, ...]
) -> tuple[set[str], set[str], set[str]]:
    baseline_index = {row.section_id: row.content_hash for row in baseline_rows}
    candidate_index = {row.section_id: row.content_hash for row in candidate_rows}
    baseline_ids = set(baseline_index)
    candidate_ids = set(candidate_index)

    removed = baseline_ids - candidate_ids
    added = candidate_ids - baseline_ids
    changed = {
        section_id
        for section_id in baseline_ids & candidate_ids
        if baseline_index[section_id] != candidate_index[section_id]
    }
    return removed, added, changed


def _affected_question_ids(
    target_ids: list[str],
    question_by_id: dict[str, Question],
    baseline_rows: tuple[SectionRow, ...],
    removed: set[str],
    changed: set[str],
) -> tuple[list[str], list[str]]:
    fully_removed = []
    partially_affected = []
    for question_id in target_ids:
        question = question_by_id.get(question_id)
        if question is None:
            raise DocsRegressionError(
                f"resolved gap references unknown question_id {question_id!r}"
            )
        urls = set(question.expected_cited_page_urls)
        supporting = {row.section_id for row in baseline_rows if row.page_url in urls}
        if not supporting:
            continue
        if supporting <= removed:
            fully_removed.append(question_id)
        elif supporting & (removed | changed):
            partially_affected.append(question_id)
    return fully_removed, partially_affected


def run_deletion_check(
    baseline_rows: tuple[SectionRow, ...],
    candidate_rows: tuple[SectionRow, ...],
    questions: tuple[Question, ...],
    resolved_gap_question_ids: tuple[str, ...],
) -> DeletionCheckResult:
    removed, added, changed = _diff_section_ids(baseline_rows, candidate_rows)

    question_by_id = {question.question_id: question for question in questions}
    target_ids = sorted(
        {
            question.question_id
            for question in questions
            if question.label == "answerable"
        }
        | set(resolved_gap_question_ids)
    )

    fully_removed, partially_affected = _affected_question_ids(
        target_ids, question_by_id, baseline_rows, removed, changed
    )

    return DeletionCheckResult(
        removed=tuple(sorted(removed)),
        changed=tuple(sorted(changed)),
        added=tuple(sorted(added)),
        fully_removed_question_ids=tuple(sorted(fully_removed)),
        partially_affected_question_ids=tuple(sorted(partially_affected)),
    )


# ---------------------------------------------------------------------------
# plan text and live execution
# ---------------------------------------------------------------------------


def plan_lines(
    *,
    profile: str,
    model_id: str,
    chunks: tuple[PromptSetChunk, ...],
    description: str | None,
    branch_id: str | None,
    repeat_count: int | None,
) -> tuple[str, ...]:
    lines: list[str] = []
    for chunk in chunks:
        lines.append(
            f"omni --compact --profile {profile} ai-eval prompt-sets-list "
            f"--model-ids {model_id}"
        )
        lines.append(
            f"omni --compact --profile {profile} ai-eval prompt-sets-create --body -"
        )
        lines.append(
            json.dumps(prompt_set_body(model_id, chunk, description), sort_keys=True)
        )
        lines.append(f"omni --compact --profile {profile} ai-eval runs-create --body -")
        lines.append(
            json.dumps(
                runs_create_body(
                    "<prompt-set-id>",
                    description=description,
                    branch_id=branch_id,
                    repeat_count=repeat_count,
                ),
                sort_keys=True,
            )
        )
        lines.append(f"omni --compact --profile {profile} ai-eval runs-get <run-id>")
    return tuple(lines)


RESPONSE_WRAPPERS = ("prompt_set", "run")


def _extract_id(response: Any, context: str) -> str:
    """Return the id of a created object.

    The Omni CLI wraps `prompt-sets-create` under `prompt_set` and run
    responses under `run`; older shapes carry `id` at the top level.
    """
    if isinstance(response, dict):
        candidates = [response] + [
            response[key]
            for key in RESPONSE_WRAPPERS
            if isinstance(response.get(key), dict)
        ]
        for candidate in candidates:
            value = candidate.get("id")
            if isinstance(value, str) and value:
                return value
    raise DocsRegressionError(f"{context} response carries no id")


def execute_live(
    client: Client,
    *,
    model_id: str,
    chunks: tuple[PromptSetChunk, ...],
    description: str | None,
    branch_id: str | None,
    repeat_count: int | None,
    prompt_set_id: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Create one prompt set per chunk and start a run against each.

    With `prompt_set_id`, the existing set is reused (a single chunk only) and
    no set is created, which is how a later snapshot is regressed against the
    same prompts.
    """
    if prompt_set_id is not None and len(chunks) != 1:
        raise DocsRegressionError(
            "--prompt-set-id reuses one existing set, but the prompts span "
            f"{len(chunks)} chunks"
        )
    results = []
    for chunk in chunks:
        set_id = prompt_set_id or _create_prompt_set(
            client, model_id, chunk, description
        )
        run_response = client.run(
            "ai-eval",
            "runs-create",
            "--body",
            "-",
            stdin=json.dumps(
                runs_create_body(
                    set_id,
                    description=description,
                    branch_id=branch_id,
                    repeat_count=repeat_count,
                ),
                sort_keys=True,
            ),
        )
        run_id = _extract_id(run_response, "runs-create")
        run_details = client.run("ai-eval", "runs-get", run_id)
        results.append(
            {
                "slug": chunk.slug,
                "prompt_set_id": set_id,
                "run_id": run_id,
                "run": run_details,
            }
        )
    return tuple(results)


def _create_prompt_set(
    client: Client, model_id: str, chunk: PromptSetChunk, description: str | None
) -> str:
    client.run("ai-eval", "prompt-sets-list", "--model-ids", model_id)
    create_response = client.run(
        "ai-eval",
        "prompt-sets-create",
        "--body",
        "-",
        stdin=json.dumps(prompt_set_body(model_id, chunk, description), sort_keys=True),
    )
    return _extract_id(create_response, "prompt-sets-create")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m omni_docs_qa.regression")
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--slug")
    parser.add_argument("--name")
    parser.add_argument("--description")
    parser.add_argument(
        "--answerable-sample", type=int, default=DEFAULT_ANSWERABLE_SAMPLE
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--deletion-check", action="store_true")
    parser.add_argument("--branch-id")
    parser.add_argument("--repeat-count", type=int)
    parser.add_argument(
        "--prompt-set-id",
        help="start the run against this existing prompt set instead of creating one",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("OMNI_PROFILE") or None,
        help="Omni CLI profile (defaults to $OMNI_PROFILE)",
    )
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--live", action="store_true")
    return parser


def _load_queue_and_questions(
    args: argparse.Namespace,
) -> tuple[tuple[GapQueueRow, ...], tuple[Question, ...]]:
    if not args.queue or not args.questions:
        raise DocsRegressionError("--queue and --questions are required")
    return load_queue(args.queue), load_questions(args.questions)


def _run_plan_or_live(args: argparse.Namespace) -> int:
    if not args.model_id or not args.slug or not args.name:
        raise DocsRegressionError("--model-id, --slug, and --name are required")
    queue_rows, questions = _load_queue_and_questions(args)
    prompts = select_prompts(queue_rows, questions, args.answerable_sample)
    chunks = build_prompt_set_chunks(args.slug, args.name, prompts)

    if args.live:
        client = OmniClient(profile=args.profile)
        results = execute_live(
            client,
            model_id=args.model_id,
            chunks=chunks,
            description=args.description,
            branch_id=args.branch_id,
            repeat_count=args.repeat_count,
            prompt_set_id=args.prompt_set_id,
        )
        print(json.dumps({"chunks": list(results)}, sort_keys=True))
        return 0

    for line in plan_lines(
        profile=args.profile or "<profile>",
        model_id=args.model_id,
        chunks=chunks,
        description=args.description,
        branch_id=args.branch_id,
        repeat_count=args.repeat_count,
    ):
        print(line)
    return 0


def _run_deletion_check(args: argparse.Namespace) -> int:
    if not args.baseline or not args.candidate:
        raise DocsRegressionError(
            "--baseline and --candidate are required for --deletion-check"
        )
    queue_rows, questions = _load_queue_and_questions(args)
    baseline_rows = load_corpus_sections(args.baseline)
    candidate_rows = load_corpus_sections(args.candidate)
    resolved_gap_ids = tuple(
        sorted({row.question_id for row in queue_rows if row.status == "resolved"})
    )
    result = run_deletion_check(
        baseline_rows, candidate_rows, questions, resolved_gap_ids
    )
    print(
        json.dumps(
            {
                "removed": list(result.removed),
                "changed": list(result.changed),
                "added": list(result.added),
                "fully_removed_question_ids": list(result.fully_removed_question_ids),
                "partially_affected_question_ids": list(
                    result.partially_affected_question_ids
                ),
            },
            sort_keys=True,
        )
    )
    return 1 if result.fully_removed_question_ids else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.live and args.profile is None:
        parser.error("--live needs --profile (or OMNI_PROFILE)")
    try:
        if args.deletion_check:
            return _run_deletion_check(args)
        if args.plan or args.live:
            return _run_plan_or_live(args)
        raise DocsRegressionError(
            "one of --deletion-check, --plan, or --live is required"
        )
    except DocsRegressionError as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

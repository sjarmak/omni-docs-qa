import json
from pathlib import Path

import pytest

from omni_docs_qa.verdict import build_prompt
from omni_docs_qa.regression import (
    DocsRegressionError,
    GapQueueRow,
    Prompt,
    Question,
    build_prompt_set_chunks,
    chunk_prompts,
    execute_live,
    load_corpus_sections,
    load_questions,
    load_queue,
    main,
    prompt_set_body,
    run_deletion_check,
    runs_create_body,
    select_prompts,
)

FIXTURES = Path(__file__).parent / "fixtures" / "docs_qa"
QUEUE = FIXTURES / "regression_queue.jsonl"
QUESTIONS = FIXTURES / "regression_questions.json"
BASELINE = FIXTURES / "regression_baseline.jsonl"
CANDIDATE = FIXTURES / "regression_candidate.jsonl"


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[tuple, str | None]] = []

    def run(self, *arguments, stdin=None):
        self.calls.append((arguments, stdin))
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def test_load_queue_reads_status_and_question_id() -> None:
    rows = load_queue(QUEUE)

    assert [(row.question_id, row.status) for row in rows] == [
        ("q004", "resolved"),
        ("q005", "open"),
    ]


def test_load_queue_rejects_missing_gap_id(tmp_path: Path) -> None:
    path = tmp_path / "queue.jsonl"
    path.write_text(json.dumps({"question_id": "q001", "status": "open"}) + "\n")

    with pytest.raises(DocsRegressionError, match="gap_id"):
        load_queue(path)


def test_load_questions_reads_label_and_urls() -> None:
    questions = load_questions(QUESTIONS)

    by_id = {q.question_id: q for q in questions}
    assert by_id["q001"].label == "answerable"
    assert by_id["q004"].expected_cited_page_urls == (
        "https://docs.omni.co/modeling/branches",
    )


def test_load_corpus_sections_reads_hash_index() -> None:
    rows = load_corpus_sections(BASELINE)

    assert {row.section_id for row in rows} == {
        "sec_pg",
        "sec_dash",
        "sec_share",
        "sec_branch",
        "sec_other",
    }


# ---------------------------------------------------------------------------
# prompt selection and chunking
# ---------------------------------------------------------------------------


def test_select_prompts_orders_resolved_gaps_then_answerable_sample() -> None:
    queue_rows = load_queue(QUEUE)
    questions = load_questions(QUESTIONS)

    prompts = select_prompts(queue_rows, questions, answerable_sample=2)

    assert [p.prompt_text for p in prompts] == [
        build_prompt(q)
        for q in (
            "How does Omni resolve a merge conflict on a shared model branch?",
            "How do I connect a Postgres database to Omni?",
            "How do I create a new dashboard in Omni?",
        )
    ]
    assert prompts[0].prompt_text.startswith("You are answering a question")
    assert "Never judge from truncated text." in prompts[0].prompt_text
    assert prompts[0].expectation.startswith(
        "After the documentation fix this question is answerable"
    )
    assert prompts[0].expectation.endswith(
        '"cited_page_urls" must include at least one of: '
        "https://docs.omni.co/modeling/branches."
    )
    assert prompts[1].expectation == (
        "Answerable from the docs sections. The reply is one JSON object; its "
        '"verdict" must be "answerable" and its "cited_page_urls" must include '
        "at least one of: https://docs.omni.co/connections/postgres."
    )


def test_select_prompts_rejects_resolved_gap_with_unknown_question(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "queue.jsonl"
    queue_path.write_text(
        json.dumps(
            {
                "gap_id": "gap_q999",
                "question_id": "q999",
                "status": "resolved",
            }
        )
        + "\n"
    )
    queue_rows = load_queue(queue_path)
    questions = load_questions(QUESTIONS)

    with pytest.raises(DocsRegressionError, match="q999"):
        select_prompts(queue_rows, questions, answerable_sample=1)


def test_select_prompts_rejects_resolved_gap_with_no_expected_urls() -> None:
    queue_rows = (
        GapQueueRow(
            gap_id="gap_q010",
            question_id="q010",
            status="resolved",
            resolved_in_snapshot="v001",
        ),
    )
    questions = (
        Question(
            question_id="q010",
            question="What happens if the docs never covered this?",
            label="not_covered",
            expected_cited_page_urls=(),
        ),
    )

    with pytest.raises(DocsRegressionError, match="q010"):
        select_prompts(queue_rows, questions, answerable_sample=0)


def test_select_prompts_does_not_double_count_a_resolved_answerable_question() -> None:
    queue_rows = (
        GapQueueRow(
            gap_id="gap_q001",
            question_id="q001",
            status="resolved",
            resolved_in_snapshot="v001",
        ),
    )
    questions = (
        Question(
            question_id="q001",
            question="How do I connect a Postgres database to Omni?",
            label="answerable",
            expected_cited_page_urls=("https://docs.omni.co/connections/postgres",),
        ),
        Question(
            question_id="q002",
            question="How do I create a new dashboard in Omni?",
            label="answerable",
            expected_cited_page_urls=("https://docs.omni.co/dashboards/create",),
        ),
    )

    prompts = select_prompts(queue_rows, questions, answerable_sample=5)

    assert [p.prompt_text for p in prompts] == [
        build_prompt("How do I connect a Postgres database to Omni?"),
        build_prompt("How do I create a new dashboard in Omni?"),
    ]
    assert prompts[0].expectation.startswith(
        "After the documentation fix this question is answerable"
    )
    assert prompts[1].expectation.startswith("Answerable from the docs sections")


def test_chunk_prompts_splits_at_25() -> None:
    prompts = tuple(Prompt(f"question {i}?", "expectation") for i in range(60))

    chunks = chunk_prompts(prompts)

    assert [len(c) for c in chunks] == [25, 25, 10]


def test_build_prompt_set_chunks_single_chunk_keeps_base_slug() -> None:
    prompts = (Prompt("q?", "e"),)

    chunks = build_prompt_set_chunks(
        "docs-qa-regression-v1", "Docs QA regression", prompts
    )

    assert len(chunks) == 1
    assert chunks[0].slug == "docs-qa-regression-v1"
    assert chunks[0].name == "Docs QA regression"


def test_build_prompt_set_chunks_multi_chunk_suffixes_slug_and_name() -> None:
    prompts = tuple(Prompt(f"q{i}?", "e") for i in range(30))

    chunks = build_prompt_set_chunks(
        "docs-qa-regression-v1", "Docs QA regression", prompts
    )

    assert len(chunks) == 2
    assert chunks[0].slug == "docs-qa-regression-v1-1"
    assert chunks[0].name == "Docs QA regression (part 1 of 2)"
    assert chunks[1].slug == "docs-qa-regression-v1-2"
    assert chunks[1].name == "Docs QA regression (part 2 of 2)"


def test_build_prompt_set_chunks_rejects_bad_slug() -> None:
    with pytest.raises(DocsRegressionError, match="slug"):
        build_prompt_set_chunks("Bad Slug", "Name", (Prompt("q?", "e"),))


def test_build_prompt_set_chunks_rejects_no_prompts() -> None:
    with pytest.raises(DocsRegressionError, match="no prompts"):
        build_prompt_set_chunks("slug", "Name", ())


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------


def test_prompt_set_body_shape() -> None:
    chunks = build_prompt_set_chunks(
        "docs-qa-regression-v1", "Docs QA regression", (Prompt("How?", "Cite it."),)
    )

    body = prompt_set_body("11111111-1111-1111-1111-111111111111", chunks[0], "desc")

    assert body == {
        "model_id": "11111111-1111-1111-1111-111111111111",
        "name": "Docs QA regression",
        "slug": "docs-qa-regression-v1",
        "description": "desc",
        "prompts": [{"prompt_text": "How?", "expectation": "Cite it."}],
    }


def test_prompt_set_body_omits_description_when_absent() -> None:
    chunks = build_prompt_set_chunks("slug", "Name", (Prompt("How?", "Cite it."),))

    body = prompt_set_body("model-id", chunks[0])

    assert "description" not in body


def test_prompt_set_body_rejects_overlong_prompt_text() -> None:
    chunks = build_prompt_set_chunks("slug", "Name", (Prompt("x" * 8001, "e"),))

    with pytest.raises(DocsRegressionError, match="prompt_text"):
        prompt_set_body("model-id", chunks[0])


def test_runs_create_body_minimal() -> None:
    body = runs_create_body("prompt-set-id")

    assert body == {"prompt_set_id": "prompt-set-id"}


def test_runs_create_body_with_run_config() -> None:
    body = runs_create_body(
        "prompt-set-id", branch_id="branch-1", repeat_count=3, description="d"
    )

    assert body == {
        "prompt_set_id": "prompt-set-id",
        "description": "d",
        "run_config": {"branch_id": "branch-1", "repeat_count": 3},
    }


def test_runs_create_body_rejects_bad_repeat_count() -> None:
    with pytest.raises(DocsRegressionError, match="repeat-count"):
        runs_create_body("prompt-set-id", repeat_count=11)


# ---------------------------------------------------------------------------
# deletion check
# ---------------------------------------------------------------------------


def test_run_deletion_check_flags_full_and_partial_removal() -> None:
    baseline_rows = load_corpus_sections(BASELINE)
    candidate_rows = load_corpus_sections(CANDIDATE)
    questions = load_questions(QUESTIONS)

    result = run_deletion_check(
        baseline_rows, candidate_rows, questions, resolved_gap_question_ids=("q004",)
    )

    assert result.removed == ("sec_branch",)
    assert result.changed == ("sec_pg",)
    assert result.added == ("sec_new",)
    assert result.fully_removed_question_ids == ("q004",)
    assert result.partially_affected_question_ids == ("q001",)


def test_run_deletion_check_no_changes_when_corpora_match() -> None:
    baseline_rows = load_corpus_sections(BASELINE)
    questions = load_questions(QUESTIONS)

    result = run_deletion_check(
        baseline_rows, baseline_rows, questions, resolved_gap_question_ids=()
    )

    assert result.removed == ()
    assert result.changed == ()
    assert result.fully_removed_question_ids == ()
    assert result.partially_affected_question_ids == ()


# ---------------------------------------------------------------------------
# live execution (FakeClient, no network)
# ---------------------------------------------------------------------------


def test_execute_live_walks_prompt_set_and_run_creation() -> None:
    chunks = build_prompt_set_chunks("slug", "Name", (Prompt("How?", "Cite it."),))
    client = FakeClient(
        [
            {"records": []},
            {"id": "prompt-set-1"},
            {"id": "run-1"},
            {"id": "run-1", "status": "COMPLETED"},
        ]
    )

    results = execute_live(
        client,
        model_id="model-1",
        chunks=chunks,
        description=None,
        branch_id=None,
        repeat_count=None,
    )

    assert results == (
        {
            "slug": "slug",
            "prompt_set_id": "prompt-set-1",
            "run_id": "run-1",
            "run": {"id": "run-1", "status": "COMPLETED"},
        },
    )
    assert client.calls[0][0] == (
        "ai-eval",
        "prompt-sets-list",
        "--model-ids",
        "model-1",
    )
    assert client.calls[1][0] == ("ai-eval", "prompt-sets-create", "--body", "-")
    assert client.calls[2][0] == ("ai-eval", "runs-create", "--body", "-")
    assert client.calls[3][0] == ("ai-eval", "runs-get", "run-1")


def test_execute_live_unwraps_the_cli_response_envelopes() -> None:
    chunks = build_prompt_set_chunks("slug", "Name", (Prompt("How?", "Cite it."),))
    client = FakeClient(
        [
            {"prompt_sets": []},
            {"prompt_set": {"id": "prompt-set-1", "slug": "slug"}},
            {"run": {"id": "run-1"}},
            {"run": {"id": "run-1", "status": "PENDING"}},
        ]
    )

    results = execute_live(
        client,
        model_id="model-1",
        chunks=chunks,
        description=None,
        branch_id=None,
        repeat_count=None,
    )

    assert results[0]["prompt_set_id"] == "prompt-set-1"
    assert results[0]["run_id"] == "run-1"
    assert client.calls[3][0] == ("ai-eval", "runs-get", "run-1")


def test_execute_live_reuses_an_existing_prompt_set() -> None:
    chunks = build_prompt_set_chunks("slug", "Name", (Prompt("How?", "Cite it."),))
    client = FakeClient([{"run": {"id": "run-2"}}, {"run": {"id": "run-2"}}])

    results = execute_live(
        client,
        model_id="model-1",
        chunks=chunks,
        description=None,
        branch_id=None,
        repeat_count=None,
        prompt_set_id="existing-set",
    )

    assert [call[0][1] for call in client.calls] == ["runs-create", "runs-get"]
    assert json.loads(client.calls[0][1])["prompt_set_id"] == "existing-set"
    assert results[0]["prompt_set_id"] == "existing-set"


def test_execute_live_rejects_prompt_set_reuse_across_chunks() -> None:
    prompts = tuple(Prompt(f"Q{i}?", "Cite it.") for i in range(26))
    chunks = build_prompt_set_chunks("slug", "Name", prompts)
    assert len(chunks) == 2

    with pytest.raises(DocsRegressionError, match="2 chunks"):
        execute_live(
            FakeClient([]),
            model_id="model-1",
            chunks=chunks,
            description=None,
            branch_id=None,
            repeat_count=None,
            prompt_set_id="existing-set",
        )


def test_execute_live_rejects_response_without_id() -> None:
    chunks = build_prompt_set_chunks("slug", "Name", (Prompt("How?", "Cite it."),))
    client = FakeClient([{"records": []}, {"no_id": True}])

    with pytest.raises(DocsRegressionError, match="prompt-sets-create"):
        execute_live(
            client,
            model_id="model-1",
            chunks=chunks,
            description=None,
            branch_id=None,
            repeat_count=None,
        )


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_plan_prints_commands_and_bodies(capsys: pytest.CaptureFixture) -> None:
    exit_code = main(
        [
            "--plan",
            "--queue",
            str(QUEUE),
            "--questions",
            str(QUESTIONS),
            "--model-id",
            "model-1",
            "--slug",
            "docs-qa-regression-v1",
            "--name",
            "Docs QA regression",
            "--answerable-sample",
            "2",
            "--profile",
            "benchmark-infra",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert (
        "omni --compact --profile benchmark-infra ai-eval prompt-sets-list --model-ids model-1"
        in out
    )
    assert (
        "omni --compact --profile benchmark-infra ai-eval prompt-sets-create --body -"
        in out
    )
    assert '"slug": "docs-qa-regression-v1"' in out
    assert "omni --compact --profile benchmark-infra ai-eval runs-get <run-id>" in out


def test_main_deletion_check_reports_and_exits_nonzero(
    capsys: pytest.CaptureFixture,
) -> None:
    exit_code = main(
        [
            "--deletion-check",
            "--queue",
            str(QUEUE),
            "--questions",
            str(QUESTIONS),
            "--baseline",
            str(BASELINE),
            "--candidate",
            str(CANDIDATE),
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["fully_removed_question_ids"] == ["q004"]
    assert payload["partially_affected_question_ids"] == ["q001"]


def test_main_requires_a_mode(capsys: pytest.CaptureFixture) -> None:
    exit_code = main(["--queue", str(QUEUE), "--questions", str(QUESTIONS)])

    assert exit_code == 1
    error = json.loads(capsys.readouterr().err)
    assert "error" in error

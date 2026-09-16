import json
from pathlib import Path

import pytest

from omni_docs_qa.verdict import (
    DocsVerdictError,
    PROMPT_TEMPLATE,
    ResultRow,
    build_prompt,
    build_request_body,
    load_questions,
    main,
    parse_verdict,
    read_results,
    score,
    submit_all,
    write_results,
)

FIXTURES = Path(__file__).parent / "fixtures" / "docs_qa"


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def run(self, *arguments, stdin=None):
        self.calls.append((arguments, stdin))
        return self.responses.pop(0)


# --- prompt / request body -------------------------------------------------


def test_build_prompt_inserts_question_verbatim_and_keeps_template_wording() -> None:
    question = "How long can an Omni embed session last before it expires?"

    prompt = build_prompt(question)

    assert "<question>" not in prompt
    assert f"Question:\n{question}\n" in prompt
    assert "Do not use outside knowledge." in prompt
    assert 'use verdict "not_covered" with empty' in prompt
    assert "Never judge from truncated text." in prompt
    assert PROMPT_TEMPLATE.count("<question>") == 1


def test_prompt_asks_for_two_step_retrieval_before_declaring_a_gap() -> None:
    prompt = build_prompt("q")

    assert "query only section_id, page_url, page_title, and heading_path" in prompt
    assert "query body filtered" in prompt
    assert 'Before answering "not_covered", run at least one step-one listing' in prompt


def test_build_request_body_minimal() -> None:
    body = build_request_body(
        model_id="model-1", prompt="prompt text", topic_name="docs_sections"
    )

    assert body == {
        "modelId": "model-1",
        "prompt": "prompt text",
        "topicName": "docs_sections",
    }


def test_build_request_body_includes_only_set_optional_fields() -> None:
    body = build_request_body(
        model_id="model-1",
        prompt="prompt text",
        topic_name="docs_sections",
        branch_id="branch-1",
        conversation_id="conv-1",
    )

    assert body == {
        "modelId": "model-1",
        "prompt": "prompt text",
        "topicName": "docs_sections",
        "branchId": "branch-1",
        "conversationId": "conv-1",
    }
    assert "webhookUrl" not in body


def test_build_request_body_webhook_requires_signing_secret() -> None:
    with pytest.raises(DocsVerdictError, match="webhookSigningSecret"):
        build_request_body(
            model_id="model-1",
            prompt="prompt text",
            topic_name="docs_sections",
            webhook_url="https://example.com/hook",
        )


def test_build_request_body_webhook_with_secret() -> None:
    body = build_request_body(
        model_id="model-1",
        prompt="prompt text",
        topic_name="docs_sections",
        webhook_url="https://example.com/hook",
        webhook_signing_secret="secret",
    )

    assert body["webhookUrl"] == "https://example.com/hook"
    assert body["webhookSigningSecret"] == "secret"


# --- verdict parsing (structural validation only) --------------------------


def _valid_answerable() -> dict:
    return {
        "verdict": "answerable",
        "answer": "Up to 576 hours.",
        "cited_section_ids": ["sec_1"],
        "cited_page_urls": ["https://docs.omni.co/embed/limitations"],
        "reason": "The Session length section states the expiry window.",
    }


def test_parse_verdict_accepts_valid_answerable() -> None:
    verdict, parse_error = parse_verdict(_valid_answerable())

    assert parse_error is None
    assert verdict == _valid_answerable()


def test_parse_verdict_returns_a_copy_not_the_caller_dict() -> None:
    raw = _valid_answerable()

    verdict, parse_error = parse_verdict(raw)

    assert parse_error is None
    assert verdict is not raw
    verdict["answer"] = "mutated"
    assert raw["answer"] != "mutated"


def test_parse_verdict_accepts_valid_not_covered() -> None:
    raw = {
        "verdict": "not_covered",
        "answer": "",
        "cited_section_ids": [],
        "cited_page_urls": [],
        "reason": "No mentions found.",
    }

    verdict, parse_error = parse_verdict(raw)

    assert parse_error is None
    assert verdict == raw


def test_parse_verdict_rejects_non_dict() -> None:
    verdict, parse_error = parse_verdict("not json")

    assert verdict is None
    assert "must be a JSON object" in parse_error


def test_parse_verdict_rejects_missing_key() -> None:
    raw = _valid_answerable()
    del raw["cited_page_urls"]

    verdict, parse_error = parse_verdict(raw)

    assert verdict is None
    assert "missing required keys" in parse_error
    assert "cited_page_urls" in parse_error


def test_parse_verdict_rejects_unknown_key() -> None:
    raw = _valid_answerable()
    raw["missing_information"] = "extra"

    verdict, parse_error = parse_verdict(raw)

    assert verdict is None
    assert "unknown keys" in parse_error


def test_parse_verdict_rejects_unknown_verdict_value() -> None:
    raw = _valid_answerable()
    raw["verdict"] = "maybe"

    verdict, parse_error = parse_verdict(raw)

    assert verdict is None
    assert "unknown verdict value" in parse_error


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cited_section_ids", "sec_1", "must be a list"),
        ("cited_page_urls", [1], "must be a list"),
        ("cited_section_ids", ["sec_1", "sec_1"], "duplicates"),
    ],
)
def test_parse_verdict_rejects_malformed_cited_lists(
    field: str, value: object, message: str
) -> None:
    raw = _valid_answerable()
    raw[field] = value

    verdict, parse_error = parse_verdict(raw)

    assert verdict is None
    assert message in parse_error


def test_parse_verdict_rejects_not_covered_with_citations() -> None:
    raw = {
        "verdict": "not_covered",
        "answer": "",
        "cited_section_ids": ["sec_1"],
        "cited_page_urls": [],
        "reason": "reason text",
    }

    verdict, parse_error = parse_verdict(raw)

    assert verdict is None
    assert "not_covered verdict must have empty cited lists" in parse_error


def test_parse_verdict_rejects_answerable_without_citations() -> None:
    raw = _valid_answerable()
    raw["cited_section_ids"] = []

    verdict, parse_error = parse_verdict(raw)

    assert verdict is None
    assert "requires non-empty cited_section_ids" in parse_error


def test_parse_verdict_rejects_empty_reason() -> None:
    raw = _valid_answerable()
    raw["reason"] = ""

    verdict, parse_error = parse_verdict(raw)

    assert verdict is None
    assert "reason must be a non-empty string" in parse_error


# --- questions loading -------------------------------------------------


def test_load_questions_returns_expected_fields() -> None:
    questions = load_questions(FIXTURES / "verdict_questions.json")

    assert [q["question_id"] for q in questions] == [
        "q001",
        "q002",
        "q003",
        "q004",
        "q005",
    ]
    assert questions[0]["label"] == "answerable"
    assert questions[0]["expected_cited_page_urls"] == [
        "https://docs.omni.co/embed/limitations"
    ]


def test_load_questions_rejects_invalid_label(tmp_path: Path) -> None:
    bad = tmp_path / "questions.json"
    bad.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "question_id": "q001",
                        "question": "text?",
                        "label": "unknown",
                        "expected_cited_page_urls": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DocsVerdictError, match="invalid label"):
        load_questions(bad)


# --- results read/write --------------------------------------------------


def test_read_results_fixture_has_expected_parse_error_row() -> None:
    rows = read_results(FIXTURES / "verdict_results.jsonl")

    assert len(rows) == 5
    by_id = {row.question_id: row for row in rows}
    assert by_id["q004"].parse_error is not None
    assert by_id["q004"].verdict is None
    assert by_id["q001"].verdict["verdict"] == "answerable"
    assert by_id["q001"].parse_error is None


def test_write_results_then_read_results_roundtrip(tmp_path: Path) -> None:
    rows = [
        ResultRow(
            question_id="q001",
            label="answerable",
            job_id=None,
            request={"modelId": "m", "prompt": "p", "topicName": "docs_sections"},
            raw_response=_valid_answerable(),
            verdict=_valid_answerable(),
            parse_error=None,
        ),
        ResultRow(
            question_id="q002",
            label="partial",
            job_id="job-1",
            request={"modelId": "m", "prompt": "p2", "topicName": "docs_sections"},
            raw_response="garbage",
            verdict=None,
            parse_error="verdict response must be a JSON object",
        ),
    ]
    out = tmp_path / "results.jsonl"

    write_results(rows, out)
    round_tripped = read_results(out)

    assert round_tripped == rows
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        assert json.loads(line)
        assert line == json.dumps(
            json.loads(line), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )


def test_read_results_rejects_row_with_both_verdict_and_parse_error(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "results.jsonl"
    bad.write_text(
        json.dumps(
            {
                "question_id": "q001",
                "label": "answerable",
                "job_id": None,
                "request": {},
                "raw_response": {},
                "verdict": _valid_answerable(),
                "parse_error": "oops",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DocsVerdictError, match="exactly one of verdict/parse_error"):
        read_results(bad)


# --- scoring ---------------------------------------------------------------


def test_score_matches_fixture_confusion_matrix_and_rates() -> None:
    rows = read_results(FIXTURES / "verdict_results.jsonl")
    questions = load_questions(FIXTURES / "verdict_questions.json")
    expected_urls = {q["question_id"]: q["expected_cited_page_urls"] for q in questions}

    result = score(rows, expected_urls)

    assert result["scored"] == 4
    assert result["correct"] == 2
    assert result["accuracy"] == 0.5
    assert result["confusion_matrix"]["answerable"]["answerable"] == 1
    assert result["confusion_matrix"]["partial"]["answerable"] == 1
    assert result["confusion_matrix"]["not_covered"]["not_covered"] == 1
    assert result["confusion_matrix"]["answerable"]["not_covered"] == 1
    assert result["confusion_matrix"]["conflicting"]["conflicting"] == 0
    assert result["unparsed"] == {
        "answerable": 0,
        "partial": 0,
        "not_covered": 0,
        "conflicting": 1,
    }
    assert result["false_gap_rate"] == 0.5
    assert result["false_answer_rate"] == 0.5
    assert result["cited_answer_correct_rate"] == 1.0
    assert set(result["confusion_matrix"].keys()) == {
        "answerable",
        "partial",
        "not_covered",
        "conflicting",
    }
    for label_matrix in result["confusion_matrix"].values():
        assert set(label_matrix.keys()) == {
            "answerable",
            "partial",
            "not_covered",
            "conflicting",
        }


def test_score_zero_denominators_yield_zero() -> None:
    result = score([], {})

    assert result["scored"] == 0
    assert result["accuracy"] == 0.0
    assert result["false_gap_rate"] == 0.0
    assert result["false_answer_rate"] == 0.0
    assert result["cited_answer_correct_rate"] == 0.0
    for label in ("answerable", "partial", "not_covered", "conflicting"):
        assert result["recall"][label] == 0.0
        assert result["precision"][label] == 0.0


# --- main(): --plan ---------------------------------------------------------


def test_main_plan_prints_request_bodies_without_touching_a_client(capsys) -> None:
    exit_code = main(
        [
            "--questions",
            str(FIXTURES / "verdict_questions.json"),
            "--model-id",
            "model-1",
            "--plan",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert len(output["plans"]) == 5
    first = output["plans"][0]
    assert first["question_id"] == "q001"
    assert first["command"][-3:] == ["ai", "job-submit", "--body"] or first["command"][
        -4:
    ] == ["ai", "job-submit", "--body", "-"]
    assert first["body"]["modelId"] == "model-1"
    assert first["body"]["topicName"] == "docs_sections"
    assert "<question>" not in first["body"]["prompt"]


def test_main_plan_requires_model_id(capsys) -> None:
    exit_code = main(
        [
            "--questions",
            str(FIXTURES / "verdict_questions.json"),
            "--plan",
        ]
    )
    assert exit_code == 1
    assert "--model-id" in capsys.readouterr().err


def test_main_plan_redacts_webhook_signing_secret(capsys) -> None:
    exit_code = main(
        [
            "--questions",
            str(FIXTURES / "verdict_questions.json"),
            "--model-id",
            "model-1",
            "--webhook-url",
            "https://example.com/hook",
            "--webhook-signing-secret",
            "super-secret",
            "--plan",
        ]
    )
    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    for plan in output["plans"]:
        assert plan["body"]["webhookSigningSecret"] == "<redacted>"


# --- main(): --replay --------------------------------------------------


def test_main_replay_scores_fixture_responses_and_writes_results(
    tmp_path: Path, capsys
) -> None:
    out = tmp_path / "results.jsonl"

    exit_code = main(
        [
            "--questions",
            str(FIXTURES / "verdict_questions.json"),
            "--model-id",
            "model-1",
            "--replay",
            "--responses",
            str(FIXTURES / "verdict_responses.jsonl"),
            "--results",
            str(out),
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["scored"] == 4
    assert output["correct"] == 2
    written_rows = read_results(out)
    assert len(written_rows) == 5
    assert {row.question_id for row in written_rows} == {
        "q001",
        "q002",
        "q003",
        "q004",
        "q005",
    }


def test_main_replay_requires_responses(capsys) -> None:
    exit_code = main(
        [
            "--questions",
            str(FIXTURES / "verdict_questions.json"),
            "--model-id",
            "model-1",
            "--replay",
        ]
    )
    assert exit_code == 1
    assert "--responses" in capsys.readouterr().err


# --- main(): --score ---------------------------------------------------


def test_main_score_reads_existing_results_file(capsys) -> None:
    exit_code = main(
        [
            "--questions",
            str(FIXTURES / "verdict_questions.json"),
            "--score",
            "--results",
            str(FIXTURES / "verdict_results.jsonl"),
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["scored"] == 4
    assert output["correct"] == 2


def test_main_score_requires_results(capsys) -> None:
    exit_code = main(
        [
            "--questions",
            str(FIXTURES / "verdict_questions.json"),
            "--score",
        ]
    )
    assert exit_code == 1
    assert "--results" in capsys.readouterr().err


# --- main(): action flag discipline / --live never executes ---------------


def test_main_requires_exactly_one_action_flag(capsys) -> None:
    exit_code = main(["--questions", str(FIXTURES / "verdict_questions.json")])
    assert exit_code == 1
    assert "exactly one" in capsys.readouterr().err


def test_main_live_refuses_without_model_id(capsys) -> None:
    exit_code = main(
        [
            "--questions",
            str(FIXTURES / "verdict_questions.json"),
            "--live",
            "--profile",
            "test-profile",
        ]
    )
    assert exit_code == 1
    assert "--model-id" in capsys.readouterr().err


def test_main_live_is_never_invoked_with_a_working_client(monkeypatch, capsys) -> None:
    """Guard against accidental real network execution in this suite.

    OmniClient must only be constructed on the --live path. Every --live
    test in this module replaces OmniClient with a FakeClient, so the real
    omni binary is never invoked here.
    """
    from omni_docs_qa import verdict

    def _boom(*args, **kwargs):
        raise AssertionError("OmniClient must never be constructed in tests")

    monkeypatch.setattr(verdict, "OmniClient", _boom)

    exit_code = main(
        [
            "--questions",
            str(FIXTURES / "verdict_questions.json"),
            "--live",
            "--profile",
            "test-profile",
        ]
    )
    assert exit_code == 1
    assert "--model-id" in capsys.readouterr().err


# --- submit_all(): polling to a terminal job state -------------------------


def _question(question_id="q001", label="answerable"):
    return {
        "question_id": question_id,
        "question": "How long can an Omni embed session last?",
        "label": label,
        "expected_cited_page_urls": ["https://docs.omni.co/embed/limitations"],
    }


def _job_result(message: str) -> dict:
    """Shape of `omni ai job-result` as observed live on 2026-09-15."""
    return {
        "actions": [],
        "message": message,
        "omniChatUrl": "https://example.omniapp.co/chat/conv-1",
        "resultSummary": message,
        "topic": "docs_sections",
    }


def test_submit_all_polls_job_status_until_complete_then_fetches_result() -> None:
    client = FakeClient(
        [
            {"jobId": "job-1"},
            {"state": "PENDING"},
            {"state": "COMPLETE"},
            _job_result(json.dumps(_valid_answerable())),
        ]
    )

    rows = submit_all(
        client,
        [_question()],
        model_id="model-1",
        topic_name="docs_sections",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.job_id == "job-1"
    assert row.parse_error is None
    assert row.verdict == _valid_answerable()
    assert client.calls[0][0] == ("ai", "job-submit", "--body", "-")
    assert client.calls[1][0] == ("ai", "job-status", "job-1")
    assert client.calls[2][0] == ("ai", "job-status", "job-1")
    assert client.calls[3][0] == ("ai", "job-result", "job-1")


def test_submit_all_raises_on_failed_job() -> None:
    client = FakeClient([{"jobId": "job-1"}, {"state": "FAILED"}])

    with pytest.raises(DocsVerdictError, match="job-1.*FAILED"):
        submit_all(
            client,
            [_question()],
            model_id="model-1",
            topic_name="docs_sections",
        )


def test_submit_all_raises_when_job_submit_omits_job_id() -> None:
    client = FakeClient([{}])

    with pytest.raises(DocsVerdictError, match="jobId"):
        submit_all(
            client,
            [_question()],
            model_id="model-1",
            topic_name="docs_sections",
        )


def test_submit_all_raises_when_polling_never_reaches_terminal_state(
    monkeypatch,
) -> None:
    from omni_docs_qa import verdict

    monkeypatch.setattr(verdict, "MAX_POLL_ATTEMPTS", 2)
    monkeypatch.setattr(verdict.time, "sleep", lambda _seconds: None)
    client = FakeClient(
        [
            {"jobId": "job-1"},
            {"state": "PENDING"},
            {"state": "PENDING"},
        ]
    )

    with pytest.raises(DocsVerdictError, match="did not reach a terminal state"):
        submit_all(
            client,
            [_question()],
            model_id="model-1",
            topic_name="docs_sections",
        )


def test_submit_all_keeps_non_json_message_as_parse_error() -> None:
    client = FakeClient(
        [
            {"jobId": "job-1"},
            {"state": "COMPLETE"},
            _job_result("Sorry, I could not find that in the docs."),
        ]
    )

    rows = submit_all(
        client,
        [_question()],
        model_id="model-1",
        topic_name="docs_sections",
    )

    assert rows[0].verdict is None
    assert rows[0].raw_response == "Sorry, I could not find that in the docs."
    assert rows[0].parse_error.startswith("job-result message is not JSON")


def test_submit_all_raises_when_job_result_has_no_message() -> None:
    client = FakeClient(
        [{"jobId": "job-1"}, {"state": "COMPLETE"}, {"resultSummary": "x"}]
    )

    with pytest.raises(DocsVerdictError, match="no message string"):
        submit_all(
            client,
            [_question()],
            model_id="model-1",
            topic_name="docs_sections",
        )


def test_submit_all_calls_on_row_before_a_later_job_fails() -> None:
    client = FakeClient(
        [
            {"jobId": "job-1"},
            {"state": "COMPLETE"},
            _job_result(json.dumps(_valid_answerable())),
            {"jobId": "job-2"},
            {"state": "FAILED"},
        ]
    )
    seen = []

    with pytest.raises(DocsVerdictError, match="job-2.*FAILED"):
        submit_all(
            client,
            [_question(), _question("q002", "not_covered")],
            model_id="model-1",
            topic_name="docs_sections",
            on_row=seen.append,
        )

    assert [row.job_id for row in seen] == ["job-1"]
    assert seen[0].verdict == _valid_answerable()


def test_live_run_writes_each_row_as_it_completes(tmp_path, monkeypatch) -> None:
    from omni_docs_qa import verdict

    responses = [
        {"jobId": "job-1"},
        {"state": "COMPLETE"},
        _job_result(json.dumps(_valid_answerable())),
        {"jobId": "job-2"},
        {"state": "FAILED"},
    ]
    monkeypatch.setattr(verdict, "OmniClient", lambda profile: FakeClient(responses))
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps({"questions": [_question(), _question("q002", "not_covered")]}),
        encoding="utf-8",
    )
    results_path = tmp_path / "results.jsonl"

    exit_code = main(
        [
            "--questions",
            str(questions_path),
            "--model-id",
            "model-1",
            "--results",
            str(results_path),
            "--live",
            "--profile",
            "test-profile",
        ]
    )

    assert exit_code == 1
    rows = read_results(results_path)
    assert [row.job_id for row in rows] == ["job-1"]


def test_live_run_refuses_to_overwrite_existing_results(tmp_path, monkeypatch) -> None:
    from omni_docs_qa import verdict

    monkeypatch.setattr(verdict, "OmniClient", lambda profile: FakeClient([]))
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps({"questions": [_question()]}), encoding="utf-8"
    )
    results_path = tmp_path / "results.jsonl"
    results_path.write_text("", encoding="utf-8")

    exit_code = main(
        [
            "--questions",
            str(questions_path),
            "--model-id",
            "model-1",
            "--results",
            str(results_path),
            "--live",
            "--profile",
            "test-profile",
        ]
    )

    assert exit_code == 1


# --- CLI entry point smoke test --------------------------------------------


def test_module_runnable_as_main(capsys) -> None:
    exit_code = main(
        [
            "--questions",
            str(FIXTURES / "verdict_questions.json"),
            "--model-id",
            "model-1",
            "--plan",
        ]
    )
    assert exit_code == 0
    capsys.readouterr()

import json
from pathlib import Path

import pytest

from omni_docs_qa.questions import (
    DocsQuestionsError,
    load_injected_ids,
    load_questions,
    main,
    validate,
)

FIXTURES = Path(__file__).parent / "fixtures" / "docs_qa"
SAMPLE_QUESTIONS = FIXTURES / "questions_sample.json"
SAMPLE_INJECTED = FIXTURES / "questions_injected.json"
SAMPLE_CORPUS = FIXTURES / "questions_corpus.jsonl"


def _load_sample() -> dict:
    return json.loads(SAMPLE_QUESTIONS.read_text(encoding="utf-8"))


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_questions_reads_schema() -> None:
    doc = load_questions(SAMPLE_QUESTIONS)

    assert doc.slug == "docs-qa-coverage-v1"
    assert [q.question_id for q in doc.questions] == ["q001", "q002", "q003", "q004"]


@pytest.mark.parametrize("field", ["question_id", "question", "label", "rationale"])
def test_load_questions_rejects_non_string_field(tmp_path: Path, field: str) -> None:
    payload = _load_sample()
    payload["questions"][0][field] = 123
    path = _write(tmp_path, "questions.json", payload)

    with pytest.raises(DocsQuestionsError, match=field):
        load_questions(path)


@pytest.mark.parametrize("field", ["question_id", "question", "label", "rationale"])
def test_load_questions_rejects_empty_string_field(tmp_path: Path, field: str) -> None:
    payload = _load_sample()
    payload["questions"][0][field] = ""
    path = _write(tmp_path, "questions.json", payload)

    with pytest.raises(DocsQuestionsError, match=field):
        load_questions(path)


def test_validate_sample_passes_without_strict_counts() -> None:
    doc = load_questions(SAMPLE_QUESTIONS)

    result = validate(doc, strict_counts=False)

    assert result.by_label == {
        "answerable": 1,
        "partial": 1,
        "not_covered": 1,
        "conflicting": 1,
    }
    assert result.questions == 4
    assert result.checks


def test_validate_sample_fails_strict_counts_by_default() -> None:
    doc = load_questions(SAMPLE_QUESTIONS)

    with pytest.raises(DocsQuestionsError, match="55"):
        validate(doc, strict_counts=True)


def test_validate_with_injected_resolves_ids() -> None:
    doc = load_questions(SAMPLE_QUESTIONS)
    injected_ids = load_injected_ids(SAMPLE_INJECTED)

    result = validate(doc, strict_counts=False, injected_ids=injected_ids)

    assert "injected_ids_exist" in result.checks


def test_validate_with_injected_missing_id_fails(tmp_path: Path) -> None:
    payload = _load_sample()
    payload["questions"][3]["injected_section_ids"] = ["inj_does_not_exist"]
    path = _write(tmp_path, "questions.json", payload)
    doc = load_questions(path)
    injected_ids = load_injected_ids(SAMPLE_INJECTED)

    with pytest.raises(DocsQuestionsError, match="inj_does_not_exist"):
        validate(doc, strict_counts=False, injected_ids=injected_ids)


def test_validate_with_corpus_resolves_page_urls_and_injected_ids() -> None:
    result = main(
        [
            "--questions",
            str(SAMPLE_QUESTIONS),
            "--no-strict-counts",
            "--corpus",
            str(SAMPLE_CORPUS),
        ]
    )

    assert result == 0


def test_corpus_check_fails_on_unresolved_page_url(tmp_path: Path) -> None:
    payload = _load_sample()
    payload["questions"][0]["expected_cited_page_urls"] = [
        "https://docs.omni.co/nonexistent-page"
    ]
    path = _write(tmp_path, "questions.json", payload)

    exit_code = main(
        [
            "--questions",
            str(path),
            "--no-strict-counts",
            "--corpus",
            str(SAMPLE_CORPUS),
        ]
    )

    assert exit_code == 1


def test_corpus_check_fails_on_unresolved_injected_id(tmp_path: Path) -> None:
    payload = _load_sample()
    # q001 is answerable and cites the docs row already in the corpus fixture,
    # but claim it also has an injected id that only exists as a docs row's
    # section_id, never as source_kind == "injected".
    payload["questions"][0]["label"] = "conflicting"
    payload["questions"][0]["injected_section_ids"] = ["sec_aaaaaaaaaaaaaaaa"]
    path = _write(tmp_path, "questions.json", payload)

    exit_code = main(
        [
            "--questions",
            str(path),
            "--no-strict-counts",
            "--corpus",
            str(SAMPLE_CORPUS),
        ]
    )

    assert exit_code == 1


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p["questions"].__setitem__(1, p["questions"][0]), "question_id"),
        (
            lambda p: p["questions"][0].__setitem__("question_id", "Q001"),
            "question_id",
        ),
        (
            lambda p: p["questions"][0].__setitem__("question", "No question mark"),
            "question",
        ),
        (
            lambda p: p["questions"][0].__setitem__("label", "unsure"),
            "label",
        ),
        (
            lambda p: p["questions"][0].__setitem__("rationale", "too short"),
            "rationale",
        ),
        (
            lambda p: p["questions"][0].__setitem__(
                "expected_cited_page_urls", ["ftp://example.com/page"]
            ),
            "expected_cited_page_urls",
        ),
        (
            lambda p: p["questions"][2].__setitem__(
                "expected_cited_page_urls", ["https://docs.omni.co/foo"]
            ),
            "not_covered",
        ),
        (
            lambda p: p["questions"][0].__setitem__(
                "injected_section_ids", ["inj_should_not_be_here"]
            ),
            "injected_section_ids",
        ),
        (
            lambda p: p["questions"][3].__setitem__("injected_section_ids", []),
            "conflicting",
        ),
        (
            lambda p: p["questions"][0].pop("rationale"),
            "rationale",
        ),
    ],
)
def test_validate_rejects_malformed_questions(tmp_path, mutate, match) -> None:
    payload = _load_sample()
    mutate(payload)
    path = _write(tmp_path, "questions.json", payload)

    with pytest.raises(DocsQuestionsError, match=match):
        validate(load_questions(path), strict_counts=False)


def test_main_prints_json_result(capsys, tmp_path: Path) -> None:
    exit_code = main(["--questions", str(SAMPLE_QUESTIONS), "--no-strict-counts"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["questions"] == 4
    assert output["corpus_checked"] is False
    assert output["by_label"]["answerable"] == 1


def test_main_returns_1_on_validation_error(capsys) -> None:
    exit_code = main(["--questions", str(SAMPLE_QUESTIONS)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "55" in captured.err or "55" in captured.out


def test_schema_version_must_be_1(tmp_path: Path) -> None:
    payload = _load_sample()
    payload["schema_version"] = 2
    path = _write(tmp_path, "questions.json", payload)

    with pytest.raises(DocsQuestionsError, match="schema_version"):
        load_questions(path)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    payload = _load_sample()
    payload["questions"][0]["extra_key"] = "nope"
    path = _write(tmp_path, "questions.json", payload)

    with pytest.raises(DocsQuestionsError, match="extra_key|unexpected"):
        load_questions(path)


def test_never_matches_question_text_against_body(tmp_path: Path) -> None:
    # Regression guard: a question whose text has nothing to do with the
    # cited page's body must still pass, because the validator performs no
    # semantic/text matching between question and corpus body.
    payload = _load_sample()
    payload["questions"][0]["question"] = (
        "Completely unrelated wording that never appears in any corpus body?"
    )
    path = _write(tmp_path, "questions.json", payload)

    exit_code = main(
        [
            "--questions",
            str(path),
            "--no-strict-counts",
            "--corpus",
            str(SAMPLE_CORPUS),
        ]
    )

    assert exit_code == 0

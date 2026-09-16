"""Tests for the docs QA gap queue builder and snapshot-edit applier.

See the docs-qa shared contract (product bet 4), section 7, for the gap row
shape, proposed-edit schema, and apply rule this module implements.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omni_docs_qa.gaps import (
    DocsGapsError,
    apply_edits,
    build_gap_queue,
    load_edit_file,
    load_question_texts,
    load_results,
    main,
    next_snapshot_version,
    read_jsonl,
)

FIXTURES = Path(__file__).parent / "fixtures" / "docs_qa"
CORPUS = FIXTURES / "gaps_corpus.jsonl"
RESULTS = FIXTURES / "gaps_results.jsonl"
EDIT_Q014 = FIXTURES / "gaps_edit_q014.json"

QUESTION_TEXT_BY_ID = {
    "q001": "How does Omni handle page A setup?",
    "q002": "What are all the configuration options for page A?",
    "q003": "How does Omni handle page C rollback?",
    "q004": "Does page A or page B take precedence during setup?",
    "q005": "Is page B setup documented?",
}


def _write_questions_file(tmp_path: Path) -> Path:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {"question_id": qid, "question": text}
                    for qid, text in QUESTION_TEXT_BY_ID.items()
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_results_parses_verdict_and_parse_error_rows() -> None:
    results = load_results(RESULTS)

    assert [row.question_id for row in results] == [
        "q001",
        "q002",
        "q003",
        "q004",
        "q005",
    ]
    q005 = results[-1]
    assert q005.verdict is None
    assert q005.parse_error == "response was not a single JSON object"


def test_load_results_rejects_cited_section_ids_as_a_string(tmp_path: Path) -> None:
    bad = tmp_path / "results.jsonl"
    bad.write_text(
        json.dumps(
            {
                "question_id": "q001",
                "label": "partial",
                "job_id": None,
                "request": {"prompt": "Question:\nX?\n\nReply"},
                "raw_response": "{}",
                "verdict": {
                    "verdict": "partial",
                    "answer": "a",
                    "cited_section_ids": "sec_abc",
                    "cited_page_urls": ["https://docs.omni.co/a"],
                    "reason": "r",
                },
                "parse_error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DocsGapsError, match="cited_section_ids"):
        load_results(bad)


def test_load_results_rejects_not_covered_verdict_with_citations(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "results.jsonl"
    bad.write_text(
        json.dumps(
            {
                "question_id": "q001",
                "label": "not_covered",
                "job_id": None,
                "request": {"prompt": "Question:\nX?\n\nReply"},
                "raw_response": "{}",
                "verdict": {
                    "verdict": "not_covered",
                    "answer": "a",
                    "cited_section_ids": ["sec_abc"],
                    "cited_page_urls": [],
                    "reason": "r",
                },
                "parse_error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DocsGapsError, match="not_covered"):
        load_results(bad)


def test_load_results_rejects_row_with_both_verdict_and_parse_error(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "results.jsonl"
    bad.write_text(
        json.dumps(
            {
                "question_id": "q001",
                "label": "answerable",
                "job_id": None,
                "request": {"prompt": "Question:\nX?\n\nReply"},
                "raw_response": "{}",
                "verdict": {
                    "verdict": "answerable",
                    "answer": "a",
                    "cited_section_ids": ["sec_1"],
                    "cited_page_urls": ["https://docs.omni.co/a"],
                    "reason": "r",
                },
                "parse_error": "oops",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DocsGapsError, match="exactly one"):
        load_results(bad)


def test_load_question_texts_reads_id_to_text_mapping(tmp_path: Path) -> None:
    questions_path = _write_questions_file(tmp_path)

    texts = load_question_texts(questions_path)

    assert texts == QUESTION_TEXT_BY_ID


def test_load_question_texts_rejects_missing_text(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps({"questions": [{"question_id": "q001"}]}), encoding="utf-8"
    )

    with pytest.raises(DocsGapsError, match="missing"):
        load_question_texts(path)


def test_build_gap_queue_selects_gap_labels_and_uses_question_text(
    tmp_path: Path,
) -> None:
    results = load_results(RESULTS)
    questions_path = _write_questions_file(tmp_path)
    question_text_by_id = load_question_texts(questions_path)

    queue = build_gap_queue(
        results,
        source_results="results/gaps_results.jsonl",
        question_text_by_id=question_text_by_id,
    )

    assert [gap.question_id for gap in queue] == ["q002", "q003", "q004"]
    q002 = queue[0]
    assert q002.gap_id == "gap_q002"
    assert q002.expected_label == "partial"
    assert q002.predicted_verdict == "partial"
    assert q002.question == "What are all the configuration options for page A?"
    assert q002.reason == "Only partial coverage found."
    assert q002.cited_section_ids == ("sec_a1",)
    assert q002.cited_page_urls == ("https://docs.omni.co/a",)
    assert q002.source_results == "results/gaps_results.jsonl"
    assert q002.status == "open"
    assert q002.resolved_in_snapshot is None

    q003 = queue[1]
    assert q003.cited_section_ids == ()
    assert q003.cited_page_urls == ()

    q004 = queue[2]
    assert q004.cited_section_ids == ("sec_a1", "sec_b1")


def test_build_gap_queue_excludes_answerable_and_parse_error_rows(
    tmp_path: Path,
) -> None:
    results = load_results(RESULTS)
    question_text_by_id = load_question_texts(_write_questions_file(tmp_path))
    queue = build_gap_queue(
        results,
        source_results="results/gaps_results.jsonl",
        question_text_by_id=question_text_by_id,
    )
    question_ids = {gap.question_id for gap in queue}
    assert "q001" not in question_ids
    assert "q005" not in question_ids


def test_build_gap_queue_rejects_result_with_no_matching_question(
    tmp_path: Path,
) -> None:
    results = load_results(RESULTS)

    with pytest.raises(DocsGapsError, match="q002"):
        build_gap_queue(results, source_results="results.jsonl", question_text_by_id={})


def test_main_writes_queue_jsonl(tmp_path: Path) -> None:
    out_path = tmp_path / "queue.jsonl"
    questions_path = _write_questions_file(tmp_path)

    exit_code = main(
        [
            "--results",
            str(RESULTS),
            "--questions",
            str(questions_path),
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    rows = read_jsonl(out_path)
    assert [row["question_id"] for row in rows] == ["q002", "q003", "q004"]
    assert rows[0]["gap_id"] == "gap_q002"
    assert rows[0]["status"] == "open"
    # sort_keys, compact separators formatting
    with out_path.open(encoding="utf-8") as handle:
        first_line = handle.readline()
    assert json.loads(first_line) == rows[0]
    assert ", " not in first_line
    assert first_line.endswith("\n")


def test_load_edit_file_parses_operations() -> None:
    edit = load_edit_file(EDIT_Q014)

    assert edit.gap_id == "gap_q014"
    assert edit.question_id == "q014"
    assert [op["op"] for op in edit.operations] == [
        "replace_body",
        "append_section",
        "delete_section",
    ]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda d: d.pop("author"), "missing"),
        (lambda d: d.update(extra="x"), "unexpected"),
        (lambda d: d.update(schema_version=2), "schema_version"),
    ],
)
def test_load_edit_file_rejects_malformed_document(
    tmp_path: Path, mutation, match: str
) -> None:
    data = json.loads(EDIT_Q014.read_text(encoding="utf-8"))
    mutation(data)
    bad = tmp_path / "bad_edit.json"
    bad.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(DocsGapsError, match=match):
        load_edit_file(bad)


@pytest.mark.parametrize(
    ("bad_op", "match"),
    [
        ({"op": "replace_body", "section_id": "sec_a1"}, "missing keys"),
        (
            {"op": "replace_body", "section_id": "sec_a1", "body": "x", "extra": 1},
            "unexpected keys",
        ),
        ({"op": "unknown_op", "section_id": "sec_a1"}, "unknown op"),
    ],
)
def test_load_edit_file_rejects_malformed_operation(
    tmp_path: Path, bad_op: dict, match: str
) -> None:
    data = json.loads(EDIT_Q014.read_text(encoding="utf-8"))
    data["operations"] = [bad_op]
    bad = tmp_path / "bad_edit.json"
    bad.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(DocsGapsError, match=match):
        load_edit_file(bad)


def test_apply_edits_replaces_appends_deletes_and_recomputes_hashes_and_order() -> None:
    corpus = read_jsonl(CORPUS)
    edit = load_edit_file(EDIT_Q014)

    new_rows = apply_edits(corpus, [edit])

    by_id = {row["section_id"]: row for row in new_rows}
    assert "sec_a2" not in by_id
    assert by_id["sec_a1"]["body"] == "Updated intro body text for page A."
    import hashlib

    assert (
        by_id["sec_a1"]["content_hash"]
        == hashlib.sha256(b"Updated intro body text for page A.").hexdigest()
    )

    new_section = by_id["inj_new_note"]
    assert new_section["source_kind"] == "injected"
    assert new_section["part_index"] == 0
    assert new_section["part_count"] == 1
    assert new_section["heading_label"] == "Setup > Known conflicts"

    # section_order recomputed 0-based per page in final ordering
    page_a_orders = sorted(
        row["section_order"]
        for row in new_rows
        if row["page_url"] == "https://docs.omni.co/a"
    )
    assert page_a_orders == [0]
    page_b_rows = [
        row for row in new_rows if row["page_url"] == "https://docs.omni.co/b"
    ]
    assert [row["section_id"] for row in page_b_rows] == ["sec_b1", "inj_new_note"]
    assert [row["section_order"] for row in page_b_rows] == [0, 1]


def test_apply_edits_rejects_missing_referenced_section() -> None:
    corpus = read_jsonl(CORPUS)
    edit_doc = json.loads(EDIT_Q014.read_text(encoding="utf-8"))
    edit_doc["operations"] = [
        {"op": "replace_body", "section_id": "sec_does_not_exist", "body": "x"}
    ]

    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(edit_doc, handle)
        path = Path(handle.name)
    edit = load_edit_file(path)
    path.unlink()

    with pytest.raises(DocsGapsError, match="sec_does_not_exist"):
        apply_edits(corpus, [edit])


def test_apply_edits_rejects_append_section_id_collision() -> None:
    corpus = read_jsonl(CORPUS)
    edit_doc = json.loads(EDIT_Q014.read_text(encoding="utf-8"))
    edit_doc["operations"] = [
        {
            "op": "append_section",
            "anchor_section_id": "sec_b1",
            "placement": "after",
            "section_id": "sec_a1",
            "page_url": "https://docs.omni.co/b",
            "page_title": "Page B",
            "heading_path": ["Setup", "Dup"],
            "body": "dup",
        }
    ]
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(edit_doc, handle)
        path = Path(handle.name)
    edit = load_edit_file(path)
    path.unlink()

    with pytest.raises(DocsGapsError, match="already exists"):
        apply_edits(corpus, [edit])


def test_next_snapshot_version_starts_at_v001_and_increments(tmp_path: Path) -> None:
    assert next_snapshot_version(tmp_path) == "v001"
    (tmp_path / "v001").mkdir()
    (tmp_path / "v003").mkdir()
    assert next_snapshot_version(tmp_path) == "v004"


def _write_edits_dir(tmp_path: Path) -> Path:
    edits_dir = tmp_path / "edits"
    edits_dir.mkdir()
    (edits_dir / "gap_q014.json").write_text(
        EDIT_Q014.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return edits_dir


def test_main_apply_writes_snapshot_and_leaves_original_untouched(
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_bytes(CORPUS.read_bytes())
    original_bytes = corpus_path.read_bytes()
    snapshots_dir = tmp_path / "snapshots"
    edits_dir = _write_edits_dir(tmp_path)

    exit_code = main(
        [
            "--apply",
            "--edits",
            str(edits_dir),
            "--corpus",
            str(corpus_path),
            "--snapshots-dir",
            str(snapshots_dir),
        ]
    )

    assert exit_code == 0
    assert corpus_path.read_bytes() == original_bytes

    snapshot_corpus = snapshots_dir / "v001" / "corpus.jsonl"
    assert snapshot_corpus.exists()
    rows = read_jsonl(snapshot_corpus)
    section_ids = {row["section_id"] for row in rows}
    assert "sec_a2" not in section_ids
    assert "inj_new_note" in section_ids

    meta = json.loads((snapshots_dir / "v001" / "corpus.meta.json").read_text())
    assert meta["version"] == "v001"
    assert meta["parent_version"] is None
    assert meta["section_count"] == len(rows)
    assert meta["applied_gap_ids"] == ["gap_q014"]


def test_main_apply_inherits_parent_version_from_sibling_meta(
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_bytes(CORPUS.read_bytes())
    (tmp_path / "corpus.meta.json").write_text(
        json.dumps({"version": "v001"}), encoding="utf-8"
    )
    snapshots_dir = tmp_path / "snapshots"
    edits_dir = _write_edits_dir(tmp_path)

    exit_code = main(
        [
            "--apply",
            "--edits",
            str(edits_dir),
            "--corpus",
            str(corpus_path),
            "--snapshots-dir",
            str(snapshots_dir),
        ]
    )

    assert exit_code == 0
    meta = json.loads((snapshots_dir / "v001" / "corpus.meta.json").read_text())
    assert meta["parent_version"] == "v001"


def test_main_apply_dry_run_writes_nothing_and_prints_json_diff(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_bytes(CORPUS.read_bytes())
    original_bytes = corpus_path.read_bytes()
    snapshots_dir = tmp_path / "snapshots"
    edits_dir = _write_edits_dir(tmp_path)

    exit_code = main(
        [
            "--apply",
            "--edits",
            str(edits_dir),
            "--corpus",
            str(corpus_path),
            "--snapshots-dir",
            str(snapshots_dir),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert not snapshots_dir.exists()
    assert corpus_path.read_bytes() == original_bytes
    output = json.loads(capsys.readouterr().out)
    assert output["gap_ids"] == ["gap_q014"]
    assert "sec_a1" in output["diff"]
    assert "Updated intro body text for page A." in output["diff"]


def test_main_apply_refuses_existing_snapshot_directory(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_bytes(CORPUS.read_bytes())
    snapshots_dir = tmp_path / "snapshots"
    (snapshots_dir / "v001").mkdir(parents=True)
    edits_dir = _write_edits_dir(tmp_path)

    exit_code = main(
        [
            "--apply",
            "--edits",
            str(edits_dir),
            "--corpus",
            str(corpus_path),
            "--snapshots-dir",
            str(snapshots_dir),
            "--version",
            "v001",
        ]
    )

    assert exit_code == 1


def test_main_apply_requires_queue_out_when_queue_given(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_bytes(CORPUS.read_bytes())
    snapshots_dir = tmp_path / "snapshots"
    edits_dir = _write_edits_dir(tmp_path)
    queue_path = tmp_path / "queue.jsonl"
    queue_path.write_text(
        json.dumps({"gap_id": "gap_q014", "status": "open"}) + "\n", encoding="utf-8"
    )

    exit_code = main(
        [
            "--apply",
            "--edits",
            str(edits_dir),
            "--corpus",
            str(corpus_path),
            "--snapshots-dir",
            str(snapshots_dir),
            "--queue",
            str(queue_path),
        ]
    )

    assert exit_code == 1
    assert queue_path.read_text(encoding="utf-8") == (
        json.dumps({"gap_id": "gap_q014", "status": "open"}) + "\n"
    )


def test_main_apply_writes_resolved_queue_to_queue_out_leaving_input_untouched(
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_bytes(CORPUS.read_bytes())
    snapshots_dir = tmp_path / "snapshots"
    edits_dir = _write_edits_dir(tmp_path)
    queue_path = tmp_path / "queue.jsonl"
    original_queue_bytes = (
        json.dumps({"gap_id": "gap_q014", "status": "open"}) + "\n"
    ).encode("utf-8")
    queue_path.write_bytes(original_queue_bytes)
    queue_out_path = tmp_path / "queue.out.jsonl"

    exit_code = main(
        [
            "--apply",
            "--edits",
            str(edits_dir),
            "--corpus",
            str(corpus_path),
            "--snapshots-dir",
            str(snapshots_dir),
            "--queue",
            str(queue_path),
            "--queue-out",
            str(queue_out_path),
        ]
    )

    assert exit_code == 0
    assert queue_path.read_bytes() == original_queue_bytes
    resolved = read_jsonl(queue_out_path)
    assert resolved[0]["status"] == "resolved"
    assert resolved[0]["resolved_in_snapshot"] == "v001"


def test_main_apply_requires_edits(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_bytes(CORPUS.read_bytes())

    exit_code = main(["--apply", "--corpus", str(corpus_path)])

    assert exit_code == 1


def test_main_requires_results_or_apply() -> None:
    assert main([]) == 1

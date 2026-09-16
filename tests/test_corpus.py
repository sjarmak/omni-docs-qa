import hashlib
import json
from pathlib import Path

import pytest

from omni_docs_qa.corpus import (
    MAX_BODY_CHARS,
    NEON_DDL,
    UNIT,
    DocsCorpusError,
    build_corpus,
    load_injected_sections,
    main,
    parse_docs_sections,
    section_id,
    splice_injected,
)

FIXTURES = Path(__file__).parent / "fixtures" / "docs_qa"
SAMPLE_TEXT = (FIXTURES / "corpus_llms_sample.txt").read_text(encoding="utf-8")
INJECTED_PATH = FIXTURES / "corpus_injected.json"


def _by_heading(sections, page_url, heading_path):
    return [
        s
        for s in sections
        if s.page_url == page_url and list(s.heading_path) == heading_path
    ]


def test_parses_pages_preamble_and_nesting():
    sections = parse_docs_sections(SAMPLE_TEXT, max_body_chars=MAX_BODY_CHARS)

    page_one = [s for s in sections if s.page_url == "https://docs.omni.co/page-one"]
    assert page_one[0].heading_path == ()
    assert page_one[0].body == "Preamble text for page one."
    assert page_one[0].page_title == "Page One"
    assert page_one[0].section_order == 0

    section_a = _by_heading(sections, "https://docs.omni.co/page-one", ["Section A"])[0]
    assert section_a.body == "Body A text."

    subsection_a1 = _by_heading(
        sections, "https://docs.omni.co/page-one", ["Section A", "Subsection A1"]
    )[0]
    assert subsection_a1.body == "Body A1 text."
    assert subsection_a1.heading_label == "Section A > Subsection A1"


def test_fence_hides_headings_and_stray_hash_line_stays_body():
    sections = parse_docs_sections(SAMPLE_TEXT, max_body_chars=MAX_BODY_CHARS)
    section_b = _by_heading(sections, "https://docs.omni.co/page-one", ["Section B"])[0]

    assert "# not a page heading" in section_b.body
    assert "## not a section heading" in section_b.body
    assert "Not A Page Heading" in section_b.body
    # neither fenced pseudo-heading nor the stray "# " line created new sections
    assert not _by_heading(
        sections, "https://docs.omni.co/page-one", ["not a section heading"]
    )
    assert not any(s.page_title == "Not A Page Heading" for s in sections)


def test_unterminated_fence_does_not_swallow_the_next_page():
    text = (FIXTURES / "corpus_unterminated_fence.txt").read_text(encoding="utf-8")

    sections = parse_docs_sections(text, max_body_chars=MAX_BODY_CHARS)

    page_urls = {s.page_url for s in sections}
    assert "https://docs.omni.co/fence-page-one" in page_urls
    assert "https://docs.omni.co/fence-page-two" in page_urls
    page_two_section_b = _by_heading(
        sections, "https://docs.omni.co/fence-page-two", ["Section B"]
    )
    assert page_two_section_b
    assert page_two_section_b[0].body == "Body B text."


def test_page_two_deep_start_and_repeated_headings_get_duplicate_index():
    sections = parse_docs_sections(SAMPLE_TEXT, max_body_chars=MAX_BODY_CHARS)
    deep_start = _by_heading(sections, "https://docs.omni.co/page-two", ["Deep Start"])[
        0
    ]
    assert deep_start.body == "Content that starts directly at heading level three."

    repeats = _by_heading(sections, "https://docs.omni.co/page-two", ["Repeated Title"])
    assert [r.body for r in repeats] == [
        "First occurrence body.",
        "Second occurrence body.",
    ]
    assert repeats[0].section_id != repeats[1].section_id
    expected_first = section_id(
        "https://docs.omni.co/page-two", ["Repeated Title"], 0, 0
    )
    expected_second = section_id(
        "https://docs.omni.co/page-two", ["Repeated Title"], 1, 0
    )
    assert repeats[0].section_id == expected_first
    assert repeats[1].section_id == expected_second


def test_content_hash_and_source_kind():
    sections = parse_docs_sections(SAMPLE_TEXT, max_body_chars=MAX_BODY_CHARS)
    section_a = _by_heading(sections, "https://docs.omni.co/page-one", ["Section A"])[0]
    assert (
        section_a.content_hash
        == hashlib.sha256(section_a.body.encode("utf-8")).hexdigest()
    )
    assert len(section_a.content_hash) == 64
    assert section_a.source_kind == "docs"


def test_empty_body_emits_no_row():
    text = (
        "# Empty Page\n"
        "Source: https://docs.omni.co/empty\n"
        "\n"
        "## Has Body\n"
        "\n"
        "Text.\n"
        "\n"
        "## No Body\n"
        "\n"
        "## Also Has Body\n"
        "\n"
        "More text.\n"
    )
    sections = parse_docs_sections(text, max_body_chars=MAX_BODY_CHARS)
    headings = [list(s.heading_path) for s in sections]
    assert ["No Body"] not in headings
    assert ["Has Body"] in headings
    assert ["Also Has Body"] in headings


def test_id_is_deterministic_and_order_independent():
    reordered = SAMPLE_TEXT.split("# Page Two", 1)
    swapped_text = "# Page Two" + reordered[1] + "\n" + reordered[0]
    first_run = parse_docs_sections(SAMPLE_TEXT, max_body_chars=MAX_BODY_CHARS)
    second_run = parse_docs_sections(swapped_text, max_body_chars=MAX_BODY_CHARS)

    def ids_for(sections, page_url):
        return {
            (tuple(s.heading_path), s.part_index): s.section_id
            for s in sections
            if s.page_url == page_url
        }

    assert ids_for(first_run, "https://docs.omni.co/page-two") == ids_for(
        second_run, "https://docs.omni.co/page-two"
    )


def test_size_limit_splits_into_fewest_contiguous_parts():
    long_body_lines = ["x" * 9000 for _ in range(3)]
    text = (
        "# Big Page\n"
        "Source: https://docs.omni.co/big\n"
        "\n"
        "## Long Section\n"
        "\n" + "\n\n".join(long_body_lines) + "\n"
    )
    sections = parse_docs_sections(text, max_body_chars=10000)
    parts = _by_heading(sections, "https://docs.omni.co/big", ["Long Section"])
    assert len(parts) >= 2
    assert all(len(p.body) <= 10000 for p in parts)
    assert [p.part_index for p in parts] == list(range(len(parts)))
    assert all(p.part_count == len(parts) for p in parts)
    original_body = "\n\n".join(long_body_lines)
    total_chars = sum(len(p.body) for p in parts)
    # each split point drops exactly the "\n" that used to join the two parts
    assert total_chars == len(original_body) - (len(parts) - 1)


def test_size_limit_hard_cuts_a_single_over_long_line():
    text = (
        "# Big Page\n"
        "Source: https://docs.omni.co/big\n"
        "\n"
        "## Long Line Section\n"
        "\n" + ("y" * 25000) + "\n"
    )
    sections = parse_docs_sections(text, max_body_chars=10000)
    parts = _by_heading(sections, "https://docs.omni.co/big", ["Long Line Section"])
    assert len(parts) == 3
    assert len(parts[0].body) == 10000
    assert len(parts[1].body) == 10000
    assert len(parts[2].body) == 5000
    assert parts[0].body + parts[1].body + parts[2].body == "y" * 25000


def test_page_boundary_requires_both_lines():
    text = "# Just A Heading\nNot a source line\n\n## Section\n\nBody text.\n"
    sections = parse_docs_sections(text, max_body_chars=MAX_BODY_CHARS)
    assert sections == []


def test_load_injected_sections_validates_schema(tmp_path):
    valid = load_injected_sections(INJECTED_PATH)
    assert valid[0]["section_id"] == "inj_section_b_note"

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sections": [
                    {
                        "section_id": "not_valid_prefix",
                        "page_url": "https://docs.omni.co/x",
                        "page_title": "X",
                        "heading_path": [],
                        "body": "text",
                        "anchor_page_url": "https://docs.omni.co/x",
                        "anchor_heading_path": [],
                        "placement": "after",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DocsCorpusError, match="section_id"):
        load_injected_sections(bad)


def test_load_injected_sections_rejects_oversized_body(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sections": [
                    {
                        "section_id": "inj_too_big",
                        "page_url": "https://docs.omni.co/x",
                        "page_title": "X",
                        "heading_path": [],
                        "body": "z" * (MAX_BODY_CHARS + 1),
                        "anchor_page_url": "https://docs.omni.co/x",
                        "anchor_heading_path": [],
                        "placement": "after",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DocsCorpusError, match="body"):
        load_injected_sections(bad)


def test_load_injected_sections_rejects_missing_schema_version(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"sections": []}), encoding="utf-8")
    with pytest.raises(DocsCorpusError, match="schema_version"):
        load_injected_sections(bad)


def test_load_injected_sections_rejects_non_string_page_url(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sections": [
                    {
                        "section_id": "inj_bad_page_url",
                        "page_url": 123,
                        "page_title": "X",
                        "heading_path": [],
                        "body": "text",
                        "anchor_page_url": "https://docs.omni.co/x",
                        "anchor_heading_path": [],
                        "placement": "after",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DocsCorpusError, match="page_url"):
        load_injected_sections(bad)


def test_load_injected_sections_rejects_non_string_heading_path_entry(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sections": [
                    {
                        "section_id": "inj_bad_heading",
                        "page_url": "https://docs.omni.co/x",
                        "page_title": "X",
                        "heading_path": [1, 2],
                        "body": "text",
                        "anchor_page_url": "https://docs.omni.co/x",
                        "anchor_heading_path": [],
                        "placement": "after",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DocsCorpusError, match="heading_path"):
        load_injected_sections(bad)


def test_splice_injected_inserts_after_anchor_and_recomputes_order():
    docs_sections = parse_docs_sections(SAMPLE_TEXT, max_body_chars=MAX_BODY_CHARS)
    injected = load_injected_sections(INJECTED_PATH)

    spliced = splice_injected(docs_sections, injected)

    page_one = [s for s in spliced if s.page_url == "https://docs.omni.co/page-one"]
    labels = [list(s.heading_path) for s in page_one]
    section_b_index = labels.index(["Section B"])
    injected_index = labels.index(["Section B", "Contradiction"])
    assert injected_index == section_b_index + 1
    injected_row = page_one[injected_index]
    assert injected_row.source_kind == "injected"
    assert injected_row.part_index == 0
    assert injected_row.part_count == 1
    assert injected_row.section_id == "inj_section_b_note"
    assert (
        injected_row.content_hash
        == hashlib.sha256(injected_row.body.encode("utf-8")).hexdigest()
    )
    assert [s.section_order for s in page_one] == list(range(len(page_one)))


def test_splice_injected_errors_on_missing_or_ambiguous_anchor():
    docs_sections = parse_docs_sections(SAMPLE_TEXT, max_body_chars=MAX_BODY_CHARS)
    missing_anchor = [
        {
            "section_id": "inj_missing",
            "page_url": "https://docs.omni.co/page-one",
            "page_title": "Page One",
            "heading_path": ["Nowhere"],
            "body": "text",
            "anchor_page_url": "https://docs.omni.co/page-one",
            "anchor_heading_path": ["Does Not Exist"],
            "placement": "after",
        }
    ]
    with pytest.raises(DocsCorpusError, match="anchor"):
        splice_injected(docs_sections, missing_anchor)


def test_build_corpus_matches_manual_splice():
    injected = load_injected_sections(INJECTED_PATH)
    combined = build_corpus(
        SAMPLE_TEXT, injected=injected, max_body_chars=MAX_BODY_CHARS
    )
    docs_only = build_corpus(SAMPLE_TEXT, injected=None, max_body_chars=MAX_BODY_CHARS)
    assert len(combined) == len(docs_only) + 1
    assert any(s.source_kind == "injected" for s in combined)


def test_row_json_shape_has_exact_keys():
    sections = parse_docs_sections(SAMPLE_TEXT, max_body_chars=MAX_BODY_CHARS)
    row = sections[0].to_row()
    assert set(row.keys()) == {
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
    }
    assert isinstance(row["heading_path"], list)


def test_unit_delimiter_value():
    assert UNIT == "\x1f"


def test_neon_ddl_contains_expected_objects():
    assert "CREATE SCHEMA IF NOT EXISTS omni_docs_qa" in NEON_DDL
    assert "CREATE TABLE IF NOT EXISTS omni_docs_qa.sections" in NEON_DDL
    assert "sections_page_url_idx" in NEON_DDL
    assert "sections_source_kind_idx" in NEON_DDL


def test_main_dry_run_writes_nothing(tmp_path, capsys):
    out_path = tmp_path / "corpus.jsonl"
    exit_code = main(
        [
            "--source",
            str(FIXTURES / "corpus_llms_sample.txt"),
            "--out",
            str(out_path),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not out_path.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["section_count"] > 0
    assert payload["page_count"] == 2
    assert payload["injected_count"] == 0
    assert "CREATE TABLE" in payload["ddl"]


def test_main_writes_corpus_and_meta(tmp_path, capsys):
    out_path = tmp_path / "artifacts" / "corpus.jsonl"
    exit_code = main(
        [
            "--input",
            str(FIXTURES / "corpus_llms_sample.txt"),
            "--output",
            str(out_path),
            "--inject",
            str(INJECTED_PATH),
        ]
    )
    assert exit_code == 0
    lines = out_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(rows) == len(lines)
    assert any(r["source_kind"] == "injected" for r in rows)

    meta_path = out_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["section_count"] == len(rows)
    assert meta["version"] is None
    assert meta["parent_version"] is None
    assert meta["applied_gap_ids"] == []
    assert meta["corpus_hash"] == hashlib.sha256(out_path.read_bytes()).hexdigest()

    payload = json.loads(capsys.readouterr().out)
    assert payload["section_count"] == len(rows)


def test_main_plan_sql_writes_ddl_file(tmp_path):
    plan_path = tmp_path / "plan.sql"
    out_path = tmp_path / "corpus.jsonl"
    main(
        [
            "--source",
            str(FIXTURES / "corpus_llms_sample.txt"),
            "--out",
            str(out_path),
            "--plan-sql",
            str(plan_path),
        ]
    )
    contents = plan_path.read_text(encoding="utf-8")
    assert "CREATE TABLE" in contents
    assert "\\copy omni_docs_qa.sections" in contents


def test_main_plan_sql_not_written_under_dry_run(tmp_path):
    plan_path = tmp_path / "plan.sql"
    out_path = tmp_path / "corpus.jsonl"
    exit_code = main(
        [
            "--source",
            str(FIXTURES / "corpus_llms_sample.txt"),
            "--out",
            str(out_path),
            "--plan-sql",
            str(plan_path),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not plan_path.exists()


def test_main_missing_source_file_reports_error(capsys):
    exit_code = main(["--source", "/nonexistent/path/does-not-exist.txt"])
    assert exit_code == 1
    error = json.loads(capsys.readouterr().err)
    assert "does-not-exist.txt" in error["error"]


def test_main_bad_injected_section_reports_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 1, "sections": []}), encoding="utf-8")
    # empty sections list is valid; craft a genuinely invalid file instead
    bad.write_text("{}", encoding="utf-8")
    exit_code = main(
        [
            "--source",
            str(FIXTURES / "corpus_llms_sample.txt"),
            "--inject",
            str(bad),
            "--dry-run",
        ]
    )
    assert exit_code == 1
    error = json.loads(capsys.readouterr().err)
    assert "sections" in error["error"]

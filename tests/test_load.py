import json
from pathlib import Path

import pytest

from omni_docs_qa.corpus import NEON_DDL
from omni_docs_qa.load import (
    COPY_STATEMENT,
    TRUNCATE_STATEMENT,
    DocsLoadError,
    load,
    main,
    read_rows,
    snapshot_version_from_meta,
)

ROW = {
    "section_id": "sec_1",
    "page_url": "https://docs.omni.co/a",
    "page_title": "A",
    "heading_path": ["Intro"],
    "heading_label": "Intro",
    "section_order": 0,
    "part_index": 0,
    "part_count": 1,
    "body": "text",
    "content_hash": "h",
    "source_kind": "docs",
}


class FakeCopy:
    def __init__(self) -> None:
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def write_row(self, row):
        self.rows.append(row)


class FakeCursor:
    def __init__(self) -> None:
        self.statements = []
        self.copy_target = FakeCopy()

    def execute(self, statement):
        self.statements.append(statement)

    def copy(self, statement):
        self.statements.append(statement)
        return self.copy_target


def _write_corpus(tmp_path: Path, rows) -> Path:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return corpus


def test_read_rows_appends_snapshot_version(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path, [ROW])

    rows = list(read_rows(corpus, "v1"))

    assert rows == [
        (
            "sec_1",
            "https://docs.omni.co/a",
            "A",
            ["Intro"],
            "Intro",
            0,
            0,
            1,
            "text",
            "h",
            "docs",
            "v1",
        )
    ]


@pytest.mark.parametrize(
    "line, message",
    [
        ("not json", "invalid JSON"),
        ("[]", "expected an object"),
        (json.dumps({**ROW, "heading_path": "Intro"}), "heading_path"),
        (json.dumps({k: v for k, v in ROW.items() if k != "body"}), "missing body"),
    ],
)
def test_read_rows_rejects_bad_lines(tmp_path: Path, line: str, message: str) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(DocsLoadError, match=message):
        list(read_rows(corpus, "v1"))


def test_load_runs_ddl_then_copies_every_row(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path, [ROW, {**ROW, "section_id": "sec_2"}])
    cursor = FakeCursor()

    count = load(cursor, read_rows(corpus, "v1"))

    assert count == 2
    assert cursor.statements == [NEON_DDL, COPY_STATEMENT]
    assert [row[0] for row in cursor.copy_target.rows] == ["sec_1", "sec_2"]


def test_load_with_replace_truncates_before_copying(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path, [ROW])
    cursor = FakeCursor()

    count = load(cursor, read_rows(corpus, "v001"), replace=True)

    assert count == 1
    assert cursor.statements == [NEON_DDL, TRUNCATE_STATEMENT, COPY_STATEMENT]
    assert cursor.copy_target.rows[0][-1] == "v001"


def test_main_dry_run_reports_replace_flag(tmp_path: Path, capsys, monkeypatch) -> None:
    corpus = _write_corpus(tmp_path, [ROW])
    monkeypatch.delenv("DOCS_QA_ADMIN_DSN", raising=False)

    assert (
        main(
            [
                "--corpus",
                str(corpus),
                "--snapshot-version",
                "v001",
                "--dry-run",
                "--replace",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["replace"] is True


def test_snapshot_version_prefers_explicit_version_then_hash(tmp_path: Path) -> None:
    meta = tmp_path / "corpus.meta.json"
    meta.write_text(json.dumps({"version": None, "corpus_hash": "a" * 64}))
    assert snapshot_version_from_meta(meta) == "base-aaaaaaaaaaaa"
    meta.write_text(json.dumps({"version": "v002", "corpus_hash": "a" * 64}))
    assert snapshot_version_from_meta(meta) == "v002"
    meta.write_text(json.dumps({"version": None}))
    with pytest.raises(DocsLoadError, match="neither"):
        snapshot_version_from_meta(meta)


def test_main_dry_run_counts_without_a_dsn(tmp_path: Path, capsys, monkeypatch) -> None:
    corpus = _write_corpus(tmp_path, [ROW])
    corpus.with_suffix(".meta.json").write_text(json.dumps({"corpus_hash": "b" * 64}))
    monkeypatch.delenv("DOCS_QA_ADMIN_DSN", raising=False)

    assert main(["--corpus", str(corpus), "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["rows"] == 1
    assert output["snapshot_version"] == "base-bbbbbbbbbbbb"


def test_main_refuses_to_load_without_dsn(tmp_path: Path, capsys, monkeypatch) -> None:
    corpus = _write_corpus(tmp_path, [ROW])
    monkeypatch.delenv("DOCS_QA_ADMIN_DSN", raising=False)

    assert main(["--corpus", str(corpus), "--snapshot-version", "v1"]) == 1
    assert "DOCS_QA_ADMIN_DSN" in capsys.readouterr().err

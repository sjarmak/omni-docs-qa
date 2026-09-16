from pathlib import Path

import yaml

from omni_docs_qa.model_deploy import model_files


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models" / "docs_qa"
VIEW_NAME = "docs_qa_omni_docs_qa__sections"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_docs_qa_model_has_the_reviewed_file_set() -> None:
    files = {
        path.relative_to(MODEL_ROOT).as_posix()
        for path in MODEL_ROOT.rglob("*")
        if path.is_file()
    }

    assert files == {
        "model",
        "docs_sections.topic",
        "docs_qa.omni_docs_qa/sections.view",
        "relationships",
    }


def test_relationships_file_is_empty_array() -> None:
    assert (MODEL_ROOT / "relationships").read_text(encoding="utf-8") == "[]\n"


def test_view_reference_comment_and_table_identity() -> None:
    view_path = MODEL_ROOT / "docs_qa.omni_docs_qa/sections.view"
    text = view_path.read_text(encoding="utf-8")

    assert text.splitlines()[0] == f"# Reference this view as {VIEW_NAME}"

    view = _load_yaml(view_path)
    assert view["catalog"] == "docs_qa"
    assert view["schema"] == "omni_docs_qa"
    assert view["table_name"] == "sections"


def test_topic_base_view_matches_view_name() -> None:
    topic = _load_yaml(MODEL_ROOT / "docs_sections.topic")

    assert topic["base_view"] == VIEW_NAME
    assert topic["label"] == "Omni Documentation Sections"
    assert topic["fields"] == [f"{VIEW_NAME}.*"]
    assert "joins" not in topic


def test_ai_context_present_on_model_topic_and_body_dimension() -> None:
    model = _load_yaml(MODEL_ROOT / "model")
    topic = _load_yaml(MODEL_ROOT / "docs_sections.topic")
    view = _load_yaml(MODEL_ROOT / "docs_qa.omni_docs_qa/sections.view")

    assert model["ai_chat_topics"] == ["docs_sections"]
    assert isinstance(model.get("ai_context"), str) and model["ai_context"].strip()
    assert isinstance(topic.get("ai_context"), str) and topic["ai_context"].strip()

    body_dimension = view["dimensions"]["body"]
    assert (
        isinstance(body_dimension.get("ai_context"), str)
        and body_dimension["ai_context"].strip()
    )


def test_model_disables_the_query_cache_so_snapshot_swaps_take_effect() -> None:
    model = _load_yaml(MODEL_ROOT / "model")

    policy = model["default_cache_policy"]
    assert model["cache_policies"][policy]["max_cache_age"] == "0 seconds"


def test_view_dimensions_and_measures_match_contract() -> None:
    view = _load_yaml(MODEL_ROOT / "docs_qa.omni_docs_qa/sections.view")
    dimensions = view["dimensions"]

    assert set(dimensions) == {
        "section_id",
        "page_url",
        "page_title",
        "heading_label",
        "heading_path",
        "section_order",
        "part_index",
        "part_count",
        "body",
        "content_hash",
        "source_kind",
        "snapshot_version",
    }

    assert dimensions["section_id"]["format"] == "ID"
    assert dimensions["section_id"]["primary_key"] is True
    assert dimensions["heading_path"]["hidden"] is True
    assert dimensions["part_index"]["hidden"] is True
    assert dimensions["part_count"]["hidden"] is True
    assert dimensions["content_hash"]["hidden"] is True
    assert dimensions["page_title"]["synonyms"] == ["page", "doc page"]
    assert dimensions["heading_label"]["synonyms"] == ["heading", "section heading"]

    hidden_dimensions = {"heading_path", "part_index", "part_count", "content_hash"}
    for name in hidden_dimensions:
        assert dimensions[name]["hidden"] is True
    for name, dimension in dimensions.items():
        if name not in hidden_dimensions:
            assert dimension.get("hidden") is not True, (
                f"dimension {name} unexpectedly hidden"
            )

    for name, dimension in dimensions.items():
        assert (
            isinstance(dimension.get("description"), str)
            and dimension["description"].strip()
        ), f"dimension {name} missing description"

    ai_context_dimensions = {"body", "source_kind"}
    for name in ai_context_dimensions:
        assert (
            isinstance(dimensions[name].get("ai_context"), str)
            and dimensions[name]["ai_context"].strip()
        ), f"dimension {name} missing ai_context"
    for name, dimension in dimensions.items():
        if name not in ai_context_dimensions:
            assert "ai_context" not in dimension, f"unexpected ai_context on {name}"

    measures = view["measures"]
    assert set(measures) == {"sections", "pages"}
    assert measures["sections"]["aggregate_type"] == "count"
    assert measures["sections"]["description"].strip()
    assert measures["pages"]["aggregate_type"] == "count_distinct"
    assert measures["pages"]["sql"] == "${page_url}"
    assert measures["pages"]["description"].strip()


def test_model_deploy_model_files_reads_docs_qa_model() -> None:
    files = model_files(MODEL_ROOT)
    names = {file.name for file in files}

    assert names == {
        "model",
        "docs_sections.topic",
        "docs_qa.omni_docs_qa/sections.view",
        "relationships",
    }
    assert len(files) == 4

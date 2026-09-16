import json
from pathlib import Path

import pytest

from omni_docs_qa.model_deploy import (
    DeployError,
    deploy,
    model_files,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def run(self, *arguments, stdin=None):
        self.calls.append((arguments, stdin))
        return self.responses.pop(0)


def test_model_files_are_recursive_and_sorted(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "z.topic").write_text("base_view: z\n", encoding="utf-8")
    (tmp_path / "nested" / "a.view").write_text("table_name: a\n", encoding="utf-8")

    files = model_files(tmp_path)

    assert [(file.name, file.content) for file in files] == [
        ("nested/a.view", "table_name: a\n"),
        ("z.topic", "base_view: z\n"),
    ]


@pytest.mark.parametrize("state", ["missing", "empty"])
def test_model_files_reject_missing_or_empty_directory(
    tmp_path: Path, state: str
) -> None:
    root = tmp_path / "model"
    if state == "empty":
        root.mkdir()

    with pytest.raises(DeployError, match="missing|no files"):
        model_files(root)


def test_deploy_reuses_model_validates_and_merges(tmp_path: Path) -> None:
    (tmp_path / "model").write_text("ai_chat_topics: [digest]\n", encoding="utf-8")
    client = FakeClient(
        [
            {
                "records": [{"id": "model-1", "name": "Digest"}],
                "pageInfo": {"hasNextPage": False},
            },
            {"branch": {"id": "branch-1"}},
            {"records": [], "pageInfo": {"hasNextPage": False}},
            {
                "records": [{"id": "branch-1"}],
                "pageInfo": {"hasNextPage": False},
            },
            {"success": True},
            [],
            {"success": True},
        ]
    )
    waits = []

    result = deploy(
        client,
        connection_id="connection-1",
        model_name="Digest",
        files=model_files(tmp_path),
        branch_name="digest-v1",
        wait=waits.append,
    )

    assert result.model_id == "model-1"
    assert result.branch_id == "branch-1"
    assert result.merged is True
    assert waits == [1.0]
    upload_body = json.loads(client.calls[4][1])
    assert upload_body == {
        "branchId": "branch-1",
        "commitMessage": "Deploy Digest semantic model",
        "fileName": "model",
        "mode": "combined",
        "yaml": "ai_chat_topics: [digest]\n",
    }
    assert client.calls[-1] == (
        ("models", "merge-branch", "model-1", "digest-v1", "--body", "-"),
        json.dumps(
            {"commit_message": "Deploy Digest semantic model", "delete_branch": False}
        ),
    )


def test_deploy_creates_model_and_stops_on_validation_issues(tmp_path: Path) -> None:
    (tmp_path / "model").write_text("{}\n", encoding="utf-8")
    client = FakeClient(
        [
            {"records": [], "pageInfo": {"hasNextPage": False}},
            {"records": [], "pageInfo": {"hasNextPage": False}},
            {"model": {"id": "schema-model"}},
            {"jobId": "refresh-job", "status": "running"},
            {"status": "IN_PROGRESS"},
            {"status": "COMPLETED"},
            {"model": {"id": "model-new"}},
            {"branchId": "branch-1"},
            {
                "records": [{"id": "branch-1"}],
                "pageInfo": {"hasNextPage": False},
            },
            {"success": True},
            [{"message": "bad field"}],
        ]
    )
    waits = []

    result = deploy(
        client,
        connection_id="connection-1",
        model_name="Digest",
        files=model_files(tmp_path),
        branch_name="digest-v1",
        wait=waits.append,
    )

    assert result.model_id == "model-new"
    assert result.validation == ({"message": "bad field"},)
    assert result.merged is False
    assert json.loads(client.calls[2][1]) == {
        "connectionId": "connection-1",
        "modelKind": "SCHEMA",
    }
    assert client.calls[3] == (
        ("models", "refresh", "schema-model"),
        None,
    )
    assert client.calls[4] == (
        ("models", "jobs-get-status", "refresh-job"),
        None,
    )
    assert client.calls[5] == (
        ("models", "jobs-get-status", "refresh-job"),
        None,
    )
    assert waits == [1.0]
    assert json.loads(client.calls[6][1]) == {
        "connectionId": "connection-1",
        "modelKind": "SHARED",
        "modelName": "Digest",
    }
    assert not any(call[0][1] == "merge-branch" for call in client.calls)


@pytest.mark.parametrize(
    ("refresh_response", "status_response", "message"),
    [
        ("malformed", None, "refresh response is malformed"),
        ({}, None, "no job id"),
        (
            {"jobId": "refresh-job"},
            "malformed",
            "status response is malformed",
        ),
        ({"jobId": "refresh-job"}, {"status": "FAILED"}, "refresh failed"),
        ({"jobId": "refresh-job"}, {"status": "UNKNOWN"}, "unknown status"),
    ],
)
def test_deploy_rejects_invalid_schema_refresh(
    tmp_path: Path,
    refresh_response: object,
    status_response: object,
    message: str,
) -> None:
    (tmp_path / "model").write_text("{}\n", encoding="utf-8")
    responses = [
        {"records": [], "pageInfo": {"hasNextPage": False}},
        {
            "records": [{"id": "schema-model"}],
            "pageInfo": {"hasNextPage": False},
        },
        refresh_response,
    ]
    if status_response is not None:
        responses.append(status_response)

    with pytest.raises(DeployError, match=message):
        deploy(
            FakeClient(responses),
            connection_id="connection-1",
            model_name="Digest",
            files=model_files(tmp_path),
            branch_name="digest-v1",
        )


def test_deploy_stops_after_schema_refresh_poll_limit(tmp_path: Path) -> None:
    (tmp_path / "model").write_text("{}\n", encoding="utf-8")
    client = FakeClient(
        [
            {"records": [], "pageInfo": {"hasNextPage": False}},
            {
                "records": [{"id": "schema-model"}],
                "pageInfo": {"hasNextPage": False},
            },
            {"jobId": "refresh-job"},
            *({"status": "IN_PROGRESS"} for _ in range(60)),
        ]
    )
    waits = []

    with pytest.raises(DeployError, match="poll limit"):
        deploy(
            client,
            connection_id="connection-1",
            model_name="Digest",
            files=model_files(tmp_path),
            branch_name="digest-v1",
            wait=waits.append,
        )

    status_calls = [
        call for call in client.calls if call[0][:2] == ("models", "jobs-get-status")
    ]
    assert len(status_calls) == 60
    assert waits == [1.0] * 59
    assert not any(call[0][:2] == ("models", "create") for call in client.calls[3:])


@pytest.mark.parametrize(
    ("branch_response", "message"),
    [
        ("malformed", "branch lookup response is malformed"),
        (
            {"records": [], "pageInfo": None},
            "branch lookup pagination metadata is malformed",
        ),
        (
            {"records": [], "pageInfo": {"hasNextPage": True}},
            "more pages",
        ),
        (
            {
                "records": [{"id": "another-branch"}],
                "pageInfo": {"hasNextPage": False},
            },
            "unexpected model",
        ),
        (
            {
                "records": [{"id": "branch-1"}, "malformed"],
                "pageInfo": {"hasNextPage": False},
            },
            "branch lookup record is malformed",
        ),
    ],
)
def test_deploy_rejects_invalid_branch_lookup(
    tmp_path: Path, branch_response: object, message: str
) -> None:
    (tmp_path / "model").write_text("{}\n", encoding="utf-8")
    client = FakeClient(
        [
            {
                "records": [{"id": "model-1", "name": "Digest"}],
                "pageInfo": {"hasNextPage": False},
            },
            {"branchId": "branch-1"},
            branch_response,
        ]
    )

    with pytest.raises(DeployError, match=message):
        deploy(
            client,
            connection_id="connection-1",
            model_name="Digest",
            files=model_files(tmp_path),
            branch_name="digest-v1",
        )


def test_deploy_stops_after_branch_visibility_poll_limit(tmp_path: Path) -> None:
    (tmp_path / "model").write_text("{}\n", encoding="utf-8")
    empty_lookup = {"records": [], "pageInfo": {"hasNextPage": False}}
    client = FakeClient(
        [
            {
                "records": [{"id": "model-1", "name": "Digest"}],
                "pageInfo": {"hasNextPage": False},
            },
            {"branchId": "branch-1"},
            *(empty_lookup for _ in range(60)),
        ]
    )
    waits = []

    with pytest.raises(DeployError, match="branch did not become visible"):
        deploy(
            client,
            connection_id="connection-1",
            model_name="Digest",
            files=model_files(tmp_path),
            branch_name="digest-v1",
            wait=waits.append,
        )

    assert waits == [1.0] * 59
    assert len(client.calls) == 62


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        (
            [{"records": ["malformed"], "pageInfo": {"hasNextPage": False}}],
            "shared model record is malformed",
        ),
        (
            [
                {"records": [], "pageInfo": {"hasNextPage": False}},
                {"records": ["malformed"], "pageInfo": {"hasNextPage": False}},
            ],
            "schema model record is malformed",
        ),
    ],
)
def test_deploy_rejects_malformed_model_records(
    tmp_path: Path, responses: list[object], message: str
) -> None:
    (tmp_path / "model").write_text("{}\n", encoding="utf-8")

    with pytest.raises(DeployError, match=message):
        deploy(
            FakeClient(responses),
            connection_id="connection-1",
            model_name="Digest",
            files=model_files(tmp_path),
            branch_name="digest-v1",
        )


def test_deploy_refuses_ambiguous_or_paginated_model_lookup(tmp_path: Path) -> None:
    (tmp_path / "model").write_text("{}\n", encoding="utf-8")
    files = model_files(tmp_path)

    for response in (
        {
            "records": [
                {"id": "one", "name": "Digest"},
                {"id": "two", "name": "Digest"},
            ],
            "pageInfo": {"hasNextPage": False},
        },
        {"records": [], "pageInfo": {"hasNextPage": True}},
    ):
        client = FakeClient([response])
        with pytest.raises(DeployError, match="more pages|shared models"):
            deploy(
                client,
                connection_id="connection-1",
                model_name="Digest",
                files=files,
                branch_name="digest-v1",
            )


@pytest.mark.parametrize(
    "response",
    [
        {"records": []},
        {"records": [], "pageInfo": None},
        {"records": [], "pageInfo": "none"},
        {"records": [], "pageInfo": {}},
        {"records": [], "pageInfo": {"hasNextPage": 0}},
    ],
)
def test_deploy_rejects_malformed_pagination_metadata(
    tmp_path: Path, response: object
) -> None:
    (tmp_path / "model").write_text("{}\n", encoding="utf-8")

    with pytest.raises(DeployError, match="pagination metadata"):
        deploy(
            FakeClient([response]),
            connection_id="connection-1",
            model_name="Digest",
            files=model_files(tmp_path),
            branch_name="digest-v1",
        )

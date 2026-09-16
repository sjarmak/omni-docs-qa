"""Branch-first deployment of reviewed Omni semantic-model YAML."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from omni_docs_qa.omni_cli import OmniCli


REFRESH_MAX_POLLS = 60
REFRESH_POLL_INTERVAL_SECONDS = 1.0
BRANCH_MAX_POLLS = 60
BRANCH_POLL_INTERVAL_SECONDS = 1.0


class DeployError(RuntimeError):
    """A semantic-model deployment cannot continue safely."""


class Client(Protocol):
    def run(self, *arguments: str, stdin: str | None = None) -> Any: ...


@dataclass(frozen=True)
class ModelFile:
    name: str
    content: str


@dataclass(frozen=True)
class DeployResult:
    model_id: str
    branch_id: str
    branch_name: str
    uploaded: tuple[str, ...]
    validation: tuple[Any, ...]
    merged: bool


class OmniClient:
    def __init__(self, *, profile: str, binary: str = "omni") -> None:
        self._cli = OmniCli(profile=profile, binary=binary)

    def run(self, *arguments: str, stdin: str | None = None) -> Any:
        return self._cli.run_json(arguments, stdin=stdin)


def model_files(root: Path) -> tuple[ModelFile, ...]:
    if not root.is_dir():
        raise DeployError(f"model directory missing: {root}")
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    if not paths:
        raise DeployError(f"model directory has no files: {root}")
    return tuple(
        ModelFile(path.relative_to(root).as_posix(), path.read_text(encoding="utf-8"))
        for path in paths
    )


def deploy(
    client: Client,
    *,
    connection_id: str,
    model_name: str,
    files: tuple[ModelFile, ...],
    branch_name: str,
    wait: Callable[[float], None] = time.sleep,
) -> DeployResult:
    model_id = _find_model(client, connection_id, model_name)
    if model_id is None:
        schema_model_id = _find_schema_model(client, connection_id)
        if schema_model_id is None:
            schema_model_id = _record_id(
                client.run(
                    "models",
                    "create",
                    "--body",
                    "-",
                    stdin=json.dumps(
                        {
                            "connectionId": connection_id,
                            "modelKind": "SCHEMA",
                        }
                    ),
                ),
                "schema model creation",
            )
        _refresh_schema_model(client, schema_model_id, wait=wait)
        model_id = _record_id(
            client.run(
                "models",
                "create",
                "--body",
                "-",
                stdin=json.dumps(
                    {
                        "connectionId": connection_id,
                        "modelKind": "SHARED",
                        "modelName": model_name,
                    }
                ),
            ),
            "model creation",
        )
    branch_id = _record_id(
        client.run("models", "create-branch", model_id, "--name", branch_name),
        "branch creation",
    )
    _wait_for_branch(client, branch_id, wait=wait)
    commit_message = f"Deploy {model_name} semantic model"
    for file in files:
        client.run(
            "models",
            "yaml-create",
            model_id,
            "--body",
            "-",
            stdin=json.dumps(
                {
                    "branchId": branch_id,
                    "commitMessage": commit_message,
                    "fileName": file.name,
                    "mode": "combined",
                    "yaml": file.content,
                }
            ),
        )
    validation = client.run("models", "validate", model_id, "--branchid", branch_id)
    if not isinstance(validation, list):
        raise DeployError("model validation response must be a list")
    validation_issues = tuple(validation)
    if not validation_issues:
        client.run(
            "models",
            "merge-branch",
            model_id,
            branch_name,
            "--body",
            "-",
            stdin=json.dumps(
                {"commit_message": commit_message, "delete_branch": False}
            ),
        )
    return DeployResult(
        model_id,
        branch_id,
        branch_name,
        tuple(file.name for file in files),
        validation_issues,
        not validation_issues,
    )


def _find_model(client: Client, connection_id: str, model_name: str) -> str | None:
    response = client.run(
        "models",
        "list",
        "--connectionid",
        connection_id,
        "--modelkind",
        "SHARED",
        "--name",
        model_name,
    )
    records = _validated_records(response, "models list", "shared model")
    matches = [record["id"] for record in records if record.get("name") == model_name]
    if len(matches) > 1:
        raise DeployError(f"found {len(matches)} shared models named {model_name!r}")
    if not matches:
        return None
    return matches[0]


def _validated_records(
    response: Any, response_context: str, record_context: str
) -> list[dict[str, Any]]:
    if not isinstance(response, dict) or not isinstance(response.get("records"), list):
        raise DeployError(f"{response_context} response is malformed")
    page_info = response.get("pageInfo")
    if not isinstance(page_info, dict) or not isinstance(
        page_info.get("hasNextPage"), bool
    ):
        raise DeployError(f"{response_context} pagination metadata is malformed")
    if page_info["hasNextPage"]:
        raise DeployError(
            f"{response_context} has more pages; refusing a partial lookup"
        )
    records = response["records"]
    validated = []
    for record in records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("id"), str)
            or not record["id"]
        ):
            raise DeployError(f"{record_context} record is malformed")
        validated.append(record)
    return validated


def _refresh_schema_model(
    client: Client,
    schema_model_id: str,
    *,
    wait: Callable[[float], None],
) -> None:
    response = client.run("models", "refresh", schema_model_id)
    if not isinstance(response, dict):
        raise DeployError("schema refresh response is malformed")
    job_id = response.get("jobId")
    if not isinstance(job_id, str) or not job_id:
        raise DeployError("schema refresh response carries no job id")
    for attempt in range(REFRESH_MAX_POLLS):
        status_response = client.run("models", "jobs-get-status", job_id)
        if not isinstance(status_response, dict):
            raise DeployError("schema refresh status response is malformed")
        status = status_response.get("status")
        if status == "COMPLETED":
            return
        if status == "FAILED":
            raise DeployError("schema refresh failed")
        if status != "IN_PROGRESS":
            raise DeployError("schema refresh returned an unknown status")
        if attempt == REFRESH_MAX_POLLS - 1:
            raise DeployError("schema refresh did not complete before the poll limit")
        wait(REFRESH_POLL_INTERVAL_SECONDS)


def _wait_for_branch(
    client: Client,
    branch_id: str,
    *,
    wait: Callable[[float], None],
) -> None:
    for attempt in range(BRANCH_MAX_POLLS):
        response = client.run("models", "list", "--modelid", branch_id)
        records = _validated_records(response, "branch lookup", "branch lookup")
        identifiers = [record["id"] for record in records]
        if identifiers == [branch_id]:
            return
        if identifiers:
            raise DeployError("branch lookup returned an unexpected model")
        if attempt == BRANCH_MAX_POLLS - 1:
            raise DeployError("branch did not become visible before the poll limit")
        wait(BRANCH_POLL_INTERVAL_SECONDS)


def _find_schema_model(client: Client, connection_id: str) -> str | None:
    response = client.run(
        "models",
        "list",
        "--connectionid",
        connection_id,
        "--modelkind",
        "SCHEMA",
    )
    records = _validated_records(response, "schema models list", "schema model")
    identifiers = [record["id"] for record in records]
    if len(identifiers) > 1:
        raise DeployError("found multiple schema models for one connection")
    if not identifiers:
        return None
    return identifiers[0]


def _record_id(response: Any, operation: str) -> str:
    if isinstance(response, dict):
        for key in ("id", "branchId"):
            value = response.get(key)
            if isinstance(value, str) and value:
                return value
        for key in ("model", "branch"):
            value = response.get(key)
            if isinstance(value, dict) and isinstance(value.get("id"), str):
                return value["id"]
    raise DeployError(f"{operation} response carries no id")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        default=os.environ.get("OMNI_PROFILE") or None,
        help="Omni CLI profile (defaults to $OMNI_PROFILE)",
    )
    parser.add_argument("--branch-name")
    parser.add_argument("--live", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.live and arguments.profile is None:
        parser.error("--live needs --profile (or OMNI_PROFILE)")
    files = model_files(arguments.model_dir)
    branch_name = arguments.branch_name or (
        "semantic-model-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    if not arguments.live:
        print(json.dumps({"branch": branch_name, "files": [f.name for f in files]}))
        return 0
    result = deploy(
        OmniClient(profile=arguments.profile),
        connection_id=arguments.connection_id,
        model_name=arguments.model_name,
        files=files,
        branch_name=branch_name,
    )
    print(json.dumps(result.__dict__, default=list, sort_keys=True))
    return 0 if result.merged else 1


if __name__ == "__main__":
    raise SystemExit(main())

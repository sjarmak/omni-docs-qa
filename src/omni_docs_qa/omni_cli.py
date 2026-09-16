"""Small, credential-safe boundary around the Omni CLI."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from typing import Any


class OmniCliError(RuntimeError):
    """An Omni CLI operation failed or returned malformed data."""


Runner = Callable[
    [Sequence[str], Mapping[str, str], str | None, float], tuple[int, str, str]
]
SAFE_ENVIRONMENT = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "OMNI_CONFIG_PATH",
        "PATH",
        "XDG_CONFIG_HOME",
    }
)


class OmniCli:
    def __init__(
        self,
        *,
        profile: str,
        binary: str = "omni",
        timeout_seconds: float = 60.0,
        runner: Runner | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not profile.strip():
            raise OmniCliError("Omni profile must be set")
        self._prefix = (binary, "--compact", "--profile", profile.strip())
        self._timeout_seconds = timeout_seconds
        self._runner = _subprocess_runner if runner is None else runner
        source_environment = os.environ if environment is None else environment
        self._environment = {
            key: value
            for key, value in source_environment.items()
            if key in SAFE_ENVIRONMENT and value
        }

    def create_connection(self, request: Mapping[str, object]) -> str:
        response = self._run(
            ("connections", "create", "--body", "-"),
            stdin=json.dumps(request, separators=(",", ":"), sort_keys=True),
            secrets=(str(request.get("passwordUnencrypted", "")),),
        )
        identifier = (
            response.get("data") if response.get("success") else response.get("id")
        )
        if not isinstance(identifier, str) or not identifier:
            raise OmniCliError("Omni connection response carries no id")
        return identifier

    def run_json(self, command: Sequence[str], *, stdin: str | None = None) -> Any:
        """Run a non-secret Omni command and return its decoded JSON value."""
        return self._run_json(command, stdin=stdin)

    def _run(
        self,
        command: Sequence[str],
        *,
        stdin: str | None = None,
        secrets: Sequence[str] = (),
    ) -> dict[str, Any]:
        result = self._run_json(command, stdin=stdin, secrets=secrets)
        if not isinstance(result, dict):
            raise OmniCliError("Omni CLI JSON response must be an object")
        return result

    def _run_json(
        self,
        command: Sequence[str],
        *,
        stdin: str | None = None,
        secrets: Sequence[str] = (),
    ) -> Any:
        arguments = (*self._prefix, *command)
        has_secrets = any(secrets)
        secret_timeout = False
        try:
            returncode, stdout, stderr = self._runner(
                arguments, self._environment, stdin, self._timeout_seconds
            )
        except subprocess.TimeoutExpired as error:
            if has_secrets:
                secret_timeout = True
            else:
                raise OmniCliError("Omni CLI request could not complete") from error
        except OSError as error:
            raise OmniCliError("Omni CLI request could not complete") from error
        if secret_timeout:
            raise OmniCliError(
                "Omni CLI request could not complete; secret-bearing response suppressed"
            )
        if returncode != 0:
            if has_secrets:
                raise OmniCliError(
                    "Omni CLI request failed; secret-bearing response suppressed"
                )
            detail = stderr.strip() or stdout.strip() or "no detail"
            raise OmniCliError(f"Omni CLI request failed: {detail}")
        invalid_json = False
        try:
            result = json.loads(stdout)
        except (UnicodeError, json.JSONDecodeError):
            invalid_json = True
        if invalid_json:
            if has_secrets:
                raise OmniCliError(
                    "Omni CLI returned malformed output; secret-bearing response suppressed"
                )
            raise OmniCliError("Omni CLI did not return valid JSON")
        return result


def _subprocess_runner(
    arguments: Sequence[str],
    environment: Mapping[str, str],
    stdin: str | None,
    timeout_seconds: float,
) -> tuple[int, str, str]:
    completed = subprocess.run(
        list(arguments),
        input=stdin,
        capture_output=True,
        check=False,
        env=dict(environment),
        text=True,
        timeout=timeout_seconds,
    )
    return completed.returncode, completed.stdout, completed.stderr

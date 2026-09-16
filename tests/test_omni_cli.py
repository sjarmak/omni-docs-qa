import json
import subprocess

import pytest

from omni_docs_qa.omni_cli import OmniCli, OmniCliError


def test_create_connection_passes_secret_only_via_stdin() -> None:
    calls = []

    def runner(arguments, environment, stdin, timeout_seconds):
        calls.append((arguments, environment, stdin, timeout_seconds))
        return 0, '{"success":true,"data":"connection-123"}', ""

    cli = OmniCli(profile="experiments", runner=runner, environment={"PATH": "/bin"})
    request = {
        "dialect": "postgres",
        "name": "Test",
        "passwordUnencrypted": "top-secret",
    }

    assert cli.create_connection(request) == "connection-123"
    arguments, environment, stdin, timeout = calls[0]
    assert arguments == (
        "omni",
        "--compact",
        "--profile",
        "experiments",
        "connections",
        "create",
        "--body",
        "-",
    )
    assert environment == {"PATH": "/bin"}
    assert json.loads(stdin) == request
    assert "top-secret" not in " ".join(arguments)
    assert timeout == 60.0


def test_cli_redacts_password_if_omni_echoes_it_in_an_error() -> None:
    def runner(arguments, environment, stdin, timeout_seconds):
        return 1, "", "invalid password top-secret"

    cli = OmniCli(profile="experiments", runner=runner)
    request = {
        "dialect": "postgres",
        "name": "Test",
        "passwordUnencrypted": "top-secret",
    }

    with pytest.raises(OmniCliError, match="response suppressed") as error:
        cli.create_connection(request)
    assert "top-secret" not in str(error.value)


@pytest.mark.parametrize("password", ['pa"ss\\word', "line1\nline2", "pässword"])
def test_cli_redacts_json_encoded_passwords(password: str) -> None:
    encoded = json.dumps(password)[1:-1]

    def runner(arguments, environment, stdin, timeout_seconds):
        return 1, "", f'bad request: {{"passwordUnencrypted":"{encoded}"}}'

    cli = OmniCli(profile="experiments", runner=runner)
    with pytest.raises(OmniCliError) as error:
        cli.create_connection(
            {
                "dialect": "postgres",
                "name": "Test",
                "passwordUnencrypted": password,
            }
        )
    assert password not in str(error.value)
    assert encoded not in str(error.value)
    assert "response suppressed" in str(error.value)


def test_cli_suppresses_go_html_escaped_and_nested_secret_errors() -> None:
    password = "<admin>&secret"
    escaped = "\\u003cadmin\\u003e\\u0026secret"

    def runner(arguments, environment, stdin, timeout_seconds):
        return 1, "", json.dumps({"nested": json.dumps({"password": escaped})})

    cli = OmniCli(profile="experiments", runner=runner)
    with pytest.raises(OmniCliError, match="response suppressed") as error:
        cli.create_connection(
            {
                "dialect": "postgres",
                "name": "Test",
                "passwordUnencrypted": password,
            }
        )
    assert password not in str(error.value)
    assert escaped not in str(error.value)


def test_cli_validates_profile_and_response_shape() -> None:
    with pytest.raises(OmniCliError, match="profile"):
        OmniCli(profile=" ")

    def malformed(arguments, environment, stdin, timeout_seconds):
        return 0, "[]", ""

    cli = OmniCli(profile="experiments", runner=malformed)
    with pytest.raises(OmniCliError, match="must be an object"):
        cli.create_connection(
            {"dialect": "postgres", "name": "Test", "passwordUnencrypted": "x"}
        )


def test_cli_drops_malformed_secret_bearing_stdout_from_exception_context() -> None:
    password = "<admin>&secret"

    def malformed(arguments, environment, stdin, timeout_seconds):
        return 0, f'{{"nested":"{password}"', ""

    cli = OmniCli(profile="experiments", runner=malformed)
    with pytest.raises(OmniCliError, match="response suppressed") as error:
        cli.create_connection(
            {
                "dialect": "postgres",
                "name": "Test",
                "passwordUnencrypted": password,
            }
        )
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_cli_drops_secret_bearing_timeout_from_exception_context() -> None:
    password = "top-secret"

    def timeout(arguments, environment, stdin, timeout_seconds):
        raise subprocess.TimeoutExpired(
            arguments, timeout_seconds, output=password, stderr=password
        )

    cli = OmniCli(profile="experiments", runner=timeout)
    with pytest.raises(OmniCliError, match="response suppressed") as error:
        cli.create_connection(
            {
                "dialect": "postgres",
                "name": "Test",
                "passwordUnencrypted": password,
            }
        )
    assert password not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_cli_rejects_success_without_connection_id() -> None:
    def no_id(arguments, environment, stdin, timeout_seconds):
        return 0, '{"success":true}', ""

    cli = OmniCli(profile="experiments", runner=no_id)
    with pytest.raises(OmniCliError, match="no id"):
        cli.create_connection(
            {"dialect": "postgres", "name": "Test", "passwordUnencrypted": "x"}
        )

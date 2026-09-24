# Copyright 2026, Aiven, https://aiven.io/
#
# This file is under the Apache License, Version 2.0.
# See the file `LICENSE` for details.
"""Aiven Runtime application commands, exercised through a real AivenClient over a fake HTTP session."""

from __future__ import annotations

from aiven.client import AivenClient
from aiven.client.cli import (
    APPLICATION_WAIT_MAX_CONSECUTIVE_ERRORS,
    APPLICATION_WAIT_MAX_INTERVAL,
    EXIT_CODE_TIMEOUT,
    AivenCLI,
    application_status,
)
from aiven.client.client import RetrySpec
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from pytest import CaptureFixture, LogCaptureFixture
from tests.test_client import MockResponse
from typing import Any

import json
import pytest
import requests

BASE_URL = "https://api.example.invalid"
PROJECT = "test-project"
APP = "example-app"
SERVICE_PATH = f"/project/{PROJECT}/service/{APP}"
SHA = "0123456789abcdef0123456789abcdef01234567"

APPLICATION_USER_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "application": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "object",
                    "properties": {
                        "repository_url": {"type": "string"},
                        "branch": {"type": "string"},
                        "build_path": {"type": "string"},
                        "containerfile_path": {"type": ["string", "null"]},
                        "vcs_integration_id": {"type": "string"},
                        "remote_repository_id": {"type": "string"},
                    },
                },
                "ports": {"type": "array", "items": {"type": "object"}},
                "environment_variables": {"type": "array", "items": {"type": "object"}},
            },
        },
    },
}


@dataclass
class RecordedRequest:
    method: str
    path: str
    params: dict[str, Any] | None
    body: Any


@dataclass
class FakeSession:
    """Stands in for requests.Session: answers by (method, path) and records every request.

    Each route holds a list of responses served in order; the last one repeats.
    """

    routes: dict[tuple[str, str], list[MockResponse | Exception]]
    requests: list[RecordedRequest] = field(default_factory=list)

    def _handle(self, method: str, url: str, **kwargs: Any) -> MockResponse:
        path = url.removeprefix(BASE_URL + "/v1")
        data = kwargs.get("data")
        self.requests.append(
            RecordedRequest(method=method, path=path, params=kwargs.get("params"), body=json.loads(data) if data else None)
        )
        responses = self.routes.get((method, path))
        if not responses:
            return MockResponse(HTTPStatus.NOT_FOUND, {"message": f"no fake route for {method} {path}"})
        response = responses.pop(0) if len(responses) > 1 else responses[0]
        if isinstance(response, Exception):
            raise response
        return response

    def get(self, url: str, **kwargs: Any) -> MockResponse:
        return self._handle("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> MockResponse:
        return self._handle("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> MockResponse:
        return self._handle("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> MockResponse:
        return self._handle("DELETE", url, **kwargs)

    def paths(self, method: str) -> list[str]:
        return [request.path for request in self.requests if request.method == method]


@dataclass
class Harness:
    session: FakeSession
    clock: FakeClock
    config_path: Path

    def run(self, *args: str) -> int | None:
        def client_factory(**kwargs: Any) -> AivenClient:
            # One attempt per request, so connection errors reach the command under test.
            client = AivenClient(**kwargs, default_retry_spec=RetrySpec(attempts=1))
            client.session = self.session  # type: ignore[assignment]
            return client

        cli = AivenCLI(client_factory=client_factory, sleep=self.clock.sleep, monotonic=self.clock.monotonic)
        return cli.run(
            args=["--config", str(self.config_path), "--auth-token", "token", "--url", BASE_URL, *args],
        )


def make_harness(tmp_path: Path, routes: dict[tuple[str, str], list[MockResponse | Exception]]) -> Harness:
    return Harness(session=FakeSession(routes=routes), clock=FakeClock(), config_path=tmp_path / "avn.json")


def ok(body: dict[str, Any]) -> MockResponse:
    return MockResponse(HTTPStatus.OK, body, headers={"content-type": "application/json"})


def app_service(
    state: str,
    build_status: str | None = None,
    deployment_status: str | None = None,
    commit_sha: str | None = None,
    service_type: str = "application",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if build_status is not None:
        metadata["application_build_status"] = build_status
    if deployment_status is not None:
        metadata["application_deployment_status"] = deployment_status
    if commit_sha is not None:
        metadata["application_deployment_commit_sha"] = commit_sha
    return {
        "service_name": APP,
        "service_type": service_type,
        "state": state,
        "plan": "startup-50-1024",
        "metadata": metadata,
        "user_config": {
            "application": {"source": {"repository_url": "https://github.com/example/app.git", "branch": "main"}}
        },
    }


def service_response(*args: Any, **kwargs: Any) -> MockResponse:
    return ok({"service": app_service(*args, **kwargs)})


def create_routes() -> dict[tuple[str, str], list[MockResponse | Exception]]:
    return {
        ("GET", f"/project/{PROJECT}/service-types/application"): [
            ok({"user_config_schema": APPLICATION_USER_CONFIG_SCHEMA})
        ],
        ("GET", f"/project/{PROJECT}/vpcs"): [ok({"vpcs": []})],
        ("POST", f"/project/{PROJECT}/service"): [ok({"service": app_service("REBUILDING")})],
    }


def test_service_create_documented_config_options(tmp_path: Path) -> None:
    """The CLI example of docs/products/runtime/deploy-apps.md in aiven-docs."""
    harness = make_harness(tmp_path, create_routes())

    assert (
        harness.run(
            *("service", "create", APP, "--project", PROJECT, "-t", "application"),
            *("--cloud", "aws-eu-west-1", "--plan", "startup-50-1024"),
            *("-c", "application.source.vcs_integration_id=vcs123", "-c", "application.source.remote_repository_id=r1"),
            *("-c", "application.source.repository_url=https://github.com/example/app.git"),
            *("-c", "application.source.branch=main", "-c", "application.source.build_path=."),
            *("-c", "application.source.containerfile_path=Dockerfile"),
            *("-c", 'application.ports=[{"name":"http","port":8080,"protocol":"HTTP"}]'),
        )
        is None
    )

    create = harness.session.requests[-1]
    assert (create.method, create.path) == ("POST", f"/project/{PROJECT}/service")
    assert create.body["service_type"] == "application"
    assert create.body["user_config"] == {
        "application": {
            "source": {
                "vcs_integration_id": "vcs123",
                "remote_repository_id": "r1",
                "repository_url": "https://github.com/example/app.git",
                "branch": "main",
                "build_path": ".",
                "containerfile_path": "Dockerfile",
            },
            "ports": [{"name": "http", "port": 8080, "protocol": "HTTP"}],
        }
    }


def test_service_create_user_config_json(tmp_path: Path) -> None:
    user_config = {
        "application": {
            "source": {"repository_url": "https://github.com/example/app.git", "branch": "main"},
            "environment_variables": [{"key": "DEBUG", "value": "true", "kind": "variable"}],
        }
    }
    harness = make_harness(tmp_path, create_routes())

    args = ["service", "create", APP, "--project", PROJECT, "-t", "application", "--plan", "startup-50-1024"]
    assert harness.run(*args, "--user-config-json", json.dumps(user_config)) is None

    assert harness.session.paths("GET") == []
    assert harness.session.requests[-1].body["user_config"] == user_config


@pytest.mark.parametrize(
    ("command", "extra_args", "message"),
    [
        (["create", APP, "-t", "application", "--plan", "p"], ["-c", "application.source.branch=main"], "-c (user config)"),
        (["create", APP, "-t", "application", "--plan", "p"], ["--user-config-json", "[]"], "expected a JSON object"),
        (["create", APP, "-t", "application", "--plan", "p"], ["--user-config-json", "null"], "expected a JSON object"),
        (
            ["create", APP, "-t", "application", "--plan", "p"],
            ["--user-config-json", "@/nonexistent/app.json"],
            "Cannot read",
        ),
        (["update", APP], ["--remove-option", "application.source.containerfile_path"], "--remove-option"),
    ],
)
def test_service_user_config_json_conflicts(
    tmp_path: Path, caplog: LogCaptureFixture, command: list[str], extra_args: list[str], message: str
) -> None:
    user_config_json = [] if "--user-config-json" in extra_args else ["--user-config-json", "{}"]
    harness = make_harness(tmp_path, {("GET", SERVICE_PATH): [service_response("RUNNING")]})

    assert harness.run("service", *command, "--project", PROJECT, *user_config_json, *extra_args) == 1
    assert message in caplog.text
    assert harness.session.paths("POST") == harness.session.paths("PUT") == []


def test_service_create_user_config_json_file_not_utf8(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    config_file = tmp_path / "app.json"
    config_file.write_bytes(b"\xff\xfe{}")
    harness = make_harness(tmp_path, create_routes())

    args = ["service", "create", APP, "--project", PROJECT, "-t", "application", "--plan", "startup-50-1024"]
    assert harness.run(*args, "--user-config-json", f"@{config_file}") == 1
    assert "Cannot read user_config_json file" in caplog.text


def test_service_update_user_config_json(tmp_path: Path) -> None:
    harness = make_harness(
        tmp_path,
        {
            ("GET", SERVICE_PATH): [service_response("RUNNING")],
            ("PUT", SERVICE_PATH): [ok({"service": app_service("RUNNING")})],
        },
    )

    user_config = {"application": {"source": {"branch": "release"}}}
    assert harness.run("service", "update", APP, "--project", PROJECT, "--user-config-json", json.dumps(user_config)) is None

    assert harness.session.paths("GET") == [SERVICE_PATH]
    assert harness.session.requests[-1].body["user_config"] == user_config


@pytest.mark.parametrize(
    ("extra_args", "expected_log_type"),
    [(["--log-type", "application-build"], "application-build"), ([], None)],
)
def test_service_logs_log_type(tmp_path: Path, extra_args: list[str], expected_log_type: str | None) -> None:
    logs: dict[str, Any] = {"logs": [], "offset": None, "first_log_offset": None}
    harness = make_harness(tmp_path, {("POST", SERVICE_PATH + "/logs"): [ok(logs)]})

    assert harness.run("service", "logs", "--project", PROJECT, *extra_args, APP) is None
    assert harness.session.requests[0].body.get("log_type") == expected_log_type


@dataclass
class FakeClock:
    now: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


BUILDING = ("REBUILDING", "BUILDING", "IN_PROGRESS")
SUCCEEDED = ("RUNNING", "SUCCESS", "COMPLETED")


@pytest.mark.parametrize(
    ("state", "build_status", "deployment_status", "outcome"),
    [
        ("REBUILDING", None, None, "in_progress"),
        ("REBUILDING", "BUILDING", "IN_PROGRESS", "in_progress"),
        ("RUNNING", "SUCCESS", "COMPLETED", "succeeded"),
        # A failed redeploy keeps the previous revision running.
        ("RUNNING", "SUCCESS", "FAILED", "failed"),
        # A failed first build leaves nothing running.
        ("POWEROFF", "FAILURE", "ABORTED", "failed"),
        ("POWEROFF", "SUCCESS", "COMPLETED", "powered_off"),
        ("POWEROFF", None, None, "powered_off"),
    ],
)
def test_application_status_outcome(
    state: str, build_status: str | None, deployment_status: str | None, outcome: str
) -> None:
    assert application_status(app_service(state, build_status, deployment_status))["outcome"] == outcome


def test_application_status_json(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    harness = make_harness(tmp_path, {("GET", SERVICE_PATH): [service_response(*SUCCEEDED, commit_sha=SHA)]})

    assert harness.run("service", "application", "status", "--project", PROJECT, "--json", APP) is None

    assert json.loads(capsys.readouterr().out) == {
        "service_name": APP,
        "state": "RUNNING",
        "build_status": "SUCCESS",
        "deployment_status": "COMPLETED",
        "deployed_commit_sha": SHA,
        "repository_url": "https://github.com/example/app.git",
        "branch": "main",
        "outcome": "succeeded",
    }


def test_application_status_rejects_other_service_types(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    harness = make_harness(tmp_path, {("GET", SERVICE_PATH): [service_response("RUNNING", service_type="pg")]})

    assert harness.run("service", "application", "status", "--project", PROJECT, APP) == 1
    assert "is not an Aiven Runtime application" in caplog.text


def test_application_wait_succeeds_after_build(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    harness = make_harness(
        tmp_path,
        {("GET", SERVICE_PATH): [service_response("REBUILDING"), service_response(*BUILDING), service_response(*SUCCEEDED)]},
    )

    assert harness.run("service", "application", "wait", "--project", PROJECT, "--json", APP) == 0

    assert harness.clock.sleeps == [5.0, 7.5]
    assert json.loads(capsys.readouterr().out)["outcome"] == "succeeded"


@pytest.mark.parametrize(
    "final",
    [
        ("RUNNING", "SUCCESS", "FAILED"),
        ("POWEROFF", "FAILURE", "ABORTED"),
        ("POWEROFF", "SUCCESS", "COMPLETED"),
    ],
)
def test_application_wait_fails_on_failed_or_powered_off(tmp_path: Path, final: tuple[str, str, str]) -> None:
    harness = make_harness(tmp_path, {("GET", SERVICE_PATH): [service_response(*BUILDING), service_response(*final)]})

    assert harness.run("service", "application", "wait", "--project", PROJECT, APP) == 1


def test_application_wait_backs_off_and_times_out(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    harness = make_harness(tmp_path, {("GET", SERVICE_PATH): [service_response(*BUILDING)]})

    assert harness.run("service", "application", "wait", "--project", PROJECT, "--timeout", "200", APP) == EXIT_CODE_TIMEOUT

    assert harness.clock.sleeps[:6] == [5.0, 7.5, 11.25, 16.875, 25.3125, 30.0]
    assert max(harness.clock.sleeps) == APPLICATION_WAIT_MAX_INTERVAL
    assert sum(harness.clock.sleeps) == pytest.approx(200.0)
    assert len(harness.session.paths("GET")) == len(harness.clock.sleeps) + 1
    assert "Timeout waiting for application" in caplog.text


def test_application_wait_retries_transient_errors(tmp_path: Path) -> None:
    harness = make_harness(
        tmp_path,
        {
            ("GET", SERVICE_PATH): [
                MockResponse(HTTPStatus.SERVICE_UNAVAILABLE, {"message": "unavailable"}),
                MockResponse(HTTPStatus.TOO_MANY_REQUESTS, {"message": "slow down"}),
                service_response(*SUCCEEDED),
            ]
        },
    )

    assert harness.run("service", "application", "wait", "--project", PROJECT, APP) == 0
    assert harness.session.paths("GET") == [SERVICE_PATH] * 3
    # Errors do not reset or skip the backoff.
    assert harness.clock.sleeps == [5.0, 7.5]


def test_application_wait_retries_connection_errors(tmp_path: Path) -> None:
    harness = make_harness(
        tmp_path,
        {("GET", SERVICE_PATH): [requests.exceptions.ConnectionError("reset"), service_response(*SUCCEEDED)]},
    )

    assert harness.run("service", "application", "wait", "--project", PROJECT, APP) == 0
    assert harness.session.paths("GET") == [SERVICE_PATH] * 2
    assert harness.clock.sleeps == [5.0]


def test_application_status_table(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    harness = make_harness(tmp_path, {("GET", SERVICE_PATH): [service_response(*SUCCEEDED, commit_sha=SHA)]})

    assert harness.run("service", "application", "status", "--project", PROJECT, APP) is None

    header, _, row = capsys.readouterr().out.splitlines()[:3]
    assert header.split() == [
        "SERVICE_NAME",
        "STATE",
        "BUILD_STATUS",
        "DEPLOYMENT_STATUS",
        "DEPLOYED_COMMIT_SHA",
        "REPOSITORY_URL",
        "BRANCH",
        "OUTCOME",
    ]
    assert row.split() == [
        APP,
        "RUNNING",
        "SUCCESS",
        "COMPLETED",
        SHA,
        "https://github.com/example/app.git",
        "main",
        "succeeded",
    ]


def test_application_wait_aborts_after_repeated_errors(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    harness = make_harness(
        tmp_path, {("GET", SERVICE_PATH): [MockResponse(HTTPStatus.BAD_GATEWAY, {"message": "bad gateway"})]}
    )

    assert harness.run("service", "application", "wait", "--project", PROJECT, APP) == 1
    assert len(harness.session.paths("GET")) == APPLICATION_WAIT_MAX_CONSECUTIVE_ERRORS
    assert "failed repeatedly" in caplog.text


def test_application_wait_stops_on_client_error(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, {})

    assert harness.run("service", "application", "wait", "--project", PROJECT, APP) == 1
    assert harness.session.paths("GET") == [SERVICE_PATH]
    assert harness.clock.sleeps == []


def test_application_redeploy_without_wait(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    harness = make_harness(
        tmp_path,
        {
            ("POST", SERVICE_PATH + "/application/redeploy"): [MockResponse(HTTPStatus.NO_CONTENT)],
            ("GET", SERVICE_PATH): [service_response(*SUCCEEDED)],
        },
    )

    assert harness.run("service", "application", "redeploy", "--project", PROJECT, APP) is None

    assert [(r.method, r.path, r.body) for r in harness.session.requests] == [
        ("GET", SERVICE_PATH, None),
        ("POST", SERVICE_PATH + "/application/redeploy", None),
    ]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("extra_args", [[], ["--wait"]])
def test_application_redeploy_refuses_while_deployment_in_progress(
    tmp_path: Path, caplog: LogCaptureFixture, extra_args: list[str]
) -> None:
    harness = make_harness(
        tmp_path,
        {
            ("POST", SERVICE_PATH + "/application/redeploy"): [MockResponse(HTTPStatus.NO_CONTENT)],
            ("GET", SERVICE_PATH): [service_response(*BUILDING)],
        },
    )

    assert harness.run("service", "application", "redeploy", "--project", PROJECT, *extra_args, APP) == 1
    assert f"'avn service application wait {APP}'" in caplog.text
    assert harness.session.paths("POST") == []


def test_application_redeploy_wait_ignores_stale_completed_status(tmp_path: Path) -> None:
    harness = make_harness(
        tmp_path,
        {
            ("POST", SERVICE_PATH + "/application/redeploy"): [MockResponse(HTTPStatus.NO_CONTENT)],
            ("GET", SERVICE_PATH): [
                service_response(*SUCCEEDED, commit_sha=SHA),
                service_response(*SUCCEEDED, commit_sha=SHA),
                service_response(*BUILDING),
                service_response("RUNNING", "SUCCESS", "FAILED"),
            ],
        },
    )

    assert harness.run("service", "application", "redeploy", "--project", PROJECT, "--wait", APP) == 1
    assert harness.session.paths("GET") == [SERVICE_PATH] * 4


def test_application_redeploy_wait_succeeds(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    harness = make_harness(
        tmp_path,
        {
            ("POST", SERVICE_PATH + "/application/redeploy"): [MockResponse(HTTPStatus.NO_CONTENT)],
            ("GET", SERVICE_PATH): [
                service_response(*SUCCEEDED),
                service_response(*SUCCEEDED),
                service_response(*BUILDING),
                service_response(*SUCCEEDED, commit_sha=SHA),
            ],
        },
    )

    assert harness.run("service", "application", "redeploy", "--project", PROJECT, "--wait", "--json", APP) == 0
    assert json.loads(capsys.readouterr().out)["deployed_commit_sha"] == SHA


def test_application_redeploy_wait_detects_failure_between_polls(tmp_path: Path) -> None:
    # The build started and failed between two polls, so no in-progress read was ever seen.
    harness = make_harness(
        tmp_path,
        {
            ("POST", SERVICE_PATH + "/application/redeploy"): [MockResponse(HTTPStatus.NO_CONTENT)],
            ("GET", SERVICE_PATH): [
                service_response(*SUCCEEDED),
                service_response(*SUCCEEDED),
                service_response("RUNNING", "FAILURE", "ABORTED"),
            ],
        },
    )

    assert harness.run("service", "application", "redeploy", "--project", PROJECT, "--wait", APP) == 1
    assert harness.session.paths("GET") == [SERVICE_PATH] * 3


def test_application_redeploy_wait_times_out_without_new_deployment(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    harness = make_harness(
        tmp_path,
        {
            ("POST", SERVICE_PATH + "/application/redeploy"): [MockResponse(HTTPStatus.NO_CONTENT)],
            ("GET", SERVICE_PATH): [service_response(*SUCCEEDED)],
        },
    )

    assert (
        harness.run("service", "application", "redeploy", "--project", PROJECT, "--wait", "--timeout", "60", APP)
        == EXIT_CODE_TIMEOUT
    )
    assert "no new deployment" in caplog.text

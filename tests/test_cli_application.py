# Copyright 2026, Aiven, https://aiven.io/
#
# This file is under the Apache License, Version 2.0.
# See the file `LICENSE` for details.
"""Aiven Runtime application commands, exercised through a real AivenClient over a fake HTTP session."""

from __future__ import annotations

from aiven.client import AivenClient
from aiven.client.cli import (
    AivenCLI,
)
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from pytest import LogCaptureFixture
from tests.test_client import MockResponse
from typing import Any

import json
import pytest

BASE_URL = "https://api.example.invalid"
PROJECT = "test-project"
APP = "example-app"
SERVICE_PATH = f"/project/{PROJECT}/service/{APP}"

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

    routes: dict[tuple[str, str], list[MockResponse]]
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
        return responses.pop(0) if len(responses) > 1 else responses[0]

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
    config_path: Path

    def run(self, *args: str) -> int | None:
        def client_factory(**kwargs: Any) -> AivenClient:
            client = AivenClient(**kwargs)
            client.session = self.session  # type: ignore[assignment]
            return client

        cli = AivenCLI(client_factory=client_factory)
        return cli.run(
            args=["--config", str(self.config_path), "--auth-token", "token", "--url", BASE_URL, *args],
        )


def make_harness(tmp_path: Path, routes: dict[tuple[str, str], list[MockResponse]]) -> Harness:
    return Harness(session=FakeSession(routes=routes), config_path=tmp_path / "avn.json")


def ok(body: dict[str, Any]) -> MockResponse:
    return MockResponse(HTTPStatus.OK, body, headers={"content-type": "application/json"})


def app_service(
    state: str,
    build_status: str | None = None,
    deployment_status: str | None = None,
    service_type: str = "application",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if build_status is not None:
        metadata["application_build_status"] = build_status
    if deployment_status is not None:
        metadata["application_deployment_status"] = deployment_status
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


def create_routes() -> dict[tuple[str, str], list[MockResponse]]:
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

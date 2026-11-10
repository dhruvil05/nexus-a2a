"""
tests/test_v2_0_security_client.py

Client-side fixes from the 2.0 security audit (SECURITY_AUDIT.md):

  #12  routing uses the registration URL, not the remote agent's card.url
  #17  Redis URLs and storage.url are redacted in logs / to_dict()
  #19  bounded reads from remote agents (SSE line, response body)
  #21  Rich markup / control characters from remote agents are escaped
  #23  the admin token is not sent to an insecure URL taken from nexus.toml
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import random
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from click.testing import CliRunner
from rich.console import Console

from nexus_a2a.cli import admin, output
from nexus_a2a.cli.main import cli
from nexus_a2a.config import NexusConfig
from nexus_a2a.core.registry import AgentRegistry
from nexus_a2a.models.agent import AgentCard, AgentSkill
from nexus_a2a.models.task import Message
from nexus_a2a.network import AgentNetwork
from nexus_a2a.storage.dlq_store import RedisDLQStore
from nexus_a2a.storage.push_store import RedisPushStore
from nexus_a2a.storage.redis_store import RedisTaskStore
from nexus_a2a.transport.http_client import (
    A2AHttpClient,
    AgentUnreachableError,
    ResponseTooLargeError,
)
from nexus_a2a.transport.sse import (
    SSEStreamer,
    StreamEventType,
    _aiter_bounded_lines,
    iter_sse_events,
)

AGENT = "http://agent.test:8001"


def make_card(
    url: str = AGENT, skill: str = "search", name: str = "TestAgent"
) -> AgentCard:
    return AgentCard(
        name=name,
        description="test",
        url=url,
        skills=[AgentSkill(id=skill, name=skill.title(), description="test skill")],
    )


def patch_fetch(card: AgentCard) -> Any:
    client = AsyncMock()
    client.fetch_agent_card = AsyncMock(return_value=card)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return patch("nexus_a2a.core.registry.A2AHttpClient", return_value=client)


# ── #12 routing ───────────────────────────────────────────────────────────────


class TestRoutingUsesRegistrationUrl:
    async def test_foreign_card_url_is_ignored_for_routing(self, caplog):
        registry = AgentRegistry()
        evil = make_card(url="http://169.254.169.254/latest")
        with patch_fetch(evil), caplog.at_level(logging.WARNING):
            await registry.register_url("http://user:pw@agent.test:8001/")

        assert registry.find_urls_by_skill("search") == [
            "http://user:pw@agent.test:8001"
        ]
        assert registry.list_healthy_urls() == ["http://user:pw@agent.test:8001"]
        warning = next(
            r for r in caplog.records if "different origin" in r.getMessage()
        )
        assert "169.254.169.254" in warning.getMessage()
        assert "pw" not in warning.getMessage()

    async def test_scheme_downgrade_is_a_different_origin(self):
        registry = AgentRegistry()
        with patch_fetch(make_card(url="http://secure.test")):
            await registry.register_url("https://secure.test")
        assert registry.list_healthy_urls() == ["https://secure.test"]

    async def test_matching_card_url_routes_as_before(self, caplog):
        registry = AgentRegistry()
        with patch_fetch(make_card(url=AGENT)), caplog.at_level(logging.WARNING):
            await registry.register_url(AGENT)
        assert registry.find_urls_by_skill("search") == [AGENT]
        assert not [r for r in caplog.records if "different origin" in r.getMessage()]

    async def test_same_origin_path_from_card_is_kept(self):
        registry = AgentRegistry()
        with patch_fetch(make_card(url="http://agent.test:8001/a2a")):
            await registry.register_url("http://AGENT.test:8001")
        assert registry.list_healthy_urls() == ["http://agent.test:8001/a2a"]

    async def test_default_port_counts_as_same_origin(self):
        registry = AgentRegistry()
        with patch_fetch(make_card(url="https://agent.test:443/rpc")):
            await registry.register_url("https://agent.test")
        assert registry.list_healthy_urls() == ["https://agent.test/rpc"]

    async def test_network_sends_to_registered_url(self):
        network = AgentNetwork()
        with patch_fetch(make_card(url="http://attacker.test")):
            await network.add(AGENT)
        assert network._resolve_agent("search") == AGENT
        assert network._resolve_agent(None) == AGENT

    async def test_register_card_routes_to_card_url(self):
        network = AgentNetwork()
        await network.registry.register_card(make_card(url="http://agent:8001"))
        assert network._resolve_agent("search") == "http://agent:8001"

    async def test_summary_shows_both_urls(self):
        registry = AgentRegistry()
        with patch_fetch(make_card(url="http://attacker.test")):
            await registry.register_url(AGENT)
        agent_info = registry.summary()["agents"][0]  # type: ignore[index]
        assert agent_info["url"] == "http://attacker.test/"
        assert agent_info["registered_url"] == AGENT


# ── #19 bounded SSE lines ─────────────────────────────────────────────────────


def chunked_response(chunks: list[bytes], **kwargs: Any) -> httpx.Response:
    async def gen() -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk

    return httpx.Response(200, content=gen(), **kwargs)


def split_randomly(data: bytes, rng: random.Random) -> list[bytes]:
    chunks, i = [], 0
    while i < len(data):
        n = rng.randint(1, 7)
        chunks.append(data[i : i + n])
        i += n
    return chunks


class TestBoundedSSELines:
    SAMPLE = (
        'data: {"type": "task_status", "state": "working"}\r\n\r\n'
        ": heartbeat\n\n"
        'data: {"type": "artifact_chunk", "content": "a\\u2028b"}\r\r'
        "trailing\x0bvertical\x1cfs ls ps\x85nel\n"
        "last line without newline\r"
    ).encode()

    async def test_lines_match_httpx_for_any_chunking(self):
        rng = random.Random(1234)
        for _ in range(200):
            chunks = split_randomly(self.SAMPLE, rng)
            expected = [line async for line in chunked_response(chunks).aiter_lines()]
            got = [
                line
                async for line in _aiter_bounded_lines(chunked_response(chunks), 10_000)
            ]
            assert got == expected

    async def test_events_are_parsed_as_before(self):
        body = (
            b'data: {"type": "task_created", "id": "t"}\n\n'
            b": heartbeat\n\n"
            b'data: {"type": "artifact_chunk", "content": "hi"}\n\n'
            b'data: {"type": "done"}\n\n'
        )
        events = [e async for e in iter_sse_events(chunked_response([body]))]
        assert [e.type for e in events] == [
            StreamEventType.TASK_CREATED,
            StreamEventType.HEARTBEAT,
            StreamEventType.ARTIFACT_CHUNK,
            StreamEventType.DONE,
        ]

    async def test_endless_line_is_rejected(self):
        chunks = [b"data: "] + [b"x" * 1024] * 200  # never terminated
        with pytest.raises(ResponseTooLargeError, match="SSE line"):
            async for _ in iter_sse_events(
                chunked_response(chunks), max_line=64 * 1024
            ):
                pass

    async def test_long_complete_line_is_rejected(self):
        body = b"data: " + b"x" * 5000 + b"\n\n"
        with pytest.raises(AgentUnreachableError):
            async for _ in iter_sse_events(chunked_response([body]), max_line=1000):
                pass

    async def test_limit_can_be_disabled(self):
        payload = json.dumps({"type": "message", "text": "x" * 5000})
        body = f"data: {payload}\n\n".encode()
        events = [
            e async for e in iter_sse_events(chunked_response([body]), max_line=None)
        ]
        assert events[0].data["text"] == "x" * 5000

    def test_default_limit_is_one_mib(self):
        assert SSEStreamer("http://a")._max_line == 1024 * 1024
        assert SSEStreamer("http://a", max_line=10)._max_line == 10


# ── #19 bounded response bodies ───────────────────────────────────────────────


def card_json() -> dict[str, Any]:
    return make_card().to_well_known_dict()


class TestBoundedResponseBody:
    async def test_normal_card_fetch_still_works(self):
        with respx.mock() as router:
            router.get(f"{AGENT}/.well-known/agent-card.json").respond(
                200, json=card_json()
            )
            async with A2AHttpClient(AGENT) as client:
                card = await client.fetch_agent_card()
        assert card.name == "TestAgent"

    async def test_declared_length_over_limit_is_rejected(self):
        with respx.mock() as router:
            router.get(f"{AGENT}/.well-known/agent-card.json").respond(
                200, content=b"x" * 5000
            )
            async with A2AHttpClient(AGENT, max_response_bytes=1000) as client:
                with pytest.raises(ResponseTooLargeError, match="1000-byte limit"):
                    await client.fetch_agent_card()

    async def test_chunked_body_over_limit_is_rejected(self):
        async def endless() -> AsyncIterator[bytes]:
            for _ in range(100):
                yield b"x" * 1000

        with respx.mock() as router:
            router.post(f"{AGENT}/").mock(
                return_value=httpx.Response(200, content=endless())
            )
            async with A2AHttpClient(AGENT, max_response_bytes=10_000) as client:
                with pytest.raises(ResponseTooLargeError):
                    await client._rpc("tasks/get", {})

    async def test_decompressed_size_is_what_counts(self):
        bomb = gzip.compress(b'{"result": "' + b"x" * 200_000 + b'"}')
        assert len(bomb) < 2000
        with respx.mock() as router:
            router.post(f"{AGENT}/").respond(
                200, content=bomb, headers={"Content-Encoding": "gzip"}
            )
            async with A2AHttpClient(AGENT, max_response_bytes=10_000) as client:
                with pytest.raises(ResponseTooLargeError):
                    await client._rpc("tasks/get", {})

    async def test_rpc_under_limit_works(self):
        with respx.mock() as router:
            router.post(f"{AGENT}/").respond(
                200, json={"jsonrpc": "2.0", "id": "1", "result": {"ok": True}}
            )
            async with A2AHttpClient(AGENT, max_response_bytes=1000) as client:
                assert await client._rpc("tasks/get", {}) == {"ok": True}

    async def test_too_large_is_an_unreachable_error_for_health_checks(self):
        registry = AgentRegistry()
        await registry.register_card(make_card())
        with respx.mock() as router:
            router.get(f"{AGENT}/.well-known/agent-card.json").respond(
                200, content=b"x" * (11 * 1024 * 1024)
            )
            assert await registry.check_health(AGENT) is False

    async def test_stream_is_not_capped_by_body_limit(self):
        chunk = 'data: {"type": "artifact_chunk", "content": "' + "x" * 500 + '"}\n\n'
        body = (chunk * 50 + 'data: {"type": "done"}\n\n').encode()
        with respx.mock() as router:
            router.post(f"{AGENT}/").respond(
                200, content=body, headers={"Content-Type": "text/event-stream"}
            )
            async with A2AHttpClient(AGENT, max_response_bytes=1000) as client:
                events = [
                    e async for e in client.stream_message(Message.user_text("hi"))
                ]
        assert len(events) == 51

    async def test_stream_line_limit(self):
        body = ('data: {"type": "message", "t": "' + "x" * 5000 + '"}\n\n').encode()
        with respx.mock() as router:
            router.post(f"{AGENT}/").respond(200, content=body)
            async with A2AHttpClient(AGENT, max_sse_line=1000) as client:
                with pytest.raises(ResponseTooLargeError):
                    async for _ in client.stream_message(Message.user_text("hi")):
                        pass

    async def test_stream_error_body_is_previewed_not_buffered(self):
        async def huge() -> AsyncIterator[bytes]:
            yield b"boom! "
            for _ in range(10_000):
                yield b"y" * 1024

        with respx.mock() as router:
            router.post(f"{AGENT}/").mock(
                return_value=httpx.Response(500, content=huge())
            )
            async with A2AHttpClient(AGENT) as client:
                with pytest.raises(
                    AgentUnreachableError, match="HTTP 500 opening stream: boom!"
                ) as exc:
                    async for _ in client.stream_message(Message.user_text("hi")):
                        pass
        assert len(str(exc.value)) < 400


# ── #21 CLI markup injection ──────────────────────────────────────────────────


@pytest.fixture
def captured(monkeypatch) -> io.StringIO:
    buf = io.StringIO()
    console = Console(file=buf, width=200, force_terminal=False, color_system=None)
    monkeypatch.setattr(output, "console", console)
    monkeypatch.setattr(output, "err_console", console)
    return buf


class TestCliEscaping:
    HOSTILE = (
        "[bold green]PASSED[/bold green] [link=http://evil.test]x[/link]\x1b[2J\x07"
    )

    def test_safe_escapes_markup_and_strips_controls(self):
        assert output.safe("a\x1b[31mb\rc\x9b\x7f\td\n") == "a[31mbc\td\n"
        assert output.safe("[bold]x") == "\\[bold]x"
        assert output.safe("plain text") == "plain text"

    def test_inspect_shows_hostile_card_literally(self, captured):
        card = {
            "name": self.HOSTILE,
            "description": self.HOSTILE,
            "version": "[red]1",
            "skills": [{"id": self.HOSTILE, "name": "[b]n", "tags": ["[i]t"]}],
            "authentication": {"scheme": "[u]api_key", "token_url": self.HOSTILE},
        }
        output.render_inspect(card)
        text = captured.getvalue()
        assert "[bold green]PASSED[/bold green]" in text
        assert "[link=http://evil.test]" in text
        assert "[red]1" in text and "[b]n" in text and "[u]api_key" in text
        assert "\x1b" not in text and "\x07" not in text

    def test_normal_strings_render_unchanged(self, captured):
        output.render_inspect(
            {
                "name": "Tested",
                "description": "An agent.",
                "version": "2.1.0",
                "skills": [{"id": "s", "name": "S", "tags": ["list[str]"]}],
            }
        )
        text = captured.getvalue()
        assert "Tested" in text and "An agent." in text and "v2.1.0" in text
        assert "list[str]" in text

    def test_status_ping_trace_replay_verify(self, captured):
        output.render_status(
            [
                {
                    "name": self.HOSTILE,
                    "url": "[x]u",
                    "healthy": True,
                    "last_seen": "[y]",
                }
            ],
            {"total": 1, "healthy": 1},
        )
        output.render_ping({"url": "[t]", "name": self.HOSTILE, "version": "[v]"})
        output.render_trace(
            {
                "trace_id": "[tid]",
                "hops": [
                    {
                        "url": "\x1b]8;;http://evil\x1b\\u",
                        "duration_ms": 900,
                        "status": "failed",
                        "error": "[e]bad\x1b[0m",
                    }
                ],
            }
        )
        output.render_replay_preview(
            [
                {
                    "task_id": "[task]",
                    "skill_id": "[sk]",
                    "failed_at": "now",
                    "attempts": 1,
                    "error": self.HOSTILE,
                }
            ]
        )
        output.render_verify(
            {
                "url": "[url]",
                "passed": False,
                "checks": [
                    {
                        "status": "pass",
                        "group": "[g]",
                        "name": "card.name",
                        "detail": self.HOSTILE,
                    }
                ],
                "summary": {"pass": 1, "warn": 0, "fail": 0, "skip": 0},
            }
        )
        output.print_error("[network] agents = [...]")
        text = captured.getvalue()
        for literal in (
            "[x]u",
            "[y]",
            "[t]",
            "[v]",
            "[tid]",
            "[e]bad",
            "[task]",
            "[sk]",
            "[url]",
            "[g]",
            "[network] agents",
        ):
            assert literal in text, literal
        assert "[bold green]PASSED" in text
        assert "\x1b" not in text

    def test_replay_verbose_error_is_escaped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("NEXUS_ADMIN_TOKEN", raising=False)
        ops = "http://ops.test:8080"
        entries = {
            "entries": [
                {"task_id": "t1", "failed_at": 1, "retry_count": 0, "error": "x"}
            ]
        }
        with respx.mock() as router:
            router.get(f"{ops}/dlq").respond(200, json=entries)
            router.post(f"{ops}/dlq/replay").respond(
                200,
                json={"succeeded": 0, "results": [{"error": "[bold green]OK\x1b[2J"}]},
            )
            result = CliRunner().invoke(
                cli, ["--verbose", "replay", "--failed", "--yes", "--agent", ops]
            )
        assert "[bold green]OK" in result.output
        assert "\x1b[2J" not in result.output


# ── #23 admin token destination ───────────────────────────────────────────────


class TestAdminTokenDestination:
    @pytest.fixture(autouse=True)
    def _clean_env(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for var in ("NEXUS_ADMIN_TOKEN", "NEXUS_OPS_URL"):
            monkeypatch.delenv(var, raising=False)

    def test_refuses_env_token_to_plain_http_url_from_file(self, monkeypatch):
        monkeypatch.setenv("NEXUS_ADMIN_TOKEN", "secret")
        cfg = {"ops": {"url": "http://evil.test:8080"}}
        with pytest.raises(admin.UnsafeTokenDestinationError, match="--agent"):
            admin.check_token_destination(None, None, cfg)
        with pytest.raises(admin.UnsafeTokenDestinationError):
            admin.check_token_destination(
                None, "flag-secret", {"agent": {"url": "http://x"}}
            )

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://[::1]:8080",
            "http://ops.localhost",
            "https://localhost:8443",
        ],
    )
    def test_allows_loopback(self, monkeypatch, url):
        monkeypatch.setenv("NEXUS_ADMIN_TOKEN", "secret")
        admin.check_token_destination(None, None, {"ops": {"url": url}})

    def test_refuses_https_url_from_file(self, monkeypatch):
        """A hostile nexus.toml can name an https host it controls."""
        monkeypatch.setenv("NEXUS_ADMIN_TOKEN", "secret")
        with pytest.raises(admin.UnsafeTokenDestinationError):
            admin.check_token_destination(
                None, None, {"ops": {"url": "https://attacker.example"}}
            )

    def test_allows_explicit_url(self, monkeypatch):
        cfg = {"ops": {"url": "http://evil.test"}}
        admin.check_token_destination("http://ops.test", "secret", cfg)
        monkeypatch.setenv("NEXUS_OPS_URL", "http://ops.test")
        admin.check_token_destination(None, "secret", cfg)

    def test_allows_token_from_same_file(self):
        cfg = {"ops": {"url": "http://ops.test", "admin_token": "file-secret"}}
        admin.check_token_destination(None, None, cfg)

    def test_no_token_is_fine(self):
        admin.check_token_destination(None, None, {"ops": {"url": "http://ops.test"}})

    def test_trace_command_refuses_without_sending(self, tmp_path):
        (tmp_path / "nexus.toml").write_text('[ops]\nurl = "http://evil.test:8080"\n')
        with respx.mock(assert_all_called=False) as router:
            route = router.get("http://evil.test:8080/traces/tr-1").respond(
                200, json={}
            )
            result = CliRunner().invoke(
                cli, ["trace", "tr-1"], env={"NEXUS_ADMIN_TOKEN": "secret"}
            )
        assert result.exit_code == 1
        assert not route.called
        assert "--agent" in result.output

    def test_replay_command_refuses_without_sending(self, tmp_path):
        (tmp_path / "nexus.toml").write_text('[agent]\nurl = "http://evil.test:8001"\n')
        with respx.mock(assert_all_called=False) as router:
            route = router.get("http://evil.test:8001/dlq").respond(
                200, json={"entries": []}
            )
            result = CliRunner().invoke(
                cli, ["replay", "--failed", "--admin-token", "secret"]
            )
        assert result.exit_code == 1
        assert not route.called

    def test_trace_command_with_explicit_agent_sends_token(self, tmp_path):
        (tmp_path / "nexus.toml").write_text('[ops]\nurl = "http://evil.test:8080"\n')
        trace = {"trace_id": "tr-1", "hops": []}
        with respx.mock() as router:
            route = router.get("http://ops.test:8080/traces/tr-1").respond(
                200, json=trace
            )
            CliRunner().invoke(
                cli,
                ["trace", "tr-1", "--agent", "http://ops.test:8080"],
                env={"NEXUS_ADMIN_TOKEN": "secret"},
            )
        assert route.calls.last.request.headers["X-Admin-Token"] == "secret"


# ── #17 secrets in logs ───────────────────────────────────────────────────────


class TestRedaction:
    def test_to_dict_masks_storage_password(self):
        cfg = NexusConfig()
        cfg.storage.url = (
            "redis://:hunter2@cache.internal:6380/2?password=also&ssl=true"
        )
        url = cfg.to_dict()["storage"]["url"]
        assert "hunter2" not in url and "also" not in url
        assert "cache.internal:6380/2" in url and "ssl=true" in url

    def test_to_dict_leaves_plain_urls_alone(self):
        cfg = NexusConfig()
        cfg.storage.url = "redis://localhost:6379"
        assert cfg.to_dict()["storage"]["url"] == "redis://localhost:6379"
        cfg.storage.url = ""
        assert cfg.to_dict()["storage"]["url"] == ""

    @pytest.mark.parametrize(
        "store_cls", [RedisTaskStore, RedisDLQStore, RedisPushStore]
    )
    async def test_store_connect_log_is_redacted(self, store_cls, caplog):
        fake = AsyncMock()
        with (
            patch("redis.asyncio.from_url", return_value=fake),
            caplog.at_level(logging.INFO),
        ):
            store = store_cls(url="redis://:hunter2@cache.internal:6379")
            await store.connect()
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "cache.internal" in messages
        assert "hunter2" not in messages

"""
tests/test_v2_0_security.py

Regression tests for the 2.0 security fixes:

  1. Webhook SSRF: delivery re-resolves and pins to the vetted address, so a
     DNS-rebinding host cannot reach internal addresses after registration.
  2. Rate limiting is keyed on a verified identity or the client address,
     never on the self-asserted X-Nexus-Caller header.
  3. Trust rules only apply to callers whose credentials prove their URL, and
     wildcards cannot be satisfied by a look-alike domain.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import jwt
import pytest
import respx

from nexus_a2a.models.agent import AuthScheme
from nexus_a2a.models.task import Message, Task
from nexus_a2a.security.auth import AgentCredentialConfig, AuthManager
from nexus_a2a.security.middleware import CALLER_HEADER, SecurityMiddleware
from nexus_a2a.security.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimitError,
)
from nexus_a2a.security.trust import (
    AgentNotAllowedError,
    TrustBoundary,
    UnverifiedCallerError,
)
from nexus_a2a.transport.webhook import (
    WebhookConfig,
    WebhookDeliveryError,
    WebhookDispatcher,
    WebhookUrlError,
    resolve_webhook_url,
    validate_webhook_url,
)

PUBLIC_IP = "93.184.216.34"
SERVER_URL = "http://me:8000"
PEER = "http://peer:1"


def _task() -> Task:
    return Task.create(initial_message=Message.user_text("hi"))


def _dns(monkeypatch: pytest.MonkeyPatch, *answers: list[str]) -> list[str]:
    """Patch the resolver to return each answer in turn (last one repeats)."""
    seen: list[str] = []
    queue = list(answers)

    async def fake(host: str) -> list[str]:
        seen.append(host)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr("nexus_a2a.transport.webhook._resolve_host", fake)
    return seen


# ── 1. Webhook SSRF ───────────────────────────────────────────────────────────


class TestWebhookAddressPolicy:
    @pytest.mark.parametrize(
        "addr",
        [
            "127.0.0.1",
            "10.0.0.5",
            "169.254.169.254",
            "100.100.100.200",  # Alibaba metadata, CGNAT — missed before
            "100.64.0.1",
            "::1",
            "::ffff:127.0.0.1",
            "fd00::1",
            "0.0.0.0",
            "224.0.0.1",
        ],
    )
    async def test_non_public_addresses_refused(self, monkeypatch, addr):
        _dns(monkeypatch, [addr])
        with pytest.raises(WebhookUrlError, match="non-public"):
            await resolve_webhook_url("https://hook.example.com/x")

    async def test_public_address_returned(self, monkeypatch):
        _dns(monkeypatch, [PUBLIC_IP])
        assert await resolve_webhook_url("https://hook.example.com/x") == PUBLIC_IP

    async def test_one_private_record_in_a_round_robin_is_refused(self, monkeypatch):
        _dns(monkeypatch, [PUBLIC_IP, "10.0.0.1"])
        with pytest.raises(WebhookUrlError):
            await resolve_webhook_url("https://hook.example.com/x")

    async def test_slow_resolver_times_out(self, monkeypatch):
        async def slow(host: str) -> list[str]:
            await asyncio.sleep(5)
            return [PUBLIC_IP]

        monkeypatch.setattr("nexus_a2a.transport.webhook._resolve_host", slow)
        with pytest.raises(WebhookUrlError, match="timed out"):
            await resolve_webhook_url("https://hook.example.com/x", timeout=0.05)

    @pytest.mark.parametrize(
        "url", ["https://hook.example.com/\r\nX-Injected: 1", "https://a b.com/"]
    )
    def test_control_characters_refused(self, url):
        with pytest.raises(WebhookUrlError, match="control"):
            validate_webhook_url(url, allow_private=True)

    def test_sync_validator_blocks_cgnat_literal(self):
        with pytest.raises(WebhookUrlError):
            validate_webhook_url("http://100.100.100.200/latest/meta-data/")

    async def test_allow_private_skips_pinning(self):
        assert (
            await resolve_webhook_url("http://127.0.0.1:9/x", allow_private=True)
            is None
        )


class TestWebhookDeliveryPinning:
    async def test_dns_rebinding_after_registration_is_refused(self, monkeypatch):
        """Public at registration, loopback at delivery: nothing is sent."""
        _dns(monkeypatch, [PUBLIC_IP], ["127.0.0.1"])
        url = "http://rebind.attacker.example/hook"
        await resolve_webhook_url(url)  # registration passes

        dispatcher = WebhookDispatcher(WebhookConfig(max_retries=3, base_delay=0))
        with respx.mock(assert_all_called=False) as router:
            route = router.route().mock(return_value=httpx.Response(200))
            with pytest.raises(WebhookDeliveryError, match="non-public"):
                await dispatcher.dispatch(url, _task())
        assert route.call_count == 0

    async def test_delivery_connects_to_vetted_ip_with_original_host(self, monkeypatch):
        _dns(monkeypatch, [PUBLIC_IP])
        dispatcher = WebhookDispatcher(WebhookConfig(max_retries=1))
        with respx.mock() as router:
            route = router.post(f"https://{PUBLIC_IP}:8443/hook").mock(
                return_value=httpx.Response(200)
            )
            record = await dispatcher.dispatch(
                "https://hook.example.com:8443/hook", _task()
            )
        assert record.succeeded
        sent = route.calls.last.request
        assert sent.headers["host"] == "hook.example.com:8443"
        # TLS still authenticates the name the caller registered.
        assert sent.extensions["sni_hostname"] == "hook.example.com"

    async def test_every_retry_re_resolves(self, monkeypatch):
        seen = _dns(monkeypatch, [PUBLIC_IP])
        dispatcher = WebhookDispatcher(WebhookConfig(max_retries=3, base_delay=0))
        with respx.mock() as router:
            router.post(f"http://{PUBLIC_IP}/hook").mock(
                return_value=httpx.Response(503)
            )
            with pytest.raises(WebhookDeliveryError):
                await dispatcher.dispatch("http://hook.example.com/hook", _task())
        assert seen == ["hook.example.com"] * 3


# ── 2/3. Identity: rate limiting and trust ────────────────────────────────────

SECRET = "a-shared-jwt-secret-that-is-32-bytes!"


def _token(sub: str, secret: str = SECRET) -> str:
    now = int(time.time())
    return jwt.encode({"sub": sub, "iat": now, "exp": now + 60}, secret, "HS256")


def _jwt_headers(caller: str, sub: str | None = None) -> dict[str, str]:
    return {
        CALLER_HEADER: caller,
        "Authorization": f"Bearer {_token(sub or caller)}",
    }


def _shared_jwt_auth() -> AuthManager:
    return AuthManager(
        default=AgentCredentialConfig(scheme=AuthScheme.JWT, jwt_secret=SECRET)
    )


class TestIdentityVerification:
    async def test_no_auth_never_verifies(self):
        identity = await SecurityMiddleware().authorize({CALLER_HEADER: PEER})
        assert identity.verified is False

    async def test_registered_credential_verifies(self):
        auth = AuthManager()
        auth.register_agent(
            PEER, AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="k")
        )
        mw = SecurityMiddleware(auth=auth)
        identity = await mw.authorize({CALLER_HEADER: PEER, "X-API-Key": "k"})
        assert identity.verified is True

    async def test_shared_api_key_does_not_verify(self):
        auth = AuthManager(
            default=AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="k")
        )
        identity = await SecurityMiddleware(auth=auth).authorize(
            {CALLER_HEADER: PEER, "X-API-Key": "k"}
        )
        assert identity.verified is False

    async def test_jwt_sub_matching_header_verifies(self):
        mw = SecurityMiddleware(auth=_shared_jwt_auth())
        assert (await mw.authorize(_jwt_headers(PEER))).verified is True

    async def test_jwt_sub_not_matching_header_does_not_verify(self):
        mw = SecurityMiddleware(auth=_shared_jwt_auth())
        identity = await mw.authorize(_jwt_headers(PEER, sub="http://someone-else:1"))
        assert identity.verified is False


class TestRateLimitKeying:
    def _mw(self, **kwargs) -> SecurityMiddleware:
        return SecurityMiddleware(
            rate_limiter=RateLimiter(RateLimitConfig(rate=0.001, burst=1)), **kwargs
        )

    async def test_header_rotation_shares_the_address_bucket(self):
        mw = self._mw()
        await mw.authorize({CALLER_HEADER: "http://a:1"}, client_host="1.2.3.4")
        with pytest.raises(RateLimitError):
            await mw.authorize({CALLER_HEADER: "http://b:1"}, client_host="1.2.3.4")

    async def test_different_addresses_have_different_buckets(self):
        mw = self._mw()
        await mw.authorize({}, client_host="1.2.3.4")
        await mw.authorize({}, client_host="5.6.7.8")

    async def test_verified_callers_share_an_address_but_not_a_bucket(self):
        mw = self._mw(auth=_shared_jwt_auth())
        await mw.authorize(_jwt_headers("http://a:1"), client_host="10.0.0.1")
        await mw.authorize(_jwt_headers("http://b:1"), client_host="10.0.0.1")
        with pytest.raises(RateLimitError):
            await mw.authorize(_jwt_headers("http://a:1"), client_host="10.0.0.1")

    async def test_proxy_header_uses_last_hop_only(self):
        mw = self._mw(client_ip_header="X-Forwarded-For")
        await mw.authorize(
            {"X-Forwarded-For": "6.6.6.6, 1.2.3.4"}, client_host="10.0.0.1"
        )
        # Forging the client-supplied (first) entry does not buy a new bucket.
        with pytest.raises(RateLimitError):
            await mw.authorize(
                {"X-Forwarded-For": "7.7.7.7, 1.2.3.4"}, client_host="10.0.0.1"
            )


class TestTrustNeedsVerifiedIdentity:
    def _trust(self) -> TrustBoundary:
        trust = TrustBoundary()
        trust.allow(PEER, SERVER_URL)
        return trust

    async def test_claimed_header_alone_is_refused(self):
        mw = SecurityMiddleware(trust=self._trust(), server_url=SERVER_URL)
        with pytest.raises(UnverifiedCallerError):
            await mw.authorize({CALLER_HEADER: PEER})

    async def test_shared_api_key_cannot_satisfy_trust(self):
        auth = AuthManager(
            default=AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="k")
        )
        mw = SecurityMiddleware(auth=auth, trust=self._trust(), server_url=SERVER_URL)
        with pytest.raises(UnverifiedCallerError):
            await mw.authorize({CALLER_HEADER: PEER, "X-API-Key": "k"})

    async def test_jwt_bound_caller_is_allowed(self):
        mw = SecurityMiddleware(
            auth=_shared_jwt_auth(), trust=self._trust(), server_url=SERVER_URL
        )
        await mw.authorize(_jwt_headers(PEER))

    async def test_borrowing_an_allowed_url_is_refused(self):
        mw = SecurityMiddleware(
            auth=_shared_jwt_auth(), trust=self._trust(), server_url=SERVER_URL
        )
        with pytest.raises(UnverifiedCallerError):
            await mw.authorize(_jwt_headers(PEER, sub="http://intruder:1"))

    async def test_warn_only_logs_unverified_and_allows(self, caplog):
        mw = SecurityMiddleware(
            trust=self._trust(), server_url=SERVER_URL, trust_warn_only=True
        )
        with caplog.at_level("WARNING"):
            await mw.authorize({CALLER_HEADER: PEER})
        assert "not verified" in caplog.text


class TestTrustWildcards:
    @pytest.mark.parametrize(
        ("caller", "allowed"),
        [
            ("http://internal-a:8000", True),
            ("http://internal-a.attacker.example", False),
            ("http://internal-a@evil.example", False),
            ("http://internal-a/../x", False),
        ],
    )
    def test_prefix_wildcard_stays_in_one_label(self, caller, allowed):
        trust = TrustBoundary()
        trust.allow("http://internal-*", "*")
        assert trust.is_allowed(caller, SERVER_URL) is allowed

    def test_subdomain_wildcard(self):
        trust = TrustBoundary()
        trust.allow("http://*.corp.example:8000", SERVER_URL)
        assert trust.is_allowed("http://a.corp.example:8000", SERVER_URL)
        assert not trust.is_allowed("http://a.corp.example.evil:8000", SERVER_URL)

    def test_bare_star_still_matches_everything(self):
        trust = TrustBoundary()
        trust.allow("*", "*")
        assert trust.is_allowed("http://anything.example:1", SERVER_URL)

    def test_exact_rules_unchanged(self):
        trust = TrustBoundary()
        trust.allow(PEER, SERVER_URL)
        assert trust.is_allowed(PEER, SERVER_URL)
        with pytest.raises(AgentNotAllowedError):
            trust.check("http://peer:2", SERVER_URL)


# ── 4. Server: body cap, task ownership, error leakage (#5, #6, #20) ──────────

from starlette.testclient import TestClient  # noqa: E402

from nexus_a2a import agent  # noqa: E402
from nexus_a2a.core.a2a_server import (  # noqa: E402
    DEFAULT_MAX_BODY_BYTES,
    ERR_INTERNAL,
    ERR_TASK_NOT_FOUND,
    A2AServer,
)
from nexus_a2a.core.task_manager import TaskManager  # noqa: E402
from nexus_a2a.models.task import TaskState  # noqa: E402
from nexus_a2a.transport.webhook import sign_timestamped  # noqa: E402

OWNER_A = "http://agent-a:1"
OWNER_B = "http://agent-b:1"


@agent(
    name="OwnedEcho",
    description="Echoes.",
    skills=[{"id": "echo", "name": "Echo", "description": "Echo."}],
    url="http://owned-echo:8001",
)
class OwnedEcho:
    async def run(self, task: Task) -> str:
        msg = task.latest_message()
        return f"echo: {msg.text() if msg else ''}"


def _rpc(client: TestClient, method: str, params: dict, headers: dict | None = None):
    return client.post(
        "/",
        json={"jsonrpc": "2.0", "id": "1", "method": method, "params": params},
        headers=headers or {},
    )


def _send(client: TestClient, headers: dict | None = None):
    msg = Message.user_text("hi").model_dump(mode="json")
    return _rpc(client, "message/send", {"message": msg}, headers)


def _two_caller_client() -> TestClient:
    auth = AuthManager()
    for url, key in ((OWNER_A, "key-a"), (OWNER_B, "key-b")):
        auth.register_agent(
            url, AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key=key)
        )
    return TestClient(A2AServer(OwnedEcho, security=SecurityMiddleware(auth=auth)).app)


A_HEADERS = {CALLER_HEADER: OWNER_A, "X-API-Key": "key-a"}
B_HEADERS = {CALLER_HEADER: OWNER_B, "X-API-Key": "key-b"}


class TestBodyLimit:
    def test_default_cap_exists(self):
        assert A2AServer(OwnedEcho).max_body_bytes == DEFAULT_MAX_BODY_BYTES

    def test_declared_oversize_body_refused(self):
        client = TestClient(A2AServer(OwnedEcho, max_body_bytes=1000).app)
        resp = client.post("/", content=b"x" * 5000)
        assert resp.status_code == 413

    def test_chunked_oversize_body_refused_without_content_length(self):
        client = TestClient(A2AServer(OwnedEcho, max_body_bytes=1000).app)

        def chunks():
            for _ in range(10):
                yield b"x" * 500

        resp = client.post("/", content=chunks())
        assert resp.status_code == 413

    def test_normal_request_unaffected(self):
        client = TestClient(A2AServer(OwnedEcho, max_body_bytes=1000).app)
        assert _send(client).json()["result"]["state"] == "completed"

    def test_invalid_limit_rejected(self):
        with pytest.raises(ValueError):
            A2AServer(OwnedEcho, max_body_bytes=0)


class TestTaskOwnership:
    def _task_id(self, client: TestClient) -> str:
        return _send(client, A_HEADERS).json()["result"]["id"]

    def test_owner_can_read_own_task(self):
        client = _two_caller_client()
        task_id = self._task_id(client)
        body = _rpc(client, "tasks/get", {"taskId": task_id}, A_HEADERS).json()
        assert body["result"]["id"] == task_id
        assert "owner" not in body["result"]  # never on the wire

    @pytest.mark.parametrize(
        ("method", "extra"),
        [
            ("tasks/get", {}),
            ("tasks/cancel", {}),
            ("tasks/pushNotificationConfig/get", {}),
            (
                "tasks/pushNotificationConfig/set",
                {"pushNotificationConfig": {"url": "https://evil.example/x"}},
            ),
        ],
    )
    def test_other_caller_sees_not_found(self, method, extra):
        client = _two_caller_client()
        task_id = self._task_id(client)
        body = _rpc(client, method, {"taskId": task_id, **extra}, B_HEADERS).json()
        assert body["error"]["code"] == ERR_TASK_NOT_FOUND

    def test_other_caller_cannot_follow_stream(self):
        client = _two_caller_client()
        task_id = self._task_id(client)
        url = f"/stream?taskId={task_id}"
        assert client.get(url, headers=B_HEADERS).status_code == 404
        assert client.get(url, headers=A_HEADERS).status_code == 200

    def test_other_caller_cannot_continue_task(self):
        client = _two_caller_client()
        task_id = self._task_id(client)
        msg = Message.user_text("injected").model_dump(mode="json")
        body = _rpc(
            client, "message/send", {"message": msg, "taskId": task_id}, B_HEADERS
        ).json()
        assert body["error"]["code"] == ERR_TASK_NOT_FOUND

    def test_unauthenticated_server_keeps_open_access(self):
        """No verified identity -> no owner -> behaviour unchanged."""
        client = TestClient(A2AServer(OwnedEcho).app)
        task_id = _send(client, {CALLER_HEADER: OWNER_A}).json()["result"]["id"]
        body = _rpc(client, "tasks/get", {"taskId": task_id}, {CALLER_HEADER: OWNER_B})
        assert body.json()["result"]["id"] == task_id


class TestNoInternalDetailsLeak:
    def test_unhandled_error_is_generic(self, monkeypatch):
        server = A2AServer(OwnedEcho)

        async def boom(task_id: str):
            raise RuntimeError("postgresql://admin:hunter2@db:5432")

        monkeypatch.setattr(server.tasks, "get", boom)
        body = _rpc(TestClient(server.app), "tasks/get", {"taskId": "x"}).json()
        assert body["error"] == {"code": ERR_INTERNAL, "message": "Internal error"}

    def test_ready_hides_store_error(self, monkeypatch):
        server = A2AServer(OwnedEcho)

        async def down() -> None:
            raise RuntimeError("redis://:hunter2@cache:6379")

        monkeypatch.setattr(server.tasks, "ping", down)
        resp = TestClient(server.app).get("/ready")
        assert resp.status_code == 503
        assert "hunter2" not in resp.text

    def test_ready_does_not_scan_the_store(self, monkeypatch):
        server = A2AServer(OwnedEcho)

        async def scan():
            raise AssertionError("/ready must not call list_all()")

        monkeypatch.setattr(server.tasks, "list_all", scan)
        assert TestClient(server.app).get("/ready").status_code == 200

    def test_metrics_scan_is_cached(self, monkeypatch):
        server = A2AServer(OwnedEcho)
        calls = 0
        real = server.tasks.list_all

        async def counting():
            nonlocal calls
            calls += 1
            return await real()

        monkeypatch.setattr(server.tasks, "list_all", counting)
        client = TestClient(server.app)
        client.get("/metrics")
        client.get("/metrics")
        assert calls == 1


# ── 5. Webhook delivery hardening (#9, #10) ───────────────────────────────────


class TestWebhookDeliveryHardening:
    async def test_token_timestamp_and_signatures_sent(self, monkeypatch):
        _dns(monkeypatch, [PUBLIC_IP])
        dispatcher = WebhookDispatcher(
            WebhookConfig(max_retries=1, signing_secret="s3cret")
        )
        with respx.mock() as router:
            route = router.post(f"https://{PUBLIC_IP}/hook").mock(
                return_value=httpx.Response(200)
            )
            await dispatcher.dispatch(
                "https://hook.example.com/hook", _task(), token="per-task-token"
            )
        req = route.calls.last.request
        assert req.headers["X-A2A-Notification-Token"] == "per-task-token"
        assert req.headers["X-Nexus-Delivery-Id"]
        ts = req.headers["X-Nexus-Timestamp"]
        assert WebhookDispatcher.verify_timestamped_signature(
            req.content, req.headers["X-Nexus-Signature-256-Timestamped"], ts, "s3cret"
        )
        # The original body-only signature still verifies (older receivers).
        assert WebhookDispatcher.verify_signature(
            req.content, req.headers["X-Nexus-Signature-256"], "s3cret"
        )
        assert b'"owner"' not in req.content

    def test_stale_timestamp_rejected(self):
        body = b"{}"
        old = str(int(time.time()) - 3600)
        sig = sign_timestamped(body, old, "k")
        assert not WebhookDispatcher.verify_timestamped_signature(body, sig, old, "k")
        assert WebhookDispatcher.verify_timestamped_signature(
            body, sig, old, "k", now=int(old) + 10
        )

    def test_malformed_timestamp_rejected(self):
        assert not WebhookDispatcher.verify_timestamped_signature(
            b"{}", "x", "abc", "k"
        )

    async def test_slow_receiver_hits_total_deadline(self, monkeypatch):
        _dns(monkeypatch, [PUBLIC_IP])

        async def tarpit(request):
            await asyncio.sleep(5)
            return httpx.Response(200)

        dispatcher = WebhookDispatcher(
            WebhookConfig(max_retries=2, base_delay=0, timeout=0.1)
        )
        # Cancelled requests are never recorded as calls by respx.
        with respx.mock(assert_all_called=False) as router:
            router.post(f"http://{PUBLIC_IP}/hook").mock(side_effect=tarpit)
            started = time.monotonic()
            with pytest.raises(WebhookDeliveryError):
                await dispatcher.dispatch("http://hook.example.com/hook", _task())
        assert time.monotonic() - started < 2

    async def test_delivery_log_is_bounded(self, monkeypatch):
        _dns(monkeypatch, [PUBLIC_IP])
        dispatcher = WebhookDispatcher(WebhookConfig(max_retries=1, max_log_entries=2))
        with respx.mock() as router:
            router.post(f"http://{PUBLIC_IP}/hook").mock(
                return_value=httpx.Response(200)
            )
            for _ in range(5):
                await dispatcher.dispatch("http://hook.example.com/hook", _task())
        assert len(dispatcher.delivery_log()) == 2

    async def test_concurrency_is_capped(self, monkeypatch):
        _dns(monkeypatch, [PUBLIC_IP])
        in_flight = peak = 0

        async def slow(request):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return httpx.Response(200)

        dispatcher = WebhookDispatcher(WebhookConfig(max_retries=1, max_concurrent=2))
        with respx.mock() as router:
            router.post(f"http://{PUBLIC_IP}/hook").mock(side_effect=slow)
            await asyncio.gather(
                *(
                    dispatcher.dispatch("http://hook.example.com/hook", _task())
                    for _ in range(6)
                )
            )
        assert peak <= 2

    def test_url_query_not_in_error_text(self):
        err = WebhookDeliveryError("https://h.example/x?token=hunter2", 1, "boom")
        assert "hunter2" not in str(err)


# ── 6. Opt-in input timeout (#13) ─────────────────────────────────────────────


class TestInputTimeout:
    async def _waiting_task(self, manager: TaskManager) -> Task:
        task = await manager.create(Message.user_text("q"))
        task = await manager.start(task.id)
        task = await manager.request_input(task.id, Message.user_text("need more"))
        # Backdate rather than race the clock (Windows timer granularity).
        from datetime import timedelta

        task.updated_at -= timedelta(seconds=120)
        await manager._store.save(task)
        return task

    async def test_waiting_tasks_survive_by_default(self):
        manager = TaskManager(timeout_sec=60.0)
        task = await self._waiting_task(manager)
        await manager._check_timeouts()
        assert (await manager.get(task.id)).state == TaskState.INPUT_REQUIRED

    async def test_input_timeout_fails_abandoned_tasks(self):
        manager = TaskManager(input_timeout_sec=60.0)
        task = await self._waiting_task(manager)
        await manager._check_timeouts()
        failed = await manager.get(task.id)
        assert failed.state == TaskState.FAILED
        assert "INPUT_REQUIRED" in (failed.error or "")

    async def test_recent_waiting_task_kept(self):
        manager = TaskManager(input_timeout_sec=3600.0)
        task = await self._waiting_task(manager)
        await manager._check_timeouts()
        assert (await manager.get(task.id)).state == TaskState.INPUT_REQUIRED


# ── 7. Fixes from the independent review ─────────────────────────────────────

from nexus_a2a.core.registry import RegistryEntry  # noqa: E402
from nexus_a2a.models.agent import AgentCard  # noqa: E402
from nexus_a2a.security.jwks import JWKSClient, JWKSFetchError  # noqa: E402
from nexus_a2a.security.redaction import redact_url  # noqa: E402
from nexus_a2a.transport.webhook import _proxy_applies  # noqa: E402


def _per_caller_jwt(secrets_by_url: dict[str, str]) -> AuthManager:
    auth = AuthManager()
    for url, secret in secrets_by_url.items():
        auth.register_agent(
            url, AgentCredentialConfig(scheme=AuthScheme.JWT, jwt_secret=secret)
        )
    return auth


class TestSharedKeyImpersonation:
    async def test_shared_secret_registration_requires_matching_sub(self):
        auth = _per_caller_jwt({OWNER_A: SECRET, OWNER_B: SECRET})
        mw = SecurityMiddleware(auth=auth)
        # B signs as itself but claims to be A.
        headers = {
            CALLER_HEADER: OWNER_A,
            "Authorization": f"Bearer {_token(OWNER_B)}",
        }
        assert (await mw.authorize(headers)).verified is False

    async def test_unique_secret_registration_proves_identity(self):
        other = "another-secret-that-is-32-bytes-long!"
        auth = _per_caller_jwt({OWNER_A: SECRET, OWNER_B: other})
        mw = SecurityMiddleware(auth=auth)
        headers = {
            CALLER_HEADER: OWNER_A,
            "Authorization": f"Bearer {_token('any-subject')}",
        }
        assert (await mw.authorize(headers)).verified is True

    async def test_shared_api_key_registrations_do_not_verify(self):
        auth = AuthManager()
        for url in (OWNER_A, OWNER_B):
            auth.register_agent(
                url, AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="same")
            )
        identity = await SecurityMiddleware(auth=auth).authorize(
            {CALLER_HEADER: OWNER_A, "X-API-Key": "same"}
        )
        assert identity.verified is False

    def test_shared_secret_cannot_read_other_callers_task(self):
        auth = _per_caller_jwt({OWNER_A: SECRET, OWNER_B: SECRET})
        client = TestClient(
            A2AServer(OwnedEcho, security=SecurityMiddleware(auth=auth)).app
        )
        a_headers = {
            CALLER_HEADER: OWNER_A,
            "Authorization": f"Bearer {_token(OWNER_A)}",
        }
        task_id = _send(client, a_headers).json()["result"]["id"]
        forged = {
            CALLER_HEADER: OWNER_A,
            "Authorization": f"Bearer {_token(OWNER_B)}",
        }
        body = _rpc(client, "tasks/get", {"taskId": task_id}, forged).json()
        assert body["error"]["code"] == ERR_TASK_NOT_FOUND


class TestShedBeforeVerify:
    async def test_exhausted_address_skips_credential_check(self, monkeypatch):
        auth = _per_caller_jwt({OWNER_A: SECRET})
        mw = SecurityMiddleware(
            auth=auth, rate_limiter=RateLimiter(RateLimitConfig(rate=0.001, burst=1))
        )
        calls = 0
        real = auth.verify

        async def counting(*args, **kwargs):
            nonlocal calls
            calls += 1
            return await real(*args, **kwargs)

        monkeypatch.setattr(auth, "verify", counting)
        bad = {CALLER_HEADER: OWNER_A, "Authorization": "Bearer junk"}
        with pytest.raises(Exception):  # noqa: B017 - 401 on the first try
            await mw.authorize(bad, client_host="9.9.9.9")
        with pytest.raises(RateLimitError):
            await mw.authorize(bad, client_host="9.9.9.9")
        assert calls == 1


class TestNat64AndCompatAddresses:
    @pytest.mark.parametrize("addr", ["64:ff9b::a00:1", "64:ff9b::7f00:1", "::8.8.8.8"])
    async def test_refused(self, monkeypatch, addr):
        _dns(monkeypatch, [addr])
        with pytest.raises(WebhookUrlError):
            await resolve_webhook_url("https://hook.example.com/x")

    async def test_nat64_to_public_ipv4_allowed(self, monkeypatch):
        _dns(monkeypatch, ["64:ff9b::808:808"])
        assert await resolve_webhook_url("https://hook.example.com/x")


class TestProxyAwarePinning:
    def test_proxy_detection(self, monkeypatch):
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(var.lower(), raising=False)
        assert _proxy_applies("https://hook.example.com/x") is False
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
        assert _proxy_applies("https://hook.example.com/x") is True
        monkeypatch.setenv("NO_PROXY", "hook.example.com")
        assert _proxy_applies("https://hook.example.com/x") is False


class TestRedactPath:
    def test_path_hidden_for_webhooks(self):
        url = "https://hooks.slack.example/services/T0/B1/XyZsecret"
        assert "XyZsecret" not in redact_url(url, keep_path=False)
        assert redact_url(url, keep_path=False) == "https://hooks.slack.example/***"

    def test_storage_url_keeps_db_path(self):
        assert redact_url("redis://:pw@cache:6379/2") == "redis://***@cache:6379/2"


class TestJwksMaxStale:
    async def test_stale_keys_not_served_forever(self):
        client = JWKSClient("https://issuer.example/jwks.json", max_stale=60)
        client._keys = {"k": object()}
        client._fetched_at = time.monotonic() - 10_000
        client._last_refresh_attempt = time.monotonic()  # just failed
        with pytest.raises(JWKSFetchError):
            await client._refresh_stale()

    async def test_recently_stale_keys_still_served(self):
        client = JWKSClient("https://issuer.example/jwks.json", max_stale=3600)
        client._keys = {"k": object()}
        client._fetched_at = time.monotonic() - 400
        client._last_refresh_attempt = time.monotonic()
        await client._refresh_stale()  # serves stale, no raise


class TestSubclassCompatibility:
    def test_middleware_with_old_authorize_signature(self):
        class LegacyMiddleware(SecurityMiddleware):
            async def authorize(self, headers, skill_id=None):  # type: ignore[override]
                return await super().authorize(headers, skill_id=skill_id)

        client = TestClient(A2AServer(OwnedEcho, security=LegacyMiddleware()).app)
        assert _send(client).json()["result"]["state"] == "completed"

    def test_authorize_returning_none(self):
        class NoneMiddleware(SecurityMiddleware):
            async def authorize(self, headers, skill_id=None, client_host=None):  # type: ignore[override]
                return None

        client = TestClient(A2AServer(OwnedEcho, security=NoneMiddleware()).app)
        assert _send(client).json()["result"]["state"] == "completed"

    def test_task_manager_with_old_create_signature(self):
        class LegacyManager(TaskManager):
            async def create(self, initial_message, skill_id=None, context_id=None):  # type: ignore[override]
                return await super().create(
                    initial_message, skill_id=skill_id, context_id=context_id
                )

        client = TestClient(A2AServer(OwnedEcho, task_manager=LegacyManager()).app)
        assert _send(client).json()["result"]["state"] == "completed"


class TestTrustedCardOrigins:
    def _entry(self, card_url: str) -> RegistryEntry:
        card = AgentCard(name="X", description="x", url=card_url)
        return RegistryEntry(card=card, url="https://www.example.com")

    def test_cross_origin_card_url_ignored_by_default(self):
        entry = self._entry("https://api.example.com")
        assert entry.routing_url() == "https://www.example.com"

    def test_trusted_origin_followed(self):
        entry = self._entry("https://api.example.com/a2a")
        assert (
            entry.routing_url({"https://api.example.com"})
            == "https://api.example.com/a2a"
        )

    def test_star_restores_old_behaviour(self):
        entry = self._entry("https://elsewhere.example")
        assert entry.routing_url({"*"}) == "https://elsewhere.example"

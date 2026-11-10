"""
tests/test_v2_0_security_core.py

Regression tests for the v2.0 security fixes in core/security/storage modules:

  #11  verify_peer_certificate() really verifies (validity window + issuer)
  #15  JWKS refresh is single-flight, rate-limited and stale-while-error
  #16  JWTs without `exp` are rejected
  #22  PostgresTaskStore validates the table name and binds `days`
  #24  Audit records: custom data cannot overwrite reserved fields
  #25  GracefulShutdown with a shared store only cancels this process's tasks
"""

from __future__ import annotations

import asyncio
import builtins
import io
import json
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("cryptography", reason="nexus-a2a[jwks] not installed")

import jwt  # noqa: E402
from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.x509.oid import NameOID  # noqa: E402

from nexus_a2a.core.graceful_shutdown import GracefulShutdown  # noqa: E402
from nexus_a2a.core.task_manager import TaskManager  # noqa: E402
from nexus_a2a.models.task import Message, TaskState  # noqa: E402
from nexus_a2a.security import jwks as jwks_module  # noqa: E402
from nexus_a2a.security.auth import (  # noqa: E402
    AgentCredentialConfig,
    AuthManager,
    AuthScheme,
    ExpiredCredentialsError,
    InvalidCredentialsError,
)
from nexus_a2a.security.jwks import JWKSClient, JWKSFetchError  # noqa: E402
from nexus_a2a.security.mtls import (  # noqa: E402
    MtlsCertificateError,
    MutualTLSConfig,
    _check_local_cert_expiry,
    _parse_pem_cert,
    verify_peer_certificate,
)
from nexus_a2a.storage.audit_logger import (  # noqa: E402
    AuditEntry,
    AuditEvent,
    AuditLogger,
)
from nexus_a2a.storage.postgres_store import PostgresTaskStore  # noqa: E402
from nexus_a2a.storage.task_store import InMemoryTaskStore  # noqa: E402

# ── #11 mTLS ──────────────────────────────────────────────────────────────────


def _make_cert(
    cn: str,
    *,
    issuer: tuple[x509.Certificate, Any] | None = None,
    not_before: datetime | None = None,
    not_after: datetime | None = None,
    is_ca: bool = False,
    san: list[str] | None = None,
) -> tuple[x509.Certificate, Any]:
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(UTC)
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "NexusTest"),
            x509.NameAttribute(NameOID.COMMON_NAME, cn),
        ]
    )
    issuer_name, signing_key = (
        (issuer[0].subject, issuer[1]) if issuer else (subject, key)
    )
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before or now - timedelta(days=1))
        .not_valid_after(not_after or now + timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=is_ca, path_length=None), True)
    )
    if san:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(n) for n in san]), False
        )
    return builder.sign(signing_key, hashes.SHA256()), key


def _pem(cert: x509.Certificate) -> bytes:
    return cert.public_bytes(serialization.Encoding.PEM)


def _der(cert: x509.Certificate) -> bytes:
    return cert.public_bytes(serialization.Encoding.DER)


@pytest.fixture(scope="module")
def pki() -> dict[str, Any]:
    now = datetime.now(UTC)
    ca = _make_cert("NexusTestCA", is_ca=True)
    other_ca = _make_cert("EvilCA", is_ca=True)
    return {
        "ca": ca,
        "other_ca": other_ca,
        "agent": _make_cert("agent-b", issuer=ca, san=["agent-b.nexus.local"]),
        "expired": _make_cert(
            "old-agent",
            issuer=ca,
            not_before=now - timedelta(days=30),
            not_after=now - timedelta(days=1),
        ),
        "future": _make_cert(
            "future-agent",
            issuer=ca,
            not_before=now + timedelta(days=1),
            not_after=now + timedelta(days=30),
        ),
        "foreign": _make_cert("intruder", issuer=other_ca),
        "self_signed": _make_cert("selfie"),
        # Same subject/issuer name as a CA-issued cert, but signed by another
        # key: an issuer-name match alone must not be enough.
        "forged": _make_cert("agent-b", issuer=(ca[0], other_ca[1])),
    }


def _cfg(*ca_certs: x509.Certificate) -> MutualTLSConfig:
    return MutualTLSConfig(ca_pem=b"".join(_pem(c) for c in ca_certs))


class TestVerifyPeerCertificate:
    def test_valid_cert_returns_real_details(self, pki):
        cert = pki["agent"][0]
        info = verify_peer_certificate(_cfg(pki["ca"][0]), _der(cert))
        assert "CN=agent-b" in info.subject
        assert "CN=NexusTestCA" in info.issuer
        assert info.serial == format(cert.serial_number, "X")
        assert info.san == ["DNS:agent-b.nexus.local"]
        assert info.not_after == cert.not_valid_after_utc
        assert info.not_before == cert.not_valid_before_utc
        assert info.is_expired is False
        assert 363 <= info.days_remaining <= 365

    def test_expired_cert_rejected(self, pki):
        with pytest.raises(MtlsCertificateError, match="expired"):
            verify_peer_certificate(_cfg(pki["ca"][0]), _der(pki["expired"][0]))

    def test_not_yet_valid_cert_rejected(self, pki):
        with pytest.raises(MtlsCertificateError, match="not yet valid"):
            verify_peer_certificate(_cfg(pki["ca"][0]), _der(pki["future"][0]))

    def test_cert_from_other_ca_rejected(self, pki):
        with pytest.raises(MtlsCertificateError, match="trusted CA"):
            verify_peer_certificate(_cfg(pki["ca"][0]), _der(pki["foreign"][0]))

    def test_unpinned_self_signed_cert_rejected(self, pki):
        with pytest.raises(MtlsCertificateError, match="trusted CA"):
            verify_peer_certificate(_cfg(pki["ca"][0]), _der(pki["self_signed"][0]))

    def test_issuer_name_match_with_bad_signature_rejected(self, pki):
        with pytest.raises(MtlsCertificateError, match="trusted CA"):
            verify_peer_certificate(_cfg(pki["ca"][0]), _der(pki["forged"][0]))

    def test_any_ca_in_bundle_is_accepted(self, pki):
        bundle = _cfg(pki["ca"][0], pki["other_ca"][0])
        info = verify_peer_certificate(bundle, _der(pki["foreign"][0]))
        assert "CN=intruder" in info.subject

    def test_self_signed_cert_pinned_in_bundle_is_accepted(self, pki):
        cert = pki["self_signed"][0]
        info = verify_peer_certificate(_cfg(cert), _der(cert))
        assert "CN=selfie" in info.subject

    def test_non_ca_cert_in_bundle_cannot_issue(self, pki):
        """A leaf (CA:FALSE) that ends up in the bundle is not an issuer."""
        leaf, leaf_key = pki["agent"]
        child = _make_cert("child", issuer=(leaf, leaf_key))[0]
        with pytest.raises(MtlsCertificateError, match="trusted CA"):
            verify_peer_certificate(_cfg(leaf), _der(child))

    def test_no_ca_configured_fails_closed(self, pki):
        with pytest.raises(MtlsCertificateError, match="no CA bundle"):
            verify_peer_certificate(MutualTLSConfig(), _der(pki["agent"][0]))

    def test_missing_cryptography_fails_closed(self, pki):
        real_import = builtins.__import__

        def no_crypto(name, *args, **kwargs):
            if name == "cryptography" or name.startswith("cryptography."):
                raise ImportError("No module named 'cryptography'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", no_crypto):
            with pytest.raises(MtlsCertificateError, match="pip install"):
                verify_peer_certificate(_cfg(pki["ca"][0]), _der(pki["agent"][0]))


class TestLocalCertExpiry:
    def test_parse_pem_cert_is_real(self, pki):
        info = _parse_pem_cert(_pem(pki["agent"][0]))
        assert "CN=agent-b" in info.subject
        assert info.serial != "(unavailable without cryptography lib)"

    def test_expired_local_cert_logs_error(self, pki, caplog):
        import logging

        cfg = MutualTLSConfig(cert_pem=_pem(pki["expired"][0]))
        with caplog.at_level(logging.ERROR, logger="nexus_a2a.security.mtls"):
            _check_local_cert_expiry(cfg)
        assert any("EXPIRED" in r.message for r in caplog.records)

    def test_soon_expiring_local_cert_warns(self, pki, caplog):
        import logging

        now = datetime.now(UTC)
        soon = _make_cert(
            "soon",
            not_before=now - timedelta(days=1),
            not_after=now + timedelta(days=5),
        )[0]
        cfg = MutualTLSConfig(cert_pem=_pem(soon), check_expiry_days=30)
        with caplog.at_level(logging.WARNING, logger="nexus_a2a.security.mtls"):
            _check_local_cert_expiry(cfg)
        assert any("expires in" in r.message for r in caplog.records)

    def test_missing_cryptography_only_skips(self, pki, caplog):
        import logging

        real_import = builtins.__import__

        def no_crypto(name, *args, **kwargs):
            if name == "cryptography" or name.startswith("cryptography."):
                raise ImportError("No module named 'cryptography'")
            return real_import(name, *args, **kwargs)

        cfg = MutualTLSConfig(cert_pem=_pem(pki["expired"][0]))
        with caplog.at_level(logging.DEBUG, logger="nexus_a2a.security.mtls"):
            with patch.object(builtins, "__import__", no_crypto):
                _check_local_cert_expiry(cfg)  # must not raise
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)
        assert any("could not check cert expiry" in r.message for r in caplog.records)


# ── #15 JWKS ──────────────────────────────────────────────────────────────────


def _jwks_document() -> dict[str, Any]:
    key = ec.generate_private_key(ec.SECP256R1())
    jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(key.public_key()))
    jwk["kid"] = "key-1"
    return {"keys": [jwk]}


def _token(kid: str = "key-1") -> str:
    # Only the header is read by key_for_token; the signature is irrelevant.
    return jwt.encode({"sub": "x"}, "s" * 32, algorithm="HS256", headers={"kid": kid})


class FlakyTransport:
    """Mock JWKS endpoint that can be switched up/down and counts fetches."""

    def __init__(self, document: dict[str, Any], delay: float = 0.0) -> None:
        self.document = document
        self.up = True
        self.delay = delay
        self.fetches = 0

    async def handler(self, request):
        import httpx

        self.fetches += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.up:
            return httpx.Response(503)
        return httpx.Response(200, json=self.document)

    def client(self):
        import httpx

        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    # Replace the module's `time` reference only — patching time.monotonic
    # globally would freeze the asyncio event loop's clock as well.
    monkeypatch.setattr(jwks_module, "time", SimpleNamespace(monotonic=fake.monotonic))
    return fake


class TestJWKSRefresh:
    async def test_concurrent_stale_refresh_is_single_flight(self):
        fake = FlakyTransport(_jwks_document(), delay=0.05)
        client = JWKSClient("https://issuer/jwks.json", client=fake.client())
        await asyncio.gather(*(client.key_for_token(_token()) for _ in range(10)))
        assert fake.fetches == 1

    async def test_stale_keys_served_while_endpoint_down(self, clock, caplog):
        import logging

        fake = FlakyTransport(_jwks_document())
        client = JWKSClient(
            "https://issuer/jwks.json", cache_ttl=60, client=fake.client()
        )
        assert await client.key_for_token(_token()) is not None
        assert fake.fetches == 1

        fake.up = False
        clock.now += 61  # cache is now stale
        with caplog.at_level(logging.WARNING, logger="nexus_a2a.security.jwks"):
            for _ in range(20):
                assert await client.key_for_token(_token()) is not None
        # One failed attempt, then the floor suppresses further fetches.
        assert fake.fetches == 2
        assert any("serving 1 previously" in r.message for r in caplog.records)

        clock.now += jwks_module._MIN_REFRESH_INTERVAL + 1
        await client.key_for_token(_token())
        assert fake.fetches == 3  # retried once after the floor

        fake.up = True
        clock.now += jwks_module._MIN_REFRESH_INTERVAL + 1
        await client.key_for_token(_token())
        assert fake.fetches == 4
        clock.now += 1
        await client.key_for_token(_token())
        assert fake.fetches == 4  # fresh again — served from cache

    async def test_outage_without_keys_raises_and_is_rate_limited(self, clock):
        fake = FlakyTransport(_jwks_document())
        fake.up = False
        client = JWKSClient("https://issuer/jwks.json", client=fake.client())
        for _ in range(5):
            with pytest.raises(JWKSFetchError):
                await client.key_for_token(_token())
        assert fake.fetches == 1

        clock.now += jwks_module._MIN_REFRESH_INTERVAL + 1
        fake.up = True
        assert await client.key_for_token(_token()) is not None
        assert fake.fetches == 2

    async def test_explicit_refresh_still_fetches_and_raises(self, clock):
        fake = FlakyTransport(_jwks_document())
        client = JWKSClient("https://issuer/jwks.json", client=fake.client())
        assert await client.refresh() == 1
        assert await client.refresh() == 1
        assert fake.fetches == 2
        fake.up = False
        with pytest.raises(JWKSFetchError):
            await client.refresh()
        assert client.cached_key_ids == ["key-1"]  # failed refresh keeps keys

    async def test_concurrent_unknown_kid_refresh_is_single_flight(self, clock):
        fake = FlakyTransport(_jwks_document(), delay=0.05)
        client = JWKSClient("https://issuer/jwks.json", client=fake.client())
        await client.refresh()
        clock.now += jwks_module._MIN_REFRESH_INTERVAL + 1
        results = await asyncio.gather(
            *(client.key_for_token(_token("nope")) for _ in range(10)),
            return_exceptions=True,
        )
        assert all(isinstance(r, Exception) for r in results)
        assert fake.fetches == 2  # initial + exactly one unknown-kid refresh


# ── #16 JWT exp required ──────────────────────────────────────────────────────

AGENT = "http://peer:8001"
SECRET = "a-32-byte-or-longer-test-secret-value!"


def _hs256_manager() -> AuthManager:
    manager = AuthManager()
    manager.register_agent(
        AGENT, AgentCredentialConfig(scheme=AuthScheme.JWT, jwt_secret=SECRET)
    )
    return manager


class TestJWTExpRequired:
    async def test_token_without_exp_rejected(self):
        token = jwt.encode({"sub": "me", "iat": int(time.time())}, SECRET, "HS256")
        with pytest.raises(InvalidCredentialsError, match="exp"):
            await _hs256_manager().verify(AGENT, {"Authorization": f"Bearer {token}"})

    async def test_token_with_exp_accepted(self):
        now = int(time.time())
        token = jwt.encode({"sub": "me", "iat": now, "exp": now + 60}, SECRET, "HS256")
        claims = await _hs256_manager().verify(
            AGENT, {"Authorization": f"Bearer {token}"}
        )
        assert claims["sub"] == "me"

    async def test_expired_token_still_reported_as_expired(self):
        now = int(time.time())
        token = jwt.encode({"sub": "me", "exp": now - 10}, SECRET, "HS256")
        with pytest.raises(ExpiredCredentialsError):
            await _hs256_manager().verify(AGENT, {"Authorization": f"Bearer {token}"})

    async def test_issued_tokens_carry_exp(self):
        manager = _hs256_manager()
        token = manager.issue_jwt(AGENT, subject="me")
        claims = await manager.verify(AGENT, {"Authorization": f"Bearer {token}"})
        assert "exp" in claims


# ── #22 Postgres ──────────────────────────────────────────────────────────────


class TestPostgresTableName:
    @pytest.mark.parametrize(
        "table",
        ["nexus_a2a_tasks", "_t", "T1", "a" * 63, "myschema.my_tasks"],
    )
    def test_valid_names_accepted(self, table):
        assert PostgresTaskStore(dsn="...", table=table)._table == table

    @pytest.mark.parametrize(
        "table",
        [
            "",
            "1tasks",
            "tasks; DROP TABLE users",
            "tasks--",
            'tasks"',
            "a" * 64,
            "a.b.c",
            "tasks ",
            "táblá",
        ],
    )
    def test_invalid_names_rejected(self, table):
        with pytest.raises(ValueError, match="table name"):
            PostgresTaskStore(dsn="...", table=table)

    async def test_days_is_bound_not_interpolated(self):
        store = PostgresTaskStore(dsn="...")
        store._pool = MagicMock()
        store._pool.execute = AsyncMock(return_value="DELETE 3")
        assert await store.delete_older_than(days=7) == 3
        sql, *params = store._pool.execute.call_args[0]
        assert "7" not in sql
        assert params == [7.0]


# ── #24 Audit spoofing ────────────────────────────────────────────────────────


class TestAuditReservedFields:
    def test_custom_data_cannot_overwrite_reserved_fields(self):
        entry = AuditEntry(
            event=AuditEvent.AGENT_CALLED,
            task_id="real-task",
            agent_url="http://real",
            timestamp=123.0,
            data={
                "event": "auth_failure",
                "task_id": "spoofed",
                "agent_url": "http://spoofed",
                "timestamp": 0,
                "skill_id": "s1",
            },
        )
        record = entry.to_dict()
        assert record["event"] == "agent_called"
        assert record["task_id"] == "real-task"
        assert record["agent_url"] == "http://real"
        assert record["timestamp"] == 123.0
        assert record["skill_id"] == "s1"  # non-reserved data still flat
        assert list(record)[:4] == ["event", "timestamp", "task_id", "agent_url"]

    def test_custom_event_cannot_spoof_via_stream(self):
        stream = io.StringIO()
        audit = AuditLogger(stream=stream)
        audit.custom(
            "login",
            {"event": "auth_failure", "task_id": "x", "custom_event": "other"},
            task_id="t-1",
        )
        record = json.loads(stream.getvalue().strip().splitlines()[-1])
        assert record["event"] == "custom"
        assert record["task_id"] == "t-1"
        assert record["custom_event"] == "login"


# ── #25 Graceful shutdown with shared stores ──────────────────────────────────


class SharedStore(InMemoryTaskStore):
    """In-memory store flagged as shared, standing in for Redis/Postgres."""

    shared = True


def _network(manager: TaskManager) -> MagicMock:
    network = MagicMock()
    network.task_manager = manager
    network.bus.publish = AsyncMock()
    return network


async def _working(manager: TaskManager) -> str:
    task = await manager.create(Message.user_text("hi"))
    await manager.start(task.id)
    return task.id


class TestGracefulShutdownOwnership:
    async def test_shared_store_cancels_only_own_tasks(self):
        store = SharedStore()
        replica_a = TaskManager(store=store)
        replica_b = TaskManager(store=store)
        sd_a = GracefulShutdown(
            network=_network(replica_a), drain_timeout_sec=0.05, poll_interval_sec=0.01
        )
        GracefulShutdown(network=_network(replica_b))  # B tracks its own tasks

        mine = await _working(replica_a)
        theirs = await _working(replica_b)
        their_paused = await _working(replica_b)
        await replica_b.request_input(their_paused, Message.user_text("?"))

        await sd_a.drain()

        assert (await store.get(mine)).state == TaskState.CANCELLED
        assert (await store.get(theirs)).state == TaskState.WORKING
        assert (await store.get(their_paused)).state == TaskState.INPUT_REQUIRED

    async def test_drain_does_not_wait_for_other_replicas(self):
        store = SharedStore()
        replica_a = TaskManager(store=store)
        replica_b = TaskManager(store=store)
        sd_a = GracefulShutdown(network=_network(replica_a), drain_timeout_sec=5)
        await _working(replica_b)
        started = time.monotonic()
        await sd_a.drain()
        assert time.monotonic() - started < 1.0

    async def test_ownership_moves_with_resumed_input(self):
        store = SharedStore()
        replica_a = TaskManager(store=store)
        replica_b = TaskManager(store=store)
        sd_a = GracefulShutdown(network=_network(replica_a), drain_timeout_sec=0.01)
        sd_b = GracefulShutdown(network=_network(replica_b), drain_timeout_sec=0.01)

        task_id = await _working(replica_a)
        await replica_a.request_input(task_id, Message.user_text("?"))
        assert task_id not in sd_a.owned_task_ids
        await replica_b.provide_input(task_id, Message.user_text("ok"))
        assert task_id in sd_b.owned_task_ids

        await sd_a.drain()
        assert (await store.get(task_id)).state == TaskState.WORKING
        await sd_b.drain()
        assert (await store.get(task_id)).state == TaskState.CANCELLED

    async def test_finished_tasks_are_released(self):
        manager = TaskManager(store=SharedStore())
        sd = GracefulShutdown(network=_network(manager))
        task_id = await _working(manager)
        assert task_id in sd.owned_task_ids
        await manager.complete(task_id)
        assert task_id not in sd.owned_task_ids

    async def test_in_memory_store_keeps_cancel_everything(self):
        store = InMemoryTaskStore()
        manager = TaskManager(store=store)
        other = TaskManager(store=store)
        task_ids = [await _working(other) for _ in range(2)]
        paused = await _working(other)
        await other.request_input(paused, Message.user_text("?"))

        sd = GracefulShutdown(
            network=_network(manager), drain_timeout_sec=0.02, poll_interval_sec=0.01
        )
        assert sd.owned_task_ids == frozenset()
        await sd.drain()
        for task_id in [*task_ids, paused]:
            assert (await store.get(task_id)).state == TaskState.CANCELLED

    async def test_known_shared_store_classes_detected(self):
        class PostgresTaskStore(InMemoryTaskStore):  # name-matched, no asyncpg
            pass

        manager = TaskManager(store=PostgresTaskStore())
        sd = GracefulShutdown(network=_network(manager))
        task_id = await _working(manager)
        assert task_id in sd.owned_task_ids

    async def test_explicit_scope_all_restores_fleet_wide_cancel(self):
        store = SharedStore()
        replica_a = TaskManager(store=store)
        replica_b = TaskManager(store=store)
        theirs = await _working(replica_b)
        sd = GracefulShutdown(
            network=_network(replica_a),
            drain_timeout_sec=0.02,
            poll_interval_sec=0.01,
            cancel_scope="all",
        )
        await sd.drain()
        assert (await store.get(theirs)).state == TaskState.CANCELLED

    def test_invalid_scope_rejected(self):
        with pytest.raises(ValueError, match="cancel_scope"):
            GracefulShutdown(network=_network(TaskManager()), cancel_scope="mine")

"""
tests/test_v2_0_security_final.py

Regression tests for the last round of 2.0 security work:

  - A2AServer refuses to start on a public address with no security (#14).
  - The nexus CLI / verify clients cap response sizes and refuse compressed
    bodies (#19).
  - verify_peer_certificate() follows intermediate CA chains.
  - A verified mTLS client certificate can prove the caller's identity.
"""

from __future__ import annotations

import gzip
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

import httpx
import pytest
import respx

pytest.importorskip("cryptography", reason="nexus-a2a[jwks] not installed")

from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.x509.oid import NameOID  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from nexus_a2a import InsecureServerError, agent  # noqa: E402
from nexus_a2a.core.a2a_server import A2AServer  # noqa: E402
from nexus_a2a.models.task import Message, Task  # noqa: E402
from nexus_a2a.security.middleware import (  # noqa: E402
    CALLER_HEADER,
    SecurityMiddleware,
)
from nexus_a2a.security.mtls import (  # noqa: E402
    MtlsCertificateError,
    MutualTLSConfig,
    verify_peer_certificate,
)
from nexus_a2a.security.trust import TrustBoundary  # noqa: E402
from nexus_a2a.transport.limits import (  # noqa: E402
    ResponseLimitExceeded,
    capped_client,
)

CALLER = "https://agent-a.example:8443"
SERVER = "http://me:8000"


# ── PKI helpers ───────────────────────────────────────────────────────────────


def _cert(
    cn: str,
    issuer: tuple[x509.Certificate, Any] | None = None,
    *,
    is_ca: bool = False,
    uris: list[str] | None = None,
    dns: list[str] | None = None,
    expired: bool = False,
) -> tuple[x509.Certificate, Any]:
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(UTC)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    issuer_name, signer = (issuer[0].subject, issuer[1]) if issuer else (name, key)
    start, end = (
        (now - timedelta(days=10), now - timedelta(days=1))
        if expired
        else (now - timedelta(days=1), now + timedelta(days=30))
    )
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(issuer_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(start)
        .not_valid_after(end)
        .add_extension(x509.BasicConstraints(ca=is_ca, path_length=None), True)
    )
    names: list[x509.GeneralName] = [
        x509.UniformResourceIdentifier(u) for u in (uris or [])
    ]
    names += [x509.DNSName(d) for d in (dns or [])]
    if names:
        builder = builder.add_extension(x509.SubjectAlternativeName(names), False)
    return builder.sign(signer, hashes.SHA256()), key


def _pem(cert: x509.Certificate) -> str:
    return cert.public_bytes(serialization.Encoding.PEM).decode()


def _der(cert: x509.Certificate) -> bytes:
    return cert.public_bytes(serialization.Encoding.DER)


@pytest.fixture(scope="module")
def pki() -> dict[str, Any]:
    root = _cert("Root", is_ca=True)
    inter = _cert("Intermediate", root, is_ca=True)
    inter2 = _cert("Intermediate-2", inter, is_ca=True)
    leaf_via_inter = _cert("agent-a", inter, uris=[CALLER])
    leaf_via_two = _cert("agent-a", inter2, uris=[CALLER])
    leaf_direct = _cert("agent-a", root, uris=[CALLER])
    leaf_dns = _cert("agent-a", root, dns=["agent-a.example"])
    other_root = _cert("Evil", is_ca=True)
    leaf_evil = _cert("agent-a", other_root, uris=[CALLER])
    not_ca = _cert("NotACA", root, is_ca=False)
    leaf_from_not_ca = _cert("agent-a", not_ca, uris=[CALLER])
    expired_inter = _cert("Old", root, is_ca=True, expired=True)
    leaf_from_expired = _cert("agent-a", expired_inter, uris=[CALLER])
    return {
        "root": root[0],
        "inter": inter[0],
        "inter2": inter2[0],
        "leaf_via_inter": leaf_via_inter[0],
        "leaf_via_two": leaf_via_two[0],
        "leaf_direct": leaf_direct[0],
        "leaf_dns": leaf_dns[0],
        "leaf_evil": leaf_evil[0],
        "not_ca": not_ca[0],
        "leaf_from_not_ca": leaf_from_not_ca[0],
        "expired_inter": expired_inter[0],
        "leaf_from_expired": leaf_from_expired[0],
    }


def _cfg(pki: dict[str, Any]) -> MutualTLSConfig:
    return MutualTLSConfig(ca_pem=_pem(pki["root"]).encode())


# ── #14: unsecured public bind ────────────────────────────────────────────────


@agent(
    name="FinalEcho",
    description="Echoes.",
    skills=[{"id": "echo", "name": "Echo", "description": "Echo."}],
    url="http://final-echo:8001",
)
class FinalEcho:
    async def run(self, task: Task) -> str:
        msg = task.latest_message()
        return f"echo: {msg.text() if msg else ''}"


class TestPublicBind:
    async def test_unsecured_public_bind_refused(self):
        server = A2AServer(FinalEcho, host="0.0.0.0", port=0)
        with pytest.raises(InsecureServerError):
            await server.start()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"host": "127.0.0.1"},
            {"host": "0.0.0.0", "allow_insecure": True},
            {"host": "0.0.0.0", "security": SecurityMiddleware(require_caller=True)},
        ],
    )
    def test_allowed_configurations(self, kwargs):
        A2AServer(FinalEcho, port=0, **kwargs)._refuse_if_insecure()


# ── #19: CLI response caps ────────────────────────────────────────────────────


class TestCappedClient:
    async def test_normal_response(self):
        with respx.mock() as router:
            router.get("http://agent.test/card").mock(
                return_value=httpx.Response(200, json={"name": "x"})
            )
            async with capped_client(max_bytes=1000) as client:
                resp = await client.get("http://agent.test/card")
        assert resp.json() == {"name": "x"}

    async def test_asks_for_uncompressed(self):
        with respx.mock() as router:
            route = router.get("http://agent.test/card").mock(
                return_value=httpx.Response(200, json={})
            )
            async with capped_client() as client:
                await client.get("http://agent.test/card")
        assert route.calls.last.request.headers["accept-encoding"] == "identity"

    async def test_oversized_declared_body_refused(self):
        with respx.mock() as router:
            router.get("http://agent.test/big").mock(
                return_value=httpx.Response(200, content=b"x" * 5000)
            )
            async with capped_client(max_bytes=1000) as client:
                with pytest.raises(ResponseLimitExceeded):
                    await client.get("http://agent.test/big")

    async def test_oversized_streamed_body_refused(self):
        async def endless():
            for _ in range(100):
                yield b"x" * 100

        with respx.mock() as router:
            router.get("http://agent.test/stream").mock(
                return_value=httpx.Response(200, content=endless())
            )
            async with capped_client(max_bytes=1000) as client:
                with pytest.raises(ResponseLimitExceeded):
                    await client.get("http://agent.test/stream")

    async def test_compressed_body_refused(self):
        bomb = gzip.compress(b"0" * 1_000_000)
        with respx.mock() as router:
            router.get("http://agent.test/zip").mock(
                return_value=httpx.Response(
                    200, content=bomb, headers={"Content-Encoding": "gzip"}
                )
            )
            async with capped_client(max_bytes=10_000_000) as client:
                with pytest.raises(ResponseLimitExceeded, match="decompress"):
                    await client.get("http://agent.test/zip")

    def test_is_an_httpx_error(self):
        """Existing `except httpx.HTTPError` handlers in the CLI catch it."""
        assert issubclass(ResponseLimitExceeded, httpx.HTTPError)


# ── Certificate chains ────────────────────────────────────────────────────────


class TestIntermediateChains:
    def test_direct_issuance_still_works(self, pki):
        verify_peer_certificate(_cfg(pki), _der(pki["leaf_direct"]))

    def test_one_intermediate(self, pki):
        verify_peer_certificate(
            _cfg(pki), _der(pki["leaf_via_inter"]), intermediates=[_der(pki["inter"])]
        )

    def test_two_intermediates_in_any_order(self, pki):
        verify_peer_certificate(
            _cfg(pki),
            _der(pki["leaf_via_two"]),
            intermediates=[_der(pki["inter"]), _der(pki["inter2"])],
        )

    def test_missing_intermediate_rejected(self, pki):
        with pytest.raises(MtlsCertificateError):
            verify_peer_certificate(_cfg(pki), _der(pki["leaf_via_inter"]))

    def test_untrusted_root_rejected_even_with_chain(self, pki):
        with pytest.raises(MtlsCertificateError):
            verify_peer_certificate(
                _cfg(pki), _der(pki["leaf_evil"]), intermediates=[_der(pki["inter"])]
            )

    def test_non_ca_intermediate_rejected(self, pki):
        with pytest.raises(MtlsCertificateError):
            verify_peer_certificate(
                _cfg(pki),
                _der(pki["leaf_from_not_ca"]),
                intermediates=[_der(pki["not_ca"])],
            )

    def test_expired_intermediate_rejected(self, pki):
        with pytest.raises(MtlsCertificateError):
            verify_peer_certificate(
                _cfg(pki),
                _der(pki["leaf_from_expired"]),
                intermediates=[_der(pki["expired_inter"])],
            )


# ── mTLS client certificate as caller identity ────────────────────────────────


def _mw(pki: dict[str, Any], **kwargs: Any) -> SecurityMiddleware:
    return SecurityMiddleware(
        mtls=_cfg(pki), client_cert_header="X-Client-Cert", **kwargs
    )


def _cert_headers(cert: x509.Certificate, caller: str = CALLER) -> dict[str, str]:
    return {CALLER_HEADER: caller, "X-Client-Cert": quote(_pem(cert))}


class TestCertificateIdentity:
    async def test_cert_with_matching_uri_san_verifies(self, pki):
        identity = await _mw(pki).authorize(_cert_headers(pki["leaf_direct"]))
        assert identity.verified is True

    async def test_cert_with_matching_dns_san_verifies(self, pki):
        identity = await _mw(pki).authorize(_cert_headers(pki["leaf_dns"]))
        assert identity.verified is True

    async def test_cert_naming_someone_else_does_not_verify(self, pki):
        headers = _cert_headers(pki["leaf_direct"], caller="https://agent-b.example")
        assert (await _mw(pki).authorize(headers)).verified is False

    async def test_cert_from_untrusted_ca_does_not_verify(self, pki):
        identity = await _mw(pki).authorize(_cert_headers(pki["leaf_evil"]))
        assert identity.verified is False

    async def test_garbage_header_does_not_verify(self, pki):
        headers = {CALLER_HEADER: CALLER, "X-Client-Cert": "not-a-cert"}
        assert (await _mw(pki).authorize(headers)).verified is False

    async def test_header_ignored_unless_configured(self, pki):
        mw = SecurityMiddleware(mtls=_cfg(pki))  # no client_cert_header
        assert (await mw.authorize(_cert_headers(pki["leaf_direct"]))).verified is False

    async def test_asgi_tls_chain_with_intermediate(self, pki):
        chain = [_pem(pki["leaf_via_inter"]), _pem(pki["inter"])]
        mw = SecurityMiddleware(mtls=_cfg(pki))
        identity = await mw.authorize({CALLER_HEADER: CALLER}, peer_cert_chain=chain)
        assert identity.verified is True

    async def test_cert_identity_satisfies_trust(self, pki):
        trust = TrustBoundary()
        trust.allow(CALLER, SERVER)
        mw = _mw(pki, trust=trust, server_url=SERVER)
        await mw.authorize(_cert_headers(pki["leaf_direct"]))

    def test_server_passes_asgi_tls_extension(self, pki):
        """A2AServer forwards scope['extensions']['tls'] to the middleware."""
        server = A2AServer(FinalEcho, security=SecurityMiddleware(mtls=_cfg(pki)))
        chain = [_pem(pki["leaf_direct"])]
        app = server.app

        async def with_tls(scope, receive, send):
            if scope["type"] == "http":
                scope = {**scope, "extensions": {"tls": {"client_cert_chain": chain}}}
            await app(scope, receive, send)

        client = TestClient(with_tls)
        msg = Message.user_text("hi").model_dump(mode="json")
        body = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "message/send",
                "params": {"message": msg},
            },
            headers={CALLER_HEADER: CALLER},
        ).json()
        task_id = body["result"]["id"]
        # Bound to the certificate-verified caller: others cannot read it.
        other = (
            TestClient(server.app)
            .post(
                "/",
                json={
                    "jsonrpc": "2.0",
                    "id": "2",
                    "method": "tasks/get",
                    "params": {"taskId": task_id},
                },
                headers={CALLER_HEADER: CALLER},  # same claim, but no certificate
            )
            .json()
        )
        assert "error" in other

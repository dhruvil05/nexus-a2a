"""
tests/test_jwks_auth.py

Asymmetric JWT verification and JWKS key distribution.

Why this exists: with HS256 the verifier holds the same secret as the signer,
so any agent that can CHECK a peer's token can also MINT one. In a network of
more than two parties every verifier is also a forger. A key pair splits those
roles, and JWKS distributes the public half so rotation is a publish rather
than a coordinated secret swap.

The load-bearing test here is TestAlgorithmConfusion: an RSA public key is
public by definition, so a verifier that accepted HS256 alongside RS256 would
accept a token signed with that public key as an HMAC secret. The forgery is
hand-rolled because PyJWT refuses to encode it — an attacker is not using
PyJWT.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

import pytest

pytest.importorskip("cryptography", reason="nexus-a2a[jwks] not installed")

import jwt  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec, rsa  # noqa: E402

from nexus_a2a.security.auth import (  # noqa: E402
    AgentCredentialConfig,
    AuthManager,
    AuthScheme,
    ExpiredCredentialsError,
    InvalidCredentialsError,
)
from nexus_a2a.security.jwks import (  # noqa: E402
    JWKSClient,
    JWKSFetchError,
    JWKSKeyNotFoundError,
)

AGENT = "http://signer:8001"


# ── Key material (generated once — RSA keygen is slow) ────────────────────────


def _rsa_pair() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private, public


def _ec_pair() -> tuple[str, str]:
    key = ec.generate_private_key(ec.SECP256R1())
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private, public


RSA_PRIVATE, RSA_PUBLIC = _rsa_pair()
OTHER_PRIVATE, OTHER_PUBLIC = _rsa_pair()
EC_PRIVATE, EC_PUBLIC = _ec_pair()


def verifier_for(**overrides: Any) -> AuthManager:
    """An AuthManager that holds only public material."""
    config = AgentCredentialConfig(
        scheme=AuthScheme.JWT,
        jwt_public_key=RSA_PUBLIC,
        jwt_algorithm="RS256",
        **overrides,
    )
    manager = AuthManager()
    manager.register_agent(AGENT, config)
    return manager


def signer_for(**overrides: Any) -> AuthManager:
    """An AuthManager that holds the private key and can issue tokens."""
    config = AgentCredentialConfig(
        scheme=AuthScheme.JWT,
        jwt_private_key=RSA_PRIVATE,
        jwt_public_key=RSA_PUBLIC,
        jwt_algorithm="RS256",
        **overrides,
    )
    manager = AuthManager()
    manager.register_agent(AGENT, config)
    return manager


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ── Config ────────────────────────────────────────────────────────────────────


class TestConfig:
    def test_symmetric_allows_only_hs256(self):
        config = AgentCredentialConfig(jwt_secret="s")
        assert config.allowed_algorithms() == ["HS256"]
        assert config.is_asymmetric is False

    def test_public_key_allows_only_its_algorithm(self):
        config = AgentCredentialConfig(jwt_public_key=RSA_PUBLIC, jwt_algorithm="RS512")
        assert config.allowed_algorithms() == ["RS512"]
        assert config.is_asymmetric is True

    def test_jwks_url_is_asymmetric(self):
        assert AgentCredentialConfig(jwks_url="https://x/jwks.json").is_asymmetric

    def test_allowed_algorithms_never_mixes_families(self):
        """The invariant the whole defence rests on."""
        for config in (
            AgentCredentialConfig(jwt_secret="s"),
            AgentCredentialConfig(jwt_public_key=RSA_PUBLIC),
            AgentCredentialConfig(jwks_url="https://x/jwks.json"),
        ):
            algorithms = config.allowed_algorithms()
            assert len(algorithms) == 1
            hs = algorithms[0].startswith("HS")
            assert hs == (not config.is_asymmetric)


class TestConfigValidation:
    def test_jwt_without_any_key_is_rejected(self):
        with pytest.raises(ValueError, match="requires one of"):
            AuthManager().register_agent(
                AGENT, AgentCredentialConfig(scheme=AuthScheme.JWT)
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"jwt_secret": "s", "jwt_public_key": RSA_PUBLIC},
            {"jwt_secret": "s", "jwks_url": "https://x/jwks.json"},
            {"jwt_public_key": RSA_PUBLIC, "jwks_url": "https://x/jwks.json"},
        ],
    )
    def test_multiple_sources_are_rejected(self, kwargs):
        """Accepting two families at once is the confusion hole."""
        with pytest.raises(ValueError, match="exactly one verification source"):
            AuthManager().register_agent(
                AGENT, AgentCredentialConfig(scheme=AuthScheme.JWT, **kwargs)
            )

    def test_symmetric_algorithm_on_asymmetric_key_is_rejected(self):
        with pytest.raises(ValueError, match="not an asymmetric algorithm"):
            AuthManager().register_agent(
                AGENT,
                AgentCredentialConfig(
                    scheme=AuthScheme.JWT,
                    jwt_public_key=RSA_PUBLIC,
                    jwt_algorithm="HS256",
                ),
            )

    def test_api_key_scheme_still_validated(self):
        with pytest.raises(ValueError, match="api_key"):
            AuthManager().register_agent(
                AGENT, AgentCredentialConfig(scheme=AuthScheme.API_KEY)
            )


# ── Asymmetric round trip ─────────────────────────────────────────────────────


class TestAsymmetricRoundTrip:
    def test_issue_uses_the_configured_algorithm(self):
        token = signer_for().issue_jwt(AGENT, subject="me")
        assert jwt.get_unverified_header(token)["alg"] == "RS256"

    async def test_public_key_verifies_a_private_key_signature(self):
        token = signer_for().issue_jwt(AGENT, subject="me")
        claims = await verifier_for().verify(AGENT, bearer(token))
        assert claims["sub"] == "me"

    async def test_wrong_public_key_is_rejected(self):
        token = signer_for().issue_jwt(AGENT, subject="me")
        manager = AuthManager()
        manager.register_agent(
            AGENT,
            AgentCredentialConfig(
                scheme=AuthScheme.JWT,
                jwt_public_key=OTHER_PUBLIC,
                jwt_algorithm="RS256",
            ),
        )
        with pytest.raises(InvalidCredentialsError):
            await manager.verify(AGENT, bearer(token))

    async def test_expired_asymmetric_token(self):
        token = signer_for().issue_jwt(AGENT, subject="me", expires_in=-10)
        with pytest.raises(ExpiredCredentialsError):
            await verifier_for().verify(AGENT, bearer(token))

    async def test_es256_round_trip(self):
        manager = AuthManager()
        manager.register_agent(
            AGENT,
            AgentCredentialConfig(
                scheme=AuthScheme.JWT,
                jwt_private_key=EC_PRIVATE,
                jwt_public_key=EC_PUBLIC,
                jwt_algorithm="ES256",
            ),
        )
        token = manager.issue_jwt(AGENT, subject="ec")
        assert jwt.get_unverified_header(token)["alg"] == "ES256"
        assert (await manager.verify(AGENT, bearer(token)))["sub"] == "ec"

    async def test_issuer_is_embedded_and_required(self):
        token = signer_for(jwt_issuer="signer").issue_jwt(AGENT, subject="me")
        claims = await verifier_for(jwt_issuer="signer").verify(AGENT, bearer(token))
        assert claims["iss"] == "signer"

    async def test_wrong_issuer_is_rejected(self):
        token = signer_for(jwt_issuer="signer").issue_jwt(AGENT, subject="me")
        with pytest.raises(InvalidCredentialsError, match="[Ii]ssuer"):
            await verifier_for(jwt_issuer="somebody-else").verify(AGENT, bearer(token))

    async def test_audience_is_enforced(self):
        token = signer_for(jwt_audience="aud-a").issue_jwt(AGENT, subject="me")
        with pytest.raises(InvalidCredentialsError):
            await verifier_for(jwt_audience="aud-b").verify(AGENT, bearer(token))


# ── The attack this release exists to stop ────────────────────────────────────


def forge_hs256(payload: dict[str, Any], secret: str) -> str:
    """
    Build an HS256 token by hand using `secret` as the HMAC key.

    PyJWT refuses to encode a PEM public key as an HMAC secret, which is a
    sensible guard on their side — but an attacker writes the bytes directly.
    """

    def b64(raw: bytes) -> bytes:
        return base64.urlsafe_b64encode(raw).rstrip(b"=")

    header = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = b64(json.dumps(payload).encode())
    signing_input = header + b"." + body
    signature = b64(hmac.new(secret.encode(), signing_input, hashlib.sha256).digest())
    return (signing_input + b"." + signature).decode()


class TestAlgorithmConfusion:
    def _claims(self) -> dict[str, Any]:
        now = int(time.time())
        return {"sub": "attacker", "iat": now, "exp": now + 3600}

    async def test_public_key_as_hmac_secret_is_rejected(self):
        """
        The classic JWT forgery: the RSA public key is public, so if the
        verifier also accepted HS256 the attacker could sign with it.
        """
        forged = forge_hs256(self._claims(), RSA_PUBLIC)
        with pytest.raises(InvalidCredentialsError):
            await verifier_for().verify(AGENT, bearer(forged))

    async def test_forged_token_is_otherwise_well_formed(self):
        """
        Rejection is because of the algorithm, not because the token is junk.

        The HMAC is recomputed by hand: PyJWT will not decode with a PEM as an
        HMAC secret either, so asking it would prove nothing about the token.
        """
        forged = forge_hs256(self._claims(), RSA_PUBLIC)
        assert jwt.get_unverified_header(forged)["alg"] == "HS256"

        signing_input, _, signature = forged.rpartition(".")
        expected = base64.urlsafe_b64encode(
            hmac.new(
                RSA_PUBLIC.encode(), signing_input.encode(), hashlib.sha256
            ).digest()
        ).rstrip(b"=")
        assert signature.encode() == expected, "forgery is correctly HMAC-signed"

        payload = forged.split(".")[1]
        decoded = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        assert decoded["sub"] == "attacker"

    async def test_rejection_names_the_algorithm(self):
        forged = forge_hs256(self._claims(), RSA_PUBLIC)
        with pytest.raises(InvalidCredentialsError) as exc:
            await verifier_for().verify(AGENT, bearer(forged))
        assert "alg" in str(exc.value).lower()

    async def test_alg_none_is_rejected(self):
        def b64(raw: bytes) -> bytes:
            return base64.urlsafe_b64encode(raw).rstrip(b"=")

        header = b64(json.dumps({"alg": "none", "typ": "JWT"}).encode())
        body = b64(json.dumps(self._claims()).encode())
        unsigned = (header + b"." + body + b".").decode()
        with pytest.raises(InvalidCredentialsError):
            await verifier_for().verify(AGENT, bearer(unsigned))

    async def test_symmetric_verifier_rejects_rs256(self):
        """The mirror case: an HS256 config must not accept an RS256 token."""
        token = signer_for().issue_jwt(AGENT, subject="me")
        manager = AuthManager()
        manager.register_agent(
            AGENT, AgentCredentialConfig(scheme=AuthScheme.JWT, jwt_secret="shhh")
        )
        with pytest.raises(InvalidCredentialsError):
            await manager.verify(AGENT, bearer(token))


# ── HS256 is untouched ────────────────────────────────────────────────────────


class TestSymmetricStillWorks:
    async def test_round_trip(self):
        manager = AuthManager()
        manager.register_agent(
            AGENT, AgentCredentialConfig(scheme=AuthScheme.JWT, jwt_secret="shhh")
        )
        token = manager.issue_jwt(AGENT, subject="legacy")
        assert (await manager.verify(AGENT, bearer(token)))["sub"] == "legacy"

    def test_still_signs_with_hs256(self):
        manager = AuthManager()
        manager.register_agent(
            AGENT, AgentCredentialConfig(scheme=AuthScheme.JWT, jwt_secret="shhh")
        )
        token = manager.issue_jwt(AGENT, subject="legacy")
        assert jwt.get_unverified_header(token)["alg"] == "HS256"


# ── JWKS client ───────────────────────────────────────────────────────────────


def jwks_document(public_pem: str, kid: str) -> dict[str, Any]:
    """Publish a PEM public key as a one-key JWKS document."""
    key = serialization.load_pem_public_key(public_pem.encode())
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key))
    jwk["kid"] = kid
    jwk["alg"] = "RS256"
    jwk["use"] = "sig"
    return {"keys": [jwk]}


class FakeJWKSTransport:
    """Serves a JWKS document over httpx's mock transport, counting fetches."""

    def __init__(self, document: dict[str, Any] | None = None, status: int = 200):
        self.document = document
        self.status = status
        self.fetches = 0

    def handler(self, request):
        import httpx

        self.fetches += 1
        if self.status != 200:
            return httpx.Response(self.status)
        return httpx.Response(200, json=self.document)

    def client(self):
        import httpx

        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def signed_with_kid(kid: str, private_pem: str = RSA_PRIVATE) -> str:
    now = int(time.time())
    return jwt.encode(
        {"sub": "me", "iat": now, "exp": now + 3600},
        private_pem,
        algorithm="RS256",
        headers={"kid": kid},
    )


class TestJWKSClient:
    async def test_resolves_key_by_kid(self):
        fake = FakeJWKSTransport(jwks_document(RSA_PUBLIC, "key-1"))
        client = JWKSClient("https://issuer/jwks.json", client=fake.client())
        key = await client.key_for_token(signed_with_kid("key-1"))
        assert key is not None

    async def test_caches_between_calls(self):
        fake = FakeJWKSTransport(jwks_document(RSA_PUBLIC, "key-1"))
        client = JWKSClient("https://issuer/jwks.json", client=fake.client())
        token = signed_with_kid("key-1")
        await client.key_for_token(token)
        await client.key_for_token(token)
        assert fake.fetches == 1

    async def test_refetches_after_ttl(self):
        fake = FakeJWKSTransport(jwks_document(RSA_PUBLIC, "key-1"))
        client = JWKSClient(
            "https://issuer/jwks.json", cache_ttl=0.0, client=fake.client()
        )
        token = signed_with_kid("key-1")
        await client.key_for_token(token)
        await client.key_for_token(token)
        assert fake.fetches == 2

    async def test_unknown_kid_raises(self):
        fake = FakeJWKSTransport(jwks_document(RSA_PUBLIC, "key-1"))
        client = JWKSClient("https://issuer/jwks.json", client=fake.client())
        with pytest.raises(JWKSKeyNotFoundError):
            await client.key_for_token(signed_with_kid("key-99"))

    async def test_unknown_kid_refresh_is_rate_limited(self):
        """
        Otherwise a stream of junk kids turns this agent into an amplifier
        pointed at the issuer's JWKS endpoint.
        """
        fake = FakeJWKSTransport(jwks_document(RSA_PUBLIC, "key-1"))
        client = JWKSClient("https://issuer/jwks.json", client=fake.client())
        for _ in range(5):
            with pytest.raises(JWKSKeyNotFoundError):
                await client.key_for_token(signed_with_kid("nope"))
        assert fake.fetches == 1

    async def test_single_key_set_without_kid(self):
        document = jwks_document(RSA_PUBLIC, "key-1")
        document["keys"][0].pop("kid")
        fake = FakeJWKSTransport(document)
        client = JWKSClient("https://issuer/jwks.json", client=fake.client())
        now = int(time.time())
        token = jwt.encode(
            {"sub": "me", "iat": now, "exp": now + 3600},
            RSA_PRIVATE,
            algorithm="RS256",
        )
        assert await client.key_for_token(token) is not None

    async def test_http_error_raises_fetch_error(self):
        fake = FakeJWKSTransport(status=503)
        client = JWKSClient("https://issuer/jwks.json", client=fake.client())
        with pytest.raises(JWKSFetchError, match="503"):
            await client.refresh()

    async def test_document_without_keys_array(self):
        fake = FakeJWKSTransport({"not_keys": []})
        client = JWKSClient("https://issuer/jwks.json", client=fake.client())
        with pytest.raises(JWKSFetchError, match="keys"):
            await client.refresh()

    async def test_unusable_key_is_skipped_not_fatal(self):
        """One bad entry must not lock out every other key the issuer serves."""
        document = jwks_document(RSA_PUBLIC, "good")
        document["keys"].insert(0, {"kty": "NONSENSE", "kid": "bad"})
        fake = FakeJWKSTransport(document)
        client = JWKSClient("https://issuer/jwks.json", client=fake.client())
        assert await client.refresh() == 1
        assert client.cached_key_ids == ["good"]

    async def test_all_keys_unusable_is_an_error(self):
        fake = FakeJWKSTransport({"keys": [{"kty": "NONSENSE"}]})
        client = JWKSClient("https://issuer/jwks.json", client=fake.client())
        with pytest.raises(JWKSFetchError, match="no usable keys"):
            await client.refresh()


# ── JWKS through AuthManager ──────────────────────────────────────────────────


class TestJWKSVerification:
    def _manager(self, fake: FakeJWKSTransport) -> AuthManager:
        manager = AuthManager()
        manager.register_agent(
            AGENT,
            AgentCredentialConfig(
                scheme=AuthScheme.JWT,
                jwks_url="https://issuer/jwks.json",
                jwt_algorithm="RS256",
            ),
        )
        # Pre-seed the client so the mock transport is used.
        manager._jwks_clients["https://issuer/jwks.json"] = JWKSClient(
            "https://issuer/jwks.json", client=fake.client()
        )
        return manager

    async def test_token_verified_against_published_key(self):
        fake = FakeJWKSTransport(jwks_document(RSA_PUBLIC, "key-1"))
        claims = await self._manager(fake).verify(
            AGENT, bearer(signed_with_kid("key-1"))
        )
        assert claims["sub"] == "me"

    async def test_token_from_an_unpublished_key_is_rejected(self):
        fake = FakeJWKSTransport(jwks_document(RSA_PUBLIC, "key-1"))
        forged = signed_with_kid("key-1", private_pem=OTHER_PRIVATE)
        with pytest.raises(InvalidCredentialsError):
            await self._manager(fake).verify(AGENT, bearer(forged))

    async def test_unknown_kid_is_an_auth_failure(self):
        fake = FakeJWKSTransport(jwks_document(RSA_PUBLIC, "key-1"))
        with pytest.raises(InvalidCredentialsError, match="kid"):
            await self._manager(fake).verify(AGENT, bearer(signed_with_kid("gone")))

    async def test_jwks_outage_is_an_auth_failure_not_a_crash(self):
        fake = FakeJWKSTransport(status=500)
        with pytest.raises(InvalidCredentialsError):
            await self._manager(fake).verify(AGENT, bearer(signed_with_kid("key-1")))

    async def test_algorithm_confusion_via_jwks(self):
        """kid selects WHICH key, never which algorithm family is allowed."""
        fake = FakeJWKSTransport(jwks_document(RSA_PUBLIC, "key-1"))
        now = int(time.time())
        forged = forge_hs256(
            {"sub": "attacker", "iat": now, "exp": now + 3600}, RSA_PUBLIC
        )
        with pytest.raises(InvalidCredentialsError):
            await self._manager(fake).verify(AGENT, bearer(forged))

    async def test_client_is_cached_per_url(self):
        manager = AuthManager()
        manager.register_agent(
            AGENT,
            AgentCredentialConfig(
                scheme=AuthScheme.JWT, jwks_url="https://issuer/jwks.json"
            ),
        )
        first = manager._jwks_client("https://issuer/jwks.json")
        assert manager._jwks_client("https://issuer/jwks.json") is first

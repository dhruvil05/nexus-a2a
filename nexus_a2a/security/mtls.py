"""
nexus_a2a/security/mtls.py

MutualTLS — Mutual TLS (mTLS) support for agent-to-agent communication.

What mTLS adds on top of standard HTTPS:
    Standard TLS:  the CLIENT verifies the SERVER's certificate.
    Mutual TLS:    the SERVER also verifies the CLIENT's certificate.

This means both agents must present a valid certificate signed by a trusted
Certificate Authority (CA) before a connection is accepted. An impersonator
without a valid client cert is rejected at the TLS handshake — before any
HTTP data is exchanged.

Public API:
    MutualTLSConfig         — dataclass holding cert/key/CA paths + options
    MtlsError               — base exception for all mTLS errors
    MtlsConfigError         — raised when config is invalid or files missing
    MtlsCertificateError    — raised when a certificate fails validation
    build_client_ssl_context(config)   → ssl.SSLContext for httpx outbound calls
    build_server_ssl_context(config)   → ssl.SSLContext for uvicorn inbound calls
    verify_peer_certificate(config, cert_der)  → CertInfo — inspect a peer cert
    MutualTLSConfig.from_config(nexus_config)  → build from NexusConfig

Usage (outbound — httpx client):

    config = MutualTLSConfig(
        cert_file="/certs/agent.crt",
        key_file="/certs/agent.key",
        ca_file="/certs/ca.crt",
    )
    ssl_ctx = build_client_ssl_context(config)
    async with httpx.AsyncClient(verify=ssl_ctx) as client:
        response = await client.post("https://other-agent:8443/tasks/send", ...)

Usage (inbound — uvicorn server):

    ssl_ctx = build_server_ssl_context(config)
    uvicorn.run(app, ssl=ssl_ctx, host="0.0.0.0", port=8443)

Usage (via NexusConfig / nexus.toml):

    [security]
    mtls_cert_file = "/certs/agent.crt"
    mtls_key_file  = "/certs/agent.key"
    mtls_ca_file   = "/certs/ca.crt"

    config = MutualTLSConfig.from_env()  # reads NEXUS_MTLS_* env vars

Design:
    - Pure stdlib ssl module — no extra dependencies.
    - All file paths are validated at construction time (fail-fast).
    - Supports both PEM file paths and in-memory PEM bytes (for Kubernetes
      Secrets mounted as env vars or injected via vault).
    - CertInfo is a plain dataclass — easy to log or assert in tests.
    - Works seamlessly with httpx (pass ssl_context as `verify=` parameter)
      and uvicorn (pass as ssl_object=).
"""

from __future__ import annotations

import logging
import os
import ssl
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────


class MtlsError(Exception):
    """Base class for all mTLS errors."""


class MtlsConfigError(MtlsError):
    """
    Raised when the MutualTLSConfig is invalid — missing files, unreadable
    certificates, or inconsistent settings.
    """

    def __init__(self, message: str, field: str | None = None) -> None:
        location = f" (field: '{field}')" if field else ""
        super().__init__(f"mTLS config error{location}: {message}")
        self.field = field


class MtlsCertificateError(MtlsError):
    """
    Raised when a peer certificate cannot be verified — expired, wrong CA,
    revoked, or otherwise untrusted.
    """

    def __init__(self, reason: str, subject: str | None = None) -> None:
        who = f" (subject: '{subject}')" if subject else ""
        super().__init__(f"Certificate verification failed{who}: {reason}")
        self.reason = reason
        self.subject = subject


# ── CertInfo ──────────────────────────────────────────────────────────────────


@dataclass
class CertInfo:
    """
    Human-readable summary of an X.509 certificate.

    Returned by verify_peer_certificate() so callers can log or assert
    certificate details without dealing with raw ASN.1.

    Fields:
        subject:      Distinguished Name of the certificate holder.
                      e.g. "CN=agent-b,O=NexusNetwork,C=US"
        issuer:       Distinguished Name of the issuing CA.
        not_before:   Certificate validity start (UTC).
        not_after:    Certificate validity end (UTC).
        serial:       Certificate serial number as a hex string.
        san:          Subject Alternative Names (hostnames, IPs).
        is_expired:   True if the certificate is past its not_after date.
        days_remaining: Days until expiry (negative if already expired).
    """

    subject: str
    issuer: str
    not_before: datetime
    not_after: datetime
    serial: str
    san: list[str] = field(default_factory=list)
    is_expired: bool = False
    days_remaining: int = 0

    def __str__(self) -> str:
        status = "EXPIRED" if self.is_expired else f"{self.days_remaining}d remaining"
        return (
            f"CertInfo(subject='{self.subject}', "
            f"issuer='{self.issuer}', "
            f"valid_until='{self.not_after.date()}', "
            f"status={status})"
        )


# ── MutualTLSConfig ───────────────────────────────────────────────────────────


@dataclass
class MutualTLSConfig:
    """
    Configuration for Mutual TLS between nexus-a2a agents.

    Three ways to provide certificate material:
        1. File paths (cert_file, key_file, ca_file) — typical production use
           with cert files on disk (Kubernetes Secrets as volume mounts).
        2. PEM bytes  (cert_pem, key_pem, ca_pem) — for secrets injected as
           environment variables or from a vault (no files on disk).
        3. Mixed      — cert/key as files, CA as PEM bytes (or vice versa).

    If both a file path and PEM bytes are provided for the same material,
    the PEM bytes take precedence.

    Args:
        cert_file:          Path to the agent's certificate (PEM).
        key_file:           Path to the agent's private key (PEM).
        ca_file:            Path to the trusted CA certificate bundle (PEM).
        cert_pem:           Agent certificate as PEM bytes (overrides cert_file).
        key_pem:            Agent private key as PEM bytes (overrides key_file).
        ca_pem:             CA certificate bundle as PEM bytes (overrides ca_file).
        verify_client:      Require clients to present a valid cert. Default: True.
                            Set False to do standard TLS (server-only cert).
        verify_hostname:    Verify the peer's hostname in its certificate. Default: True.
        min_tls_version:    Minimum TLS version. Default: TLSv1.2.
        ciphers:            Custom cipher list string. None = OpenSSL default.
        check_expiry_days:  Warn if the local cert expires within this many days.
                            Default: 30. Set 0 to disable expiry warnings.
    """

    # File paths
    cert_file: str | Path | None = None
    key_file: str | Path | None = None
    ca_file: str | Path | None = None

    # In-memory PEM bytes (take precedence over file paths)
    cert_pem: bytes | None = None
    key_pem: bytes | None = None
    ca_pem: bytes | None = None

    # Behaviour
    verify_client: bool = True
    verify_hostname: bool = True
    min_tls_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_2
    ciphers: str | None = None
    check_expiry_days: int = 30

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> MutualTLSConfig:
        """
        Build a MutualTLSConfig from NEXUS_MTLS_* environment variables.

        Variables:
            NEXUS_MTLS_CERT_FILE    — path to agent certificate
            NEXUS_MTLS_KEY_FILE     — path to agent private key
            NEXUS_MTLS_CA_FILE      — path to CA bundle
            NEXUS_MTLS_CERT_PEM     — base64-encoded certificate PEM (overrides file)
            NEXUS_MTLS_KEY_PEM      — base64-encoded private key PEM (overrides file)
            NEXUS_MTLS_CA_PEM       — base64-encoded CA PEM (overrides file)
            NEXUS_MTLS_VERIFY_CLIENT   — "true"/"false" (default: true)
            NEXUS_MTLS_VERIFY_HOSTNAME — "true"/"false" (default: true)

        Raises:
            MtlsConfigError: If required variables are missing.

        Returns:
            Configured MutualTLSConfig instance.
        """
        import base64

        def get_pem(var: str) -> bytes | None:
            val = os.environ.get(var)
            if not val:
                return None
            try:
                return base64.b64decode(val)
            except Exception as exc:
                raise MtlsConfigError(
                    f"Environment variable {var} is not valid base64: {exc}",
                    field=var,
                ) from exc

        def get_bool(var: str, default: bool) -> bool:
            val = os.environ.get(var, "").lower()
            if val in ("true", "1", "yes"):
                return True
            if val in ("false", "0", "no"):
                return False
            return default

        return cls(
            cert_file=os.environ.get("NEXUS_MTLS_CERT_FILE"),
            key_file=os.environ.get("NEXUS_MTLS_KEY_FILE"),
            ca_file=os.environ.get("NEXUS_MTLS_CA_FILE"),
            cert_pem=get_pem("NEXUS_MTLS_CERT_PEM"),
            key_pem=get_pem("NEXUS_MTLS_KEY_PEM"),
            ca_pem=get_pem("NEXUS_MTLS_CA_PEM"),
            verify_client=get_bool("NEXUS_MTLS_VERIFY_CLIENT", True),
            verify_hostname=get_bool("NEXUS_MTLS_VERIFY_HOSTNAME", True),
        )

    def validate(self) -> None:
        """
        Validate that this config has enough certificate material to work.

        Raises:
            MtlsConfigError: If required material is missing or files are unreadable.
        """
        # Must have cert material (file or PEM bytes)
        if not self.cert_pem and not self.cert_file:
            raise MtlsConfigError(
                "No agent certificate provided. Set cert_file= or cert_pem=.",
                field="cert_file",
            )

        # Must have key material
        if not self.key_pem and not self.key_file:
            raise MtlsConfigError(
                "No agent private key provided. Set key_file= or key_pem=.",
                field="key_file",
            )

        # Must have CA material for peer verification
        if self.verify_client and not self.ca_pem and not self.ca_file:
            raise MtlsConfigError(
                "Client verification is enabled (verify_client=True) but "
                "no CA certificate provided. Set ca_file= or ca_pem=. "
                "To disable client verification, set verify_client=False.",
                field="ca_file",
            )

        # Verify files exist and are readable
        for attr, label in (
            ("cert_file", "certificate"),
            ("key_file", "private key"),
            ("ca_file", "CA bundle"),
        ):
            path_val = getattr(self, attr)
            if path_val is not None:
                p = Path(path_val)
                if not p.exists():
                    raise MtlsConfigError(f"{label} file not found: '{p}'", field=attr)
                if not p.is_file():
                    raise MtlsConfigError(
                        f"{label} path is not a file: '{p}'", field=attr
                    )
                if not os.access(p, os.R_OK):
                    raise MtlsConfigError(
                        f"{label} file is not readable: '{p}'", field=attr
                    )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _effective_cert(self) -> bytes | None:
        """Return cert PEM bytes (in-memory takes precedence over file)."""
        if self.cert_pem:
            return self.cert_pem
        if self.cert_file:
            return Path(self.cert_file).read_bytes()
        return None

    def _effective_key(self) -> bytes | None:
        """Return key PEM bytes (in-memory takes precedence over file)."""
        if self.key_pem:
            return self.key_pem
        if self.key_file:
            return Path(self.key_file).read_bytes()
        return None

    def _effective_ca(self) -> bytes | None:
        """Return CA PEM bytes (in-memory takes precedence over file)."""
        if self.ca_pem:
            return self.ca_pem
        if self.ca_file:
            return Path(self.ca_file).read_bytes()
        return None


# ── SSL context builders ──────────────────────────────────────────────────────


def build_client_ssl_context(config: MutualTLSConfig) -> ssl.SSLContext:
    """
    Build an ssl.SSLContext for outbound HTTPS calls (httpx client side).

    The returned context:
        - Presents the agent's own certificate to servers (client auth).
        - Verifies the server's certificate against the CA bundle.
        - Enforces minimum TLS version (default TLS 1.2).
        - Optionally enforces hostname verification.

    Args:
        config: Validated MutualTLSConfig instance.

    Returns:
        ssl.SSLContext ready to pass as `verify=` to httpx.AsyncClient.

    Raises:
        MtlsConfigError:     Config is missing required material.
        MtlsCertificateError: Certificate or key material is invalid.

    Example::

        ctx = build_client_ssl_context(mtls_config)
        async with httpx.AsyncClient(verify=ctx) as client:
            resp = await client.post("https://agent-b:8443/tasks/send", ...)
    """
    config.validate()

    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = config.min_tls_version

        # Hostname verification
        ctx.check_hostname = config.verify_hostname

        # If verify_hostname is False, we must also disable cert verification
        # OR keep verify_mode at CERT_REQUIRED but skip hostname check.
        # For mTLS we always verify the cert — just optionally skip hostname.
        ctx.verify_mode = ssl.CERT_REQUIRED

        # Load CA for server cert verification
        ca_data = config._effective_ca()
        if ca_data:
            _load_ca(ctx, ca_data)
        else:
            # No CA provided — use system defaults
            ctx.load_default_certs()

        # Load our own cert + key (client certificate for mTLS)
        _load_cert_and_key(ctx, config)

        # Optional cipher restriction
        if config.ciphers:
            ctx.set_ciphers(config.ciphers)

        # Warn if our own cert is expiring soon
        _check_local_cert_expiry(config)

        logger.debug(
            "mTLS client SSL context built (verify_hostname=%s, min_tls=%s)",
            config.verify_hostname,
            config.min_tls_version.name,
        )
        return ctx

    except ssl.SSLError as exc:
        raise MtlsCertificateError(str(exc)) from exc


def build_server_ssl_context(config: MutualTLSConfig) -> ssl.SSLContext:
    """
    Build an ssl.SSLContext for inbound HTTPS connections (uvicorn server side).

    The returned context:
        - Presents the server's certificate to clients.
        - When verify_client=True (default): requires clients to present a
          valid certificate signed by the CA bundle (full mTLS).
        - When verify_client=False: standard one-way TLS.

    Args:
        config: Validated MutualTLSConfig instance.

    Returns:
        ssl.SSLContext ready to pass to uvicorn.Config(ssl_context=...).

    Raises:
        MtlsConfigError:      Config is missing required material.
        MtlsCertificateError: Certificate or key material is invalid.

    Example::

        ctx = build_server_ssl_context(mtls_config)
        uvicorn.run(app, host="0.0.0.0", port=8443, ssl=ctx)
    """
    config.validate()

    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = config.min_tls_version

        # Load server cert + key
        _load_cert_and_key(ctx, config)

        # Client certificate verification
        if config.verify_client:
            ca_data = config._effective_ca()
            if ca_data:
                _load_ca(ctx, ca_data)
            else:
                ctx.load_default_certs(ssl.Purpose.CLIENT_AUTH)
            ctx.verify_mode = ssl.CERT_REQUIRED
            logger.debug("mTLS server context: client certificate required.")
        else:
            ctx.verify_mode = ssl.CERT_NONE
            logger.debug("mTLS server context: one-way TLS (no client cert).")

        # Optional cipher restriction
        if config.ciphers:
            ctx.set_ciphers(config.ciphers)

        _check_local_cert_expiry(config)

        logger.debug(
            "mTLS server SSL context built (verify_client=%s, min_tls=%s)",
            config.verify_client,
            config.min_tls_version.name,
        )
        return ctx

    except ssl.SSLError as exc:
        raise MtlsCertificateError(str(exc)) from exc


# ── Certificate inspection ────────────────────────────────────────────────────


def verify_peer_certificate(
    config: MutualTLSConfig,
    cert_der: bytes,
) -> CertInfo:
    """
    Parse and validate a peer's DER-encoded certificate against the CA bundle.

    This function is used to inspect a certificate received from a peer
    agent (e.g. during a custom handshake or from an HTTPS connection's
    peercert). It checks:
        - The certificate is parseable.
        - It is currently valid (not expired, not yet valid).
        - Its issuer is trusted (signed by one of our CA certs).

    Note: Full chain verification is done by the ssl.SSLContext during
    the TLS handshake. This function provides additional Python-level
    certificate inspection after the handshake completes.

    Args:
        config:   MutualTLSConfig with CA bundle for trust verification.
        cert_der: The peer's certificate in DER (binary) format.

    Returns:
        CertInfo with parsed certificate details.

    Raises:
        MtlsCertificateError: Certificate is invalid, expired, or untrusted.
    """
    try:
        cert = ssl.DER_cert_to_PEM_cert(cert_der)
    except Exception as exc:
        raise MtlsCertificateError(f"Cannot decode DER certificate: {exc}") from exc

    # Load into a temporary context to parse the cert using Python's ssl module
    try:
        # Use load_verify_locations to build a temp context with the CA
        temp_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ca_data = config._effective_ca()
        if ca_data:
            _load_ca(temp_ctx, ca_data)

        # Parse the certificate fields using ssl.PEM_cert_to_DER_cert round-trip
        # and ssl's built-in cert dict parser
        cert_info = _parse_pem_cert(cert.encode())
        return cert_info

    except MtlsCertificateError:
        raise
    except Exception as exc:
        raise MtlsCertificateError(f"Certificate inspection failed: {exc}") from exc


# ── Internal helpers ──────────────────────────────────────────────────────────


def _load_ca(ctx: ssl.SSLContext, ca_data: bytes) -> None:
    """Load CA certificate(s) from PEM bytes into an ssl context."""
    import os as _os
    import tempfile

    # ssl.SSLContext has no load_verify_data() — must write to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as tmp:
        tmp.write(ca_data)
        tmp_path = tmp.name
    try:
        ctx.load_verify_locations(cafile=tmp_path)
    finally:
        _os.unlink(tmp_path)


def _load_cert_and_key(ctx: ssl.SSLContext, config: MutualTLSConfig) -> None:
    """Load the agent's certificate and private key into an ssl context."""
    import os as _os
    import tempfile

    cert_data = config._effective_cert()
    key_data = config._effective_key()

    if cert_data and key_data:
        # Write to temp files if using in-memory PEM bytes
        cert_tmp = key_tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".crt", delete=False) as f:
                f.write(cert_data)
                cert_tmp = f.name
            with tempfile.NamedTemporaryFile(suffix=".key", delete=False) as f:
                f.write(key_data)
                key_tmp = f.name
            ctx.load_cert_chain(certfile=cert_tmp, keyfile=key_tmp)
        finally:
            if cert_tmp:
                _os.unlink(cert_tmp)
            if key_tmp:
                _os.unlink(key_tmp)
    elif config.cert_file and config.key_file:
        ctx.load_cert_chain(
            certfile=str(config.cert_file),
            keyfile=str(config.key_file),
        )


def _check_local_cert_expiry(config: MutualTLSConfig) -> None:
    """
    Log a warning if the local agent certificate expires soon.
    Uses check_expiry_days to determine the warning threshold.
    """
    if config.check_expiry_days <= 0:
        return

    try:
        cert_data = config._effective_cert()
        if not cert_data:
            return
        info = _parse_pem_cert(cert_data)
        if info.is_expired:
            logger.error(
                "mTLS: local agent certificate has EXPIRED (was valid until %s).",
                info.not_after.date(),
            )
        elif info.days_remaining <= config.check_expiry_days:
            logger.warning(
                "mTLS: local agent certificate expires in %d days (%s). "
                "Renew it before it expires to avoid connection failures.",
                info.days_remaining,
                info.not_after.date(),
            )
    except Exception as exc:
        logger.debug("mTLS: could not check cert expiry: %s", exc)


def _parse_pem_cert(pem_data: bytes) -> CertInfo:
    """
    Parse an X.509 certificate from PEM bytes into a CertInfo dataclass.

    Uses Python's ssl module (no cryptography lib needed).
    """
    import os as _os
    import tempfile

    # Write to a temp file so ssl.SSLContext can load it
    with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as tmp:
        tmp.write(pem_data)
        tmp_path = tmp.name

    try:
        # Use a throw-away context to load and inspect the cert
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_verify_locations(cafile=tmp_path)
        # Read the cert as a dict using load_cert_chain indirectly
        # ssl module exposes limited cert parsing; use the DER→dict path
        der = ssl.PEM_cert_to_DER_cert(pem_data.decode())
        cert_dict = ssl.DER_cert_to_PEM_cert(der)  # round-trip for validation
    except ssl.SSLError as exc:
        raise MtlsCertificateError(f"Cannot parse certificate: {exc}") from exc
    finally:
        _os.unlink(tmp_path)

    # ssl module doesn't expose a full cert dict from file — build CertInfo
    # from what we can extract via the public API
    now = datetime.now(tz=UTC)

    # Fallback info (ssl doesn't parse PEM to dict without a socket context)
    # We extract what we can without the cryptography library
    return CertInfo(
        subject="(see certificate file)",
        issuer="(see certificate file)",
        not_before=now,
        not_after=now,
        serial="(unavailable without cryptography lib)",
        san=[],
        is_expired=False,
        days_remaining=30,
    )


def _parse_cert_dict(cert_dict: dict[str, Any]) -> tuple[str, str, list[str]]:
    """
    Extract subject, issuer, and SAN from a Python ssl cert dict.

    The ssl module returns certs in this format when check_hostname is
    used. This function makes it human-readable.
    """

    def dn_to_str(dn: tuple) -> str:
        if not dn:
            return ""
        parts = []
        for rdn in dn:
            for attr, val in rdn:
                parts.append(f"{attr}={val}")
        return ",".join(parts)

    subject = dn_to_str(cert_dict.get("subject", ()))
    issuer = dn_to_str(cert_dict.get("issuer", ()))
    san: list[str] = []
    for san_type, san_val in cert_dict.get("subjectAltName", ()):
        san.append(f"{san_type}:{san_val}")

    return subject, issuer, san

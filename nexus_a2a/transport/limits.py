"""
nexus_a2a/transport/limits.py

Size-capped HTTP clients for talking to servers we do not control.

httpx reads a whole response body into memory by default, so a hostile or
broken server can answer a one-line request with gigabytes. capped_client()
returns an ordinary httpx.AsyncClient whose transport refuses a body larger
than max_bytes while it is still being received.

Compressed responses are refused rather than decoded: the client asks for
'Accept-Encoding: identity', and a server that compresses anyway could
otherwise expand a small download into an unbounded one (a "zip bomb").

Only public httpx APIs are used (AsyncBaseTransport, AsyncByteStream), so
this does not depend on httpx internals.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

DEFAULT_MAX_RESPONSE_BYTES = 10 * 1024 * 1024


class ResponseLimitExceeded(httpx.TransportError):
    """
    The response was larger than allowed, or compressed when it must not be.

    An httpx.TransportError (so also an httpx.HTTPError): code that already
    handles "could not talk to the server" handles this too.
    """


class _CappedStream(httpx.AsyncByteStream):
    def __init__(
        self, inner: httpx.AsyncByteStream, limit: int, request: httpx.Request
    ) -> None:
        self._inner = inner
        self._limit = limit
        self._request = request

    async def __aiter__(self) -> AsyncIterator[bytes]:
        total = 0
        async for chunk in self._inner:
            total += len(chunk)
            if total > self._limit:
                raise ResponseLimitExceeded(
                    f"response body from {self._request.url.host} exceeds "
                    f"{self._limit} bytes",
                    request=self._request,
                )
            yield chunk

    async def aclose(self) -> None:
        await self._inner.aclose()


class CappedTransport(httpx.AsyncBaseTransport):
    """An AsyncHTTPTransport whose responses are bounded to max_bytes."""

    def __init__(
        self, max_bytes: int = DEFAULT_MAX_RESPONSE_BYTES, **transport_kwargs: Any
    ) -> None:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be > 0, got {max_bytes}")
        self._max_bytes = max_bytes
        self._inner = httpx.AsyncHTTPTransport(**transport_kwargs)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)

        encoding = response.headers.get("content-encoding", "").strip().lower()
        if encoding not in ("", "identity"):
            await response.aclose()
            raise ResponseLimitExceeded(
                f"{request.url.host} sent a {encoding!r}-encoded body after being "
                "asked for an uncompressed one; refusing to decompress it",
                request=request,
            )

        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > self._max_bytes:
            await response.aclose()
            raise ResponseLimitExceeded(
                f"response body from {request.url.host} declares {declared} "
                f"bytes, over the {self._max_bytes}-byte limit",
                request=request,
            )

        if not isinstance(response.stream, httpx.AsyncByteStream):
            await response.aclose()
            raise ResponseLimitExceeded(
                "transport returned a non-async response stream", request=request
            )
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=_CappedStream(response.stream, self._max_bytes, request),
            extensions=response.extensions,
            request=request,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


def capped_client(
    max_bytes: int = DEFAULT_MAX_RESPONSE_BYTES, **client_kwargs: Any
) -> httpx.AsyncClient:
    """
    An httpx.AsyncClient that refuses response bodies over max_bytes.

    Accepts the usual httpx.AsyncClient keyword arguments (timeout, headers,
    ...). Requests ask for an uncompressed response; a compressed one is
    refused instead of being decompressed without bound.
    """
    headers = {"Accept-Encoding": "identity"}
    headers.update(client_kwargs.pop("headers", None) or {})
    return httpx.AsyncClient(
        transport=CappedTransport(max_bytes), headers=headers, **client_kwargs
    )


__all__ = [
    "DEFAULT_MAX_RESPONSE_BYTES",
    "CappedTransport",
    "ResponseLimitExceeded",
    "capped_client",
]

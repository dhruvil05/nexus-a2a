"""
tests/test_retry_config.py

RetryConfig(max_retries < 1) is rejected up front. The value counts attempts
including the first, so 0 used to send no request and then raise
AgentUnreachableError("... unreachable: unknown").
"""

from __future__ import annotations

import pytest
import respx

from nexus_a2a import A2AHttpClient, AgentUnreachableError, RetryConfig
from nexus_a2a.config import ConfigError, NexusConfig
from nexus_a2a.models.task import Message

AGENT = "http://agent.test:8001"


# ── RetryConfig.max_retries ───────────────────────────────────────────────────


class TestRetryConfigValidation:
    @pytest.mark.parametrize("value", [0, -1])
    def test_below_one_raises_value_error(self, value: int) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            RetryConfig(max_retries=value)

    def test_error_explains_the_meaning(self) -> None:
        with pytest.raises(ValueError, match="max_retries=1 means one request"):
            RetryConfig(max_retries=0)

    @pytest.mark.parametrize("value", [True, 1.5, "3"])
    def test_non_int_raises_type_error(self, value: object) -> None:
        with pytest.raises(TypeError):
            RetryConfig(max_retries=value)  # type: ignore[arg-type]

    def test_one_is_a_single_attempt(self) -> None:
        assert RetryConfig(max_retries=1).max_retries == 1

    async def test_one_sends_exactly_one_request(self) -> None:
        with respx.mock() as router:
            route = router.post(f"{AGENT}/").respond(503)
            async with A2AHttpClient(AGENT, retry=RetryConfig(max_retries=1)) as c:
                with pytest.raises(AgentUnreachableError, match="HTTP 503"):
                    await c.send_message(Message.user_text("hi"))
        assert route.call_count == 1

    def test_config_rejects_zero(self) -> None:
        raw = {
            "agent": {"name": "a", "url": AGENT},
            "reliability": {"max_retries": 0},
        }
        with pytest.raises(ConfigError, match="at least 1"):
            NexusConfig.from_dict(raw)

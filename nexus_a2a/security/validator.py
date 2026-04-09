"""
nexus_a2a/security/validator.py

PayloadValidator — validates and sanitises every inbound Message before
it reaches agent logic.

Checks (in order):
  1. Size limit  — total serialised payload must not exceed max_bytes.
  2. Part count  — message must not have more parts than max_parts.
  3. Schema      — every Part must conform to the Part model (via Pydantic).
  4. Content     — text parts are stripped of leading/trailing whitespace
                   and must not be blank after stripping.

Why this matters:
  - Prevents memory exhaustion from oversized payloads.
  - Rejects structurally malformed messages before agent logic runs.
  - Strips accidental whitespace so agents receive clean inputs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from pydantic import ValidationError

from typing import Dict, Any

from nexus_a2a.models.task import Message, Part, PartType

logger = logging.getLogger(__name__)

# Sensible production defaults
_DEFAULT_MAX_BYTES = 1 * 1024 * 1024   # 1 MB
_DEFAULT_MAX_PARTS = 20


# ── Exceptions ────────────────────────────────────────────────────────────────

class ValidationError_(Exception):
    """Base class for all payload validation errors."""


class PayloadTooLargeError(ValidationError_):
    """Raised when the serialised payload exceeds the size limit."""

    def __init__(self, size: int, limit: int) -> None:
        super().__init__(
            f"Payload size {size:,} bytes exceeds the limit of {limit:,} bytes."
        )
        self.size  = size
        self.limit = limit


class TooManyPartsError(ValidationError_):
    """Raised when a message contains more parts than allowed."""

    def __init__(self, count: int, limit: int) -> None:
        super().__init__(
            f"Message has {count} parts, exceeding the limit of {limit}."
        )
        self.count = count
        self.limit = limit


class InvalidPartError(ValidationError_):
    """Raised when a Part fails Pydantic schema validation."""

    def __init__(self, index: int, reason: str) -> None:
        super().__init__(f"Part at index {index} is invalid: {reason}")
        self.index  = index
        self.reason = reason


class BlankTextPartError(ValidationError_):
    """Raised when a text Part is empty or whitespace-only after stripping."""

    def __init__(self, index: int) -> None:
        super().__init__(
            f"Text Part at index {index} is blank after stripping whitespace."
        )
        self.index = index


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ValidatorConfig:
    """
    Tunable limits for the PayloadValidator.

    Args:
        max_bytes: Maximum allowed size of the serialised Message in bytes.
                   Default: 1 MB.
        max_parts: Maximum number of Parts a single Message may contain.
                   Default: 20.
        strip_text: If True, strip whitespace from text Part content.
                    Default: True.
    """
    max_bytes:  int  = _DEFAULT_MAX_BYTES
    max_parts:  int  = _DEFAULT_MAX_PARTS
    strip_text: bool = True

    def __post_init__(self) -> None:
        if self.max_bytes <= 0:
            raise ValueError(f"max_bytes must be > 0, got {self.max_bytes}")
        if self.max_parts <= 0:
            raise ValueError(f"max_parts must be > 0, got {self.max_parts}")


# ── PayloadValidator ──────────────────────────────────────────────────────────

class PayloadValidator:
    """
    Validates and sanitises inbound Messages before they reach agent logic.

    Usage:
        validator = PayloadValidator()

        # Validate a message — raises on any violation
        clean_message = validator.validate(message)

        # Use a custom config
        validator = PayloadValidator(
            config=ValidatorConfig(max_bytes=512_000, max_parts=5)
        )

    Args:
        config: Validation limits. Defaults to 1 MB / 20 parts.
    """

    def __init__(self, config: ValidatorConfig | None = None) -> None:
        self._config = config or ValidatorConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(self, message: Message) -> Message:
        """
        Validate and sanitise a Message.

        Runs all checks in order. Sanitisation (stripping whitespace) happens
        in-place on the message object so the caller gets back a clean Message.

        Args:
            message: The inbound Message to validate.

        Returns:
            The same Message object, sanitised.

        Raises:
            PayloadTooLargeError: Serialised size exceeds max_bytes.
            TooManyPartsError:    More parts than max_parts.
            InvalidPartError:     A Part fails schema validation.
            BlankTextPartError:   A text Part is empty after stripping.
        """
        self._check_size(message)
        self._check_part_count(message)
        self._validate_parts(message)
        if self._config.strip_text:
            self._sanitise_text_parts(message)
        self._check_blank_text_parts(message)

        logger.debug(
            "Payload validated: parts=%d size~=%d bytes",
            len(message.parts),
            self._serialised_size(message),
        )
        return message

    def validate_dict(self, raw: Dict[str, Any]) -> Message:
        """
        Parse a raw dict into a Message and validate it.
        Useful when receiving JSON payloads from HTTP requests.

        Args:
            raw: A dict that should conform to the Message schema.

        Returns:
            A validated, sanitised Message.

        Raises:
            InvalidPartError:  Dict does not conform to Message schema.
            + all errors from validate().
        """
        try:
            message = Message.model_validate(raw)
        except ValidationError as exc:
            raise InvalidPartError(index=-1, reason=str(exc)) from exc
        return self.validate(message)

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_size(self, message: Message) -> None:
        """Reject if serialised size exceeds max_bytes."""
        size = self._serialised_size(message)
        if size > self._config.max_bytes:
            raise PayloadTooLargeError(size, self._config.max_bytes)

    def _check_part_count(self, message: Message) -> None:
        """Reject if message has more parts than max_parts."""
        count = len(message.parts)
        if count > self._config.max_parts:
            raise TooManyPartsError(count, self._config.max_parts)

    def _validate_parts(self, message: Message) -> None:
        """
        Re-validate every Part through Pydantic.
        Catches cases where a Part was mutated after initial creation.
        """
        for i, part in enumerate(message.parts):
            try:
                Part.model_validate(part.model_dump())
            except ValidationError as exc:
                raise InvalidPartError(i, str(exc)) from exc

    def _sanitise_text_parts(self, message: Message) -> None:
        """Strip leading/trailing whitespace from all TEXT part content."""
        for part in message.parts:
            if part.type == PartType.TEXT and isinstance(part.content, str):
                part.content = part.content.strip()

    def _check_blank_text_parts(self, message: Message) -> None:
        """Reject if any TEXT part is empty after sanitisation."""
        for i, part in enumerate(message.parts):
            if part.type == PartType.TEXT:
                if not str(part.content).strip():
                    raise BlankTextPartError(i)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _serialised_size(message: Message) -> int:
        """Return the byte size of the JSON-serialised message."""
        return len(json.dumps(message.model_dump(mode="json")).encode("utf-8"))

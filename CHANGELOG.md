# Changelog

All notable changes to **nexus-a2a** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.3.0] — Observability + CLI — Unreleased

### Added
- **`nexus` CLI** (`cli/main.py`) — Click-based entry point wired into `pyproject.toml`
  as `[project.scripts] nexus = "nexus_a2a.cli.main:cli"`.
  Global flags: `--config`, `--verbose / -v`, `--format json|table`.
- **`nexus run`** — Start the agent server defined in `nexus.toml`.
  Accepts `--host`, `--port`, `--module MODULE:CLASS` overrides.
- **`nexus ping <url>`** — Fetch AgentCard + hit `/health`. Reports agent name,
  version, skills count, round-trip latency (ms), health status.
- **`nexus inspect <url>`** — Pretty-print full AgentCard: capabilities, auth scheme,
  input/output modes, all skills with tags and example counts.
- **`nexus status --network`** — Table of all registered agents: name, URL,
  health (✓/✗), task queue depth, DLQ pending count, last seen timestamp.
  Summary row: total, healthy, unhealthy, total DLQ pending.
- **`nexus trace <task_id>`** — Render full distributed call tree with per-hop latency,
  status icons, error messages. Slow hops (>500 ms) highlighted in yellow.
  Supports `--agent URL` for remote TraceStore queries and `--format json`.
- **`nexus replay --failed`** — Query DLQ with filters (`--skill`, `--last DURATION`,
  `--dry-run`, `--yes`). Preview table → confirm → progress bar → summary.
- **`cli/output.py`** — Rich terminal rendering: tables, trees, progress bars,
  status icons. All display logic in one place; commands stay lean.
- **Integration tests** (`tests/integration/`) — Real end-to-end tests using
  in-process Starlette servers on random ports. No mocking of HTTP layer.
  - `test_two_agents.py` — Two agents exchange a task; AgentCard discovery;
    trace ID propagation; `find_by_skill()` routing.
  - `test_sequential_pipeline.py` — Three-agent sequential chain; output chaining;
    step timing; `stop_on_error` halts pipeline; parallel workflow concurrency.
  - `test_failure_recovery.py` — Failed task → DLQ; DLQ replay against healthy
    agent; circuit breaker opens after N failures; `INPUT_REQUIRED` state;
    graceful shutdown drains; concurrent task handling.
- **`CHANGELOG.md`** — This file. Version history in Keep a Changelog format
  with migration notes for breaking changes.

### Changed
- `pyproject.toml` — Added `[project.scripts]` entry for `nexus` CLI.
  Added `click` and `rich` to core dependencies.
  Added `pytest-asyncio` and `uvicorn` to `[dev]` extras.
- `__init__.py` — `__version__` bumped to `"1.3.0"`.
- `README.md` — Updated with CLI quickstart, integration test instructions,
  `nexus.toml` observability section documentation.

### Migration notes (v1.2 → v1.3)
- No breaking API changes. All public classes, decorators, and config keys
  from v1.2 remain unchanged.
- New CLI requires `click>=8.1` and `rich>=13.0`. Install via:
  ```
  pip install "nexus-a2a[cli]"
  ```
  or add `cli` to your extras in `pyproject.toml`:
  ```toml
  dependencies = ["nexus-a2a[cli]"]
  ```
- Integration tests require `pytest-asyncio>=0.23` and `uvicorn>=0.29`.
  Install the `dev` extra: `pip install "nexus-a2a[dev]"`.

---

## [1.2.0] — Enforcement + Infrastructure

### Added
- **`core/graceful_shutdown.py`** — `SIGTERM`/`SIGINT` handler. Stops accepting new
  tasks, waits for `WORKING` tasks up to `drain_timeout`, publishes shutdown events.
- **`core/agent_server.py`** — Built-in Starlette server with `GET /health`,
  `GET /ready`, `GET /metrics` (Prometheus text format). Kubernetes-compatible.
- **`security/mtls.py`** — Mutual TLS for agent-to-agent calls. File paths or
  in-memory PEM bytes. `from_env()` reads `NEXUS_MTLS_*` environment variables.
- **`storage/postgres_store.py`** — `asyncpg` backend. Three tables: `tasks`,
  `task_messages`, `task_artifacts`. SQL-queryable for compliance and analytics.
  Connection pooling. Migration support via versioned SQL scripts.
- **`config.py`** — `nexus.toml` parser. `AgentNetwork.from_config()` wires
  everything from one TOML file. `NEXUS_*` env var overrides for containers.
- **`.github/workflows/ci.yml`** — PR gate: `ruff` + `mypy --strict` + `pytest`
  + `pip-audit`. Matrix: Python 3.11 + 3.12.
- **`.github/workflows/publish.yml`** — Tag push `v*.*.*` → auto-publish PyPI via
  OIDC Trusted Publisher. No stored API tokens.
- **`.github/workflows/security.yml`** — Weekly `pip-audit` + `safety` CVE scan.
  Creates GitHub Issue on `HIGH`/`CRITICAL` severity findings.
- **`nexus.toml`** — Zero-config project file added to repository root.

### Changed
- `__init__.py` — `__version__` bumped to `"1.2.0"`.

### Migration notes (v1.1 → v1.2)
- No breaking API changes.
- `AgentNetwork.from_config("nexus.toml")` is now the recommended setup method.
- Postgres store requires `asyncpg` extra: `pip install "nexus-a2a[postgres]"`.
- mTLS requires valid certificate files or PEM strings. Self-signed certs work
  in dev via `verify=False` on `MutualTLSConfig`.

---

## [1.1.0] — Critical Production Fixes

### Added
- **`core/task_manager.py`** — Background asyncio watchdog auto-fails tasks stuck
  in `WORKING` beyond `timeout_sec`.
- **`core/input_handler.py`** — True pause/resume for `INPUT_REQUIRED` using
  `asyncio.Event`. No polling, no threads, no CPU waste.
- **`core/dead_letter.py`** — `DeadLetterQueue`: `replay()`, `replay_all()`,
  `replay_where()` with filters. `@dlq.on_failure` hooks. Exponential backoff.
  `max_queue_size` cap. `clear_replayed()` cleanup.
- **`transport/http_client.py`** — `CircuitBreaker` (`CLOSED`/`OPEN`/`HALF_OPEN`).
  Retry on `ConnectError` AND `5xx` status codes. Exponential backoff with 20% jitter.
  Automatic `X-Nexus-Trace-ID` injection.
- **`transport/tracing.py`** — `Tracer.inject()` / `extract()` for header propagation.
  `Tracer.span()` async context manager. `TraceStore` in-memory with `max_traces` cap.
  Optional OTEL span export. `format_tree()` for CLI display.
- **`security/capability_guard.py`** — Enforces declared capabilities match
  implementation. Three modes: `strict` (raise), `warn` (log), `off`.

### Fixed
- `Task` model: `FAILED → INPUT_REQUIRED` transition now permitted (was missing from
  `_TRANSITIONS`).
- `A2AHttpClient._require_client()` error message now references "async context manager"
  for clarity.
- `test_phase2.py`: `RetryConfig` import path corrected; mock responses include
  `status_code=200`.

### Changed
- `__init__.py` — `__version__` bumped to `"1.1.0"`.

### Migration notes (v1.0 → v1.1)
- `A2AHttpClient` now accepts `RetryConfig` via `retry=` parameter (replaces positional
  `max_retries`). Update call sites:
  ```python
  # Before
  A2AHttpClient(url, max_retries=3)
  # After
  A2AHttpClient(url, retry=RetryConfig(max_retries=3))
  ```

---

## [1.0.0] — Adapters + Observability

### Added
- `adapters/langgraph.py` — `LangGraphAdapter`
- `adapters/crewai.py` — `CrewAIAdapter`
- `adapters/google_adk.py` — `GoogleADKAdapter`
- `adapters/autogen.py` — `AutoGenAdapter`
- `storage/redis_store.py` — `RedisTaskStore` (`redis.asyncio`). `SCAN` not `KEYS`.
  Configurable TTL. Async context manager lifecycle.
- `storage/audit_logger.py` — `AuditLogger` — NDJSON to any `TextIO`. 8 event types.
  In-memory buffer with configurable max size.
- `storage/metrics.py` — `MetricsCollector` — Standalone or OTEL mode. Tracks tasks,
  per-agent latency (avg + p99), errors, rate limits, auth failures.
- `tests/test_phase5.py` — Adapter, Redis, AuditLogger, MetricsCollector tests.

### Changed
- `__init__.py` — `__version__` bumped to `"1.0.0"`. All public symbols exported.

---

## [0.4.0] — Orchestration + Streaming

### Added
- `core/orchestrator.py` — `sequential()`, `parallel()`, `dag()` workflow modes.
  DFS cycle detection for DAG. `OrchestratorResult` + `StepResult` with timing.
- `transport/sse.py` — `SSEStreamer` (async generator, client-side) + `SSEFormatter`
  (server-side static methods). Event types: `task_created`, `task_status`,
  `artifact_chunk`, `artifact_complete`, `done`, `error`, `heartbeat`.
- `transport/webhook.py` — `WebhookDispatcher` — HMAC-SHA256 signed delivery.
  Exponential backoff retries. No retry on 4xx. `dispatch_silent()`. `verify_signature()`.
- `network.py` — `AgentNetwork` top-level API. `EventBus` pub/sub.

---

## [0.3.0] — Security Layer

### Added
- `security/auth.py` — `AuthManager`: `NONE`, `API_KEY` (constant-time compare), `JWT`
  (`python-jose` HS256). `issue_jwt()`. `build_auth_headers()`.
- `security/trust.py` — `TrustBoundary`: default-deny permission matrix. `allow()`,
  `block()`, `revoke()`, `is_allowed()`. `fnmatch` wildcard support.
- `security/rate_limiter.py` — `RateLimiter`: token bucket per agent. `RateLimitConfig`.
  `RateLimitError` with `retry_after`.
- `security/validator.py` — `PayloadValidator`: size, part count, Pydantic re-validation,
  blank text detection. `validate_dict()` one-call helper.

---

## [0.2.0] — Core Engine

### Added
- `storage/task_store.py` — `AbstractTaskStore` ABC + `InMemoryTaskStore`.
- `core/task_manager.py` — 10-method task lifecycle manager. `TaskNotFoundError`,
  `TaskAlreadyDoneError`.
- `transport/http_client.py` — JSON-RPC 2.0 async HTTP client using `httpx`.
  Connection pooling. Retry on `ConnectError` and `TimeoutException`.
- `core/registry.py` — `AgentRegistry`: `register_url()`, `register_card()`,
  health checks, TTL refresh, `find_by_skill()`.

---

## [0.1.0] — Foundation

### Added
- `models/agent.py` — `AgentCard`, `AgentSkill`, `AgentCapabilities`,
  `AgentAuthentication`, `AuthScheme`, `InputMode`, `OutputMode`.
- `models/task.py` — `Task` (state machine), `TaskState`, `Message`, `Part`,
  `PartType`, `Artifact`.
- `decorators.py` — `@agent` decorator. Supports both `@agent` and `@agent(...)` forms.
  Auto-generates `AgentCard`. Validates `async def run()` present.
- `pyproject.toml` — `hatchling` build system. Optional extras: `redis`, `postgres`,
  `otel`, `dev`.
- `tests/test_models.py` — 30+ model and state machine tests.
- `tests/test_decorator.py` — `@agent` decorator tests.

[1.3.0]: https://github.com/your-org/nexus-a2a/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/your-org/nexus-a2a/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/your-org/nexus-a2a/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/your-org/nexus-a2a/compare/v0.4.0...v1.0.0
[0.4.0]: https://github.com/your-org/nexus-a2a/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/your-org/nexus-a2a/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/your-org/nexus-a2a/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/your-org/nexus-a2a/releases/tag/v0.1.0
# Changelog

All notable changes to **nexus-a2a** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.8.0] — Production scale — Unreleased

1.7.0 finished the protocol. This release is about what happens when you run
more than one copy of it, and what survives a restart.

### Added — durable Dead Letter Queue
- **`AbstractDLQStore`, `InMemoryDLQStore`, `RedisDLQStore`** — the DLQ can now
  persist. `DeadLetterQueue(store=...)` writes entries through; `load()`
  rehydrates them at startup.
- `DLQEntry.to_storage_dict()` / `from_storage_dict()` — a full round trip
  including the failed Task. `to_dict()` stays the display summary and
  deliberately omits it.
- `DeadLetterQueue.is_durable`, `.store`, `.refresh()`, `.purge_replayed()`.
- `AgentNetwork(dlq_store=...)`.
- `tests/test_dlq_persistence.py` (38 tests).

**Why it mattered:** the DLQ lived in a process-local dict, so a crash or a
redeploy lost every failed task in it — the one piece of state you least want
to lose, because an entry exists precisely because work did NOT complete and
replaying it is the only way that work ever happens.

The local view is kept as a write-through cache because every read accessor is
synchronous (`count()`, `pending_entries()`, `summary()`) and AgentServer's
`/dlq` and `/metrics`, the CLI, and anything written against them depend on
that. Eviction removes from the store too, so `load()` cannot resurrect entries
the queue already dropped.

`clear_replayed()` is synchronous and therefore cannot reach an async store;
`purge_replayed()` is its durable counterpart. Both are documented as such.

### Added — distributed rate limiting
- **`RedisRateLimiter`** — one token bucket per agent in Redis, so a limit
  holds across replicas. A drop-in for `RateLimiter` anywhere one is accepted,
  including `SecurityMiddleware`.
- **`AbstractRateLimiter`** — the shared interface both implement, so the two
  are interchangeable. `SecurityMiddleware` now types against it.
- `is_distributed` on both, so a production check can assert it.
- `RedisRateLimiter(client=...)` to share an existing pool.
- `tests/test_redis_rate_limiter.py` (31 tests).

**Why it mattered:** `RateLimiter` keeps buckets in a process-local dict, so
three replicas configured for 10 req/s actually allowed 30. The limit scaled
with the deployment, silently, which is the opposite of what a limit is for.

Refill-and-consume runs as a **Lua script** so it is atomic: doing that
read-modify-write from Python races, and two replicas both seeing one token
left would both proceed. The script reads the clock from Redis (`TIME`) rather
than the caller, so a skewed replica cannot rewind a bucket and mint itself
tokens. Bucket keys carry a TTL covering a full refill, so idle agents expire
instead of accumulating and none is ever recreated full early.

Requires Redis 7+ (the script calls `TIME`).

### Fixed
- `rate_limiter.py` pointed readers at "RedisRateLimiter (Phase 5)", a class
  that had never been written. It now exists, and the note points at it.
- Three circuit-breaker tests used a 10ms recovery timeout with a 20ms sleep —
  a 10ms margin that scheduler jitter clears on a loaded machine. One failed
  intermittently in full-suite runs while passing in isolation. Widened to a
  100ms margin.

### Testing
`fakeredis[lua]` is a dev dependency, so the Lua script is executed in the test
suite rather than skipped for want of a server. Two limiter instances sharing
one client stand in for two replicas — which is exactly what they are to Redis.
A concurrency test asserts that 20 simultaneous consumers against a 5-token
bucket get exactly 5 successes.

### Added — asymmetric JWT and JWKS
- **`AgentCredentialConfig.jwt_public_key` / `jwt_private_key` / `jwks_url` /
  `jwt_algorithm` / `jwt_issuer`** — RS256/384/512, PS256/384/512 and
  ES256/384/512 alongside the existing HS256.
- **`JWKSClient`** (`security/jwks.py`) — fetches an issuer's published key
  set, caches it, and resolves a token's signing key by `kid`. Rotation
  becomes a publish instead of a coordinated secret swap.
- `iss` is now issued and validated when `jwt_issuer` is set.
- `tests/test_jwks_auth.py` (41 tests).
- New `jwks` extra: `pip install nexus-a2a[jwks]`. HS256 needs no crypto
  library, so `cryptography` stays OUT of the core tree, which remains 20
  packages with no known vulnerabilities.

**Why it mattered:** with HS256 the verifier holds the same secret as the
signer, so any agent that can CHECK a peer's token can also MINT one. In a
network of more than two parties every verifier is also a forger. A key pair
splits those roles.

### Security — algorithm confusion is structurally prevented
An RSA public key is public by definition. A verifier that accepted HS256
alongside RS256 would therefore accept a token signed with that public key used
as an HMAC secret — the classic JWT forgery.

The acceptable algorithms are now derived from the CONFIGURED key material and
never from the token's `alg` header, and the symmetric and asymmetric families
never overlap. `AgentCredentialConfig` rejects, at registration time, any
config that sets more than one of `jwt_secret` / `jwt_public_key` / `jwks_url`,
and refuses a symmetric `jwt_algorithm` on an asymmetric key.

With JWKS, `kid` selects only WHICH public key to try; it cannot widen the
algorithm set.

The test for this hand-rolls the forged token, because PyJWT refuses to encode
or decode a PEM as an HMAC secret — an attacker writes the bytes directly, so
asking PyJWT would have proved nothing. `alg: none` and the mirror case (an
HS256 verifier handed an RS256 token) are covered too.

### Notes — JWKS
- Key sets are cached for `cache_ttl` (default 300s). An unknown `kid` triggers
  one refresh so rotation is picked up early, but refreshes are rate-limited:
  otherwise a stream of tokens bearing junk kids would turn the agent into a
  traffic amplifier aimed at the issuer's JWKS endpoint.
- A single unusable key in a document is skipped with a warning rather than
  failing the whole set, so one bad entry cannot lock out every other key.
- A JWKS outage surfaces as an authentication failure, not a crash.

### Added — durable push targets
- **`AbstractPushStore`, `InMemoryPushStore`, `RedisPushStore`** —
  `A2AServer(push_store=...)` persists where each task's updates get POSTed.
- `tests/test_push_persistence.py` (23 tests).

**Why it mattered:** registering a webhook is what a caller does precisely
because they will NOT sit and wait. The target lived in a dict on the
A2AServer instance, so the caller got nothing if the agent restarted mid-task,
and nothing if the follow-up landed on a different replica — both the normal
case in production.

`_register_push()` and `_notify()` became async, which every call site already
was. Unlike the DLQ there were no synchronous public accessors to preserve, so
this is a plain store rather than a write-through cache.

A push config carries the caller's `token`, the shared secret their receiver
checks callbacks against. Persisting it means it is at rest in the backing
store, so point `RedisPushStore` at a Redis you would keep credentials in.
Targets are released when a task reaches a terminal state, and carry a TTL to
reap tasks that never finish.

A store outage is contained: reading or releasing a target may fail without
failing the task it was reporting on.

### 1.8.0 is feature-complete
Nothing important in the library lives only in one process's memory any more,
and no security control silently weakens as replicas are added:

| | 1.7.0 | 1.8.0 |
|---|---|---|
| Dead Letter Queue | lost on restart | `RedisDLQStore` |
| Push targets | lost on restart | `RedisPushStore` |
| Rate limits | per process (N replicas = N x limit) | shared, atomic |
| JWT | HS256 — every verifier can forge | RS/PS/ES + JWKS |

---

## [1.7.0] — Streaming, multi-turn and push notifications — Unreleased

`AgentCapabilities.streaming` has existed since 1.0 and `SSEStreamer` /
`SSEFormatter` since Phase 4, but nothing ever served a stream — the classes
were constructed nowhere in the package, their only usage being their own
docstrings. An agent could declare `streaming=True` and no client could act on
it. This release serves it.

### Added
- **`message/stream`** — JSON-RPC method that sends a message and returns a
  `text/event-stream` instead of a JSON envelope. Event sequence:
  `task_created` (so the caller gets the task id immediately), one
  `artifact_chunk` per chunk the agent yields, then `task_status` and `done`.
- **`GET /stream?taskId=...`** — the endpoint the existing `SSEStreamer`
  already targeted. Reports a task's state and artifacts, then closes.
- **`A2AHttpClient.stream_message()`** — async iterator of `StreamEvent`,
  stopping at the first terminal event. Works against any A2A agent, streaming
  or not. Retries and the circuit breaker deliberately do NOT apply: a
  half-consumed stream cannot be safely replayed once the caller has seen part
  of the output.
- `iter_sse_events()` and `parse_sse_data()` in `transport/sse.py` — one wire
  parser shared by `SSEStreamer` and `stream_message()`, so both read the
  format identically.
- `tests/test_streaming.py` (53 tests) and 7 real-HTTP streaming tests in
  `tests/integration/test_a2a_roundtrip.py`.

### Agent contract
An agent streams by writing `run()` as an async generator, or by adding a
`stream()` method that is one — the contract `CapabilityGuard` already checked
for. `stream()` wins when both are present.

    @agent(name="Writer", description="...", streaming=True, url=...)
    class Writer:
        async def run(self, task):
            for word in ["hello", " ", "world"]:
                yield word

### Fixed
- **A streaming agent could not be constructed at all.** Both
  `@agent`'s `_has_async_run()` and `A2AServer._resolve_agent()` gated on
  `inspect.iscoroutinefunction()`, which returns **False** for an async
  generator. Any agent written the way `CapabilityGuard` documents was
  rejected — `TypeError` at decoration, `InvalidAgentError` at serve time.
  Both now also accept `inspect.isasyncgenfunction()`.
- **`message/send` broke on a streaming agent.** It did
  `await self._agent_instance.run(task)`; awaiting an async generator raises
  `TypeError`. It now drains the generator and folds the chunks into a single
  result, so a streaming agent serves both methods.

### Cross-compatibility
Neither side has to know how the other is written:
- a **non-streaming** agent called over `message/stream` runs to completion and
  its output is emitted as one `artifact_chunk`;
- a **streaming** agent called over `message/send` has its chunks folded into
  one artifact (strings concatenate; mixed chunks become a list).

### Added — task continuation (multi-turn / INPUT_REQUIRED)
- **`NeedsInput`** — returned from `run()` to pause a task and ask the caller
  for more. The task parks at `INPUT_REQUIRED` with the prompt appended to its
  history instead of completing:

      class Planner:
          async def run(self, task):
              if len(task.history) == 1:
                  return NeedsInput("What is your budget?")
              return f"Plan for {task.history[-1].text()}"

- **`taskId` on `message/send` and `message/stream`** — continues a task that
  is awaiting input instead of creating one. `A2AHttpClient.send_message()` and
  `stream_message()` take a matching `task_id=` argument.
- **`multi_turn` on `@agent`** — the flag could not be set at all before; the
  decorator only passed `streaming` and `push_notifications` to
  `AgentCapabilities`, so every card carried the model default.
- **`A2AServer.input_handler`** — an agent suspended inside
  `InputHandler.wait_for_input()` is resumed by the same wire call, which fires
  its event rather than re-invoking `run()`. `submit_reply()`'s docstring had
  promised a `POST /tasks/{id}/reply` endpoint that never existed.
- `tests/test_task_continuation.py` (36 tests) plus 5 real-HTTP multi-turn
  tests.

### Fixed — multi_turn was a false claim
`AgentCapabilities.multi_turn` defaults to **`True`**, so every card ever
published by this package claimed it, while `message/send` always created a new
task and `COMPLETED` is terminal in the state machine — there was no way to add
a turn to anything. It is now honoured, and an agent that sets
`multi_turn=False` has its continuations refused, so the flag means something
in both directions.

A streaming agent that yielded `NeedsInput` also failed to pause over
`message/send`: the chunk was folded into the result list and the task
completed. Draining now stops at `NeedsInput`, matching the SSE path. (Caught
by the cross-method parametrized test, not by hand.)

### Notes — continuation
- Continuation is stateless: the conversation lives in the task store, not in a
  parked coroutine, so it survives a restart and works from a different client
  or connection. That is why `NeedsInput` is preferred over
  `InputHandler.wait_for_input()` for anything served over HTTP — the latter
  holds the original request open for its timeout.
- Continuing a task that is not awaiting input returns `-32002` naming the
  actual state; an unknown id returns `-32001`.
- Over SSE, a pause emits the prompt as a `message` event before
  `task_status: input_required` and `done`.

### Notes
- Every security refusal happens **before** the stream opens — once SSE starts,
  the status line is already sent and an HTTP status can no longer be
  signalled. Failures after that point arrive as a terminal `error` event and
  the task is marked `FAILED`.
- Responses set `X-Accel-Buffering: no` so nginx does not buffer a stream into
  one lump.
- `GET /stream` reports state rather than following a run in progress: because
  `message/send` executes the agent inline, a task is already terminal by the
  time it can be looked up. Use `message/stream` to watch work as it happens.

### Added — push notifications
- **Webhook registration** — a caller that will not wait registers a callback,
  either with the message that creates the task (`pushNotification` in the
  params) or later with the new `tasks/pushNotificationConfig/set`.
  `tasks/pushNotificationConfig/get` reads it back.
- **`PushNotificationConfig`** (`url`, optional `token`), exported from the
  package root, plus `A2AHttpClient.set_push_config()` / `get_push_config()`
  and a `push_notification=` argument on `send_message()` / `stream_message()`.
- **`A2AServer(push_config=WebhookConfig(...))`** — delivery, retry and signing
  settings. The agent POSTs `task_completed`, `task_failed`, `task_cancelled`
  and `task_input_required`.
- `WebhookConfig.allow_private_urls`, `validate_webhook_url()`,
  `WebhookUrlError`, `sign_body()`, and `A2AServer.drain_notifications()`.
- `tests/test_push_notifications.py` (47 tests) and
  `tests/integration/test_push_webhooks.py` (8 tests against a real receiving
  endpoint on another port).

### Fixed — HMAC signatures never validated
`WebhookDispatcher` signed `json.dumps(payload)` and then handed the payload to
httpx as `json=`, which re-encodes with compact separators. The bytes signed
were not the bytes sent, so `verify_signature()` returned False for every
genuine delivery:

    signed:  b'{"event": "task_completed", "task_id": "abc"}'
    sent:    b'{"event":"task_completed","task_id":"abc"}'

The payload is now serialised once, canonically (compact separators, sorted
keys), and those exact bytes are both signed and posted. Signing had never
worked in any released version.

### Security — webhook URLs are an SSRF vector
A webhook URL arrives from whoever called the agent, and the server then makes
a request to it. Without validation a caller could point an agent at cloud
metadata (169.254.169.254), at localhost admin ports, or at hosts inside the
server's private network, and use the agent as a proxy.

Registration now resolves the host and refuses private, loopback, link-local,
reserved, multicast and unspecified addresses, and any scheme other than http
or https. Set `WebhookConfig(allow_private_urls=True)` for local development —
tests do. An agent declaring `push_notifications=False` refuses registration
outright (`-32003`), and a rejected URL returns `-32004`.

Registration failure aborts the whole call rather than running a task whose
updates the caller believes they will receive.

### Notes — push delivery
- Delivery is detached: a slow or dead receiver must not stall the RPC
  response, and retry with backoff can take seconds. Failures are logged, never
  raised — a broken webhook is not the task's problem. `drain_notifications()`
  awaits in-flight deliveries, mainly for tests.
- The registered target is dropped once a task reaches a terminal state, so the
  map does not grow one entry per task forever.
- A config read back never echoes the token, only whether one is set, so
  reading cannot recover a secret someone else chose.
- Targets are held in memory, like the DLQ — they do not survive a restart.

### Known limitations
- Push targets and the DLQ are in-memory; both are candidates for the
  persistence work in 1.8.0.

Every capability flag a card can raise — `streaming`, `push_notifications`,
`multi_turn` — is now honoured by the server.

---

## [1.6.0] — A2AServer: the inbound protocol — Unreleased

nexus-a2a can now BE an agent, not just call one. Through v1.5.0 the library
shipped a complete client (`A2AHttpClient`) with no server behind it: nothing
served `/.well-known/agent-card.json`, nothing handled the JSON-RPC methods,
and `nexus run` imported your agent class and then discarded it. Two nexus-a2a
agents could not actually talk to each other. This release closes that loop.

### Added
- **`A2AServer`** (`core/a2a_server.py`) — serves one `@agent`-decorated class
  over the A2A protocol:
  - `GET /.well-known/agent-card.json` — the discovery document `@agent` built.
  - `POST /` — JSON-RPC 2.0: `message/send`, `tasks/get`, `tasks/cancel`,
    matching `A2AHttpClient` byte for byte.
  - `GET /health`, `GET /ready` — probes; readiness checks the task store.
  Accepts an agent class or an instance, derives host/port from the agent
  card's URL, and runs the task through `TaskManager` so state transitions,
  persistence and the store backends all apply.
- **`SecurityMiddleware`** (`security/middleware.py`) — the wiring that makes
  the security layer actually enforce. `AuthManager`, `TrustBoundary`,
  `RateLimiter` and `PayloadValidator` were correct, tested classes that
  nothing in the library ever called, because there was no inbound request to
  call them on. `A2AServer` now runs them in order on every RPC:
  size → rate limit → auth → trust → payload validation. Every stage is
  optional; the default is a no-op so an agent works out of the box and
  hardens incrementally.
- **`A2AHttpClient(caller_url=...)`** — announces the calling agent's own URL
  in the `X-Nexus-Caller` header. Auth and trust both need to know who is
  calling, and trust rules are `caller -> target`; without it a call is
  anonymous and a server with auth or trust enabled will reject it.
- `CallerIdentity`, `MissingCallerError`, `InvalidAgentError`, `A2AServerError`
  and `PayloadValidator.max_bytes` / `.max_parts` / `.config` accessors.
- 87 tests: `tests/test_a2a_server.py` (dispatch, error envelopes, return-value
  handling, the full security chain) and
  `tests/integration/test_a2a_roundtrip.py` — the first integration tests where
  BOTH halves are nexus-a2a's own code, over real HTTP on real ports. Every
  other integration test still talks to the hand-rolled Starlette mock in
  `conftest.py`, which existed only because there was no server to point at.

### Fixed
- **`nexus run --module pkg:Agent` now actually serves the agent.** It imported
  the class, passed it to `_run_server()`, and never referenced it again —
  starting the ops server instead and advertising an agent-card URL that
  returned 404. It now starts an `A2AServer`. With no `--module` it starts the
  ops server only, and says so.
- The README Quickstart works end to end. Steps 2 and 3 (`nexus run`, then
  `client.send_message(...)`) previously could not succeed against anything
  this package produced; both paths are now covered by tests.
- **CLI import cycle removed.** `main.py` registers every sub-command at import
  time and each sub-command imported `NexusContext` / `pass_ctx` back from
  `main.py`, so the graph was `main -> commands.* -> main`. That broke both
  `python -m nexus_a2a.cli.main` and a plain
  `import nexus_a2a.cli.commands.ping`:

      ImportError: cannot import name 'ping' from partially initialized
      module 'nexus_a2a.cli.commands.ping'

  It was also a latent runtime bug beyond the import error — under `-m`,
  `main.py` was executed twice under two names, producing two distinct
  `NexusContext` classes, and `click.make_pass_decorator` matches on class
  identity, so `pass_ctx` would have failed to find the context the root group
  stored.

  The shared state moved to the new **`nexus_a2a/cli/context.py`**, which
  imports nothing from `main`. `NexusContext` and `pass_ctx` are still
  re-exported from `nexus_a2a.cli.main`, and are the same objects, so existing
  imports keep working.

### Added (CLI)
- **`nexus_a2a/cli/__main__.py`** — `python -m nexus_a2a.cli` now works, for
  environments where the `nexus` console script is not on PATH.
- `tests/test_cli.py` — 38 tests covering all three entry points, context
  identity, and a static guard that fails the moment a command module imports
  from `cli.main` again, before anyone hits the ImportError.

### Error model
Transport-level rejections that happen before dispatch return real HTTP status
codes — 401 auth, 403 trust, 413 too large, 429 rate limit (with `Retry-After`),
400 malformed — so proxies and dashboards can see them and the client does not
retry a rejected credential. Application-level failures after dispatch return
JSON-RPC error objects (-32601 method not found, -32602 invalid params, -32001
task not found), which reach the caller as `RemoteAgentError` with the code
intact. An agent that raises is neither: the task is recorded as `FAILED` and
returned as a normal result, so callers can inspect `task.error` and the Dead
Letter Queue can capture it.

### Notes
- Ops endpoints (`/metrics`, `/info`, `/dlq`, traces) remain in `AgentServer`,
  designed to run on a separate port — the standard app-port / admin-port
  split. `A2AServer` serves the protocol plus probes.
- `run()` may return `None`, `str`, `dict`/`list`, `Artifact`, `Message`, or an
  `AdapterResult`; each maps onto the task result predictably. An
  `AdapterResult` carrying `.error` fails the task.

### Known limitations
- Streaming (`message/stream`) and push notifications are advertised by
  `AgentCapabilities` but not yet served — `CapabilityGuard` can still check
  them on remote cards.
- `INPUT_REQUIRED` is reachable through `TaskManager` but `A2AServer` has no
  wire method for supplying the follow-up input yet.

---

## [1.5.0] — Security hardening — Unreleased

Security release. Two fixes intentionally change runtime behaviour; both are
noted under **Changed (breaking)** with the flag that restores the old default.

### Security
- **CRITICAL — Admin endpoints no longer unauthenticated.** `GET /info`,
  `GET /traces/{id}`, `GET /dlq` and `POST /dlq/replay` were served with no
  authentication on a server whose default bind is `0.0.0.0:8080`. Anyone who
  could reach the port could re-execute every failed task in the Dead Letter
  Queue (duplicate writes, repeated LLM spend, an amplification primitive) and
  read task payloads, error strings and the full agent topology.
  They now require a token supplied via `AgentServer(admin_token=...)` or the
  `NEXUS_ADMIN_TOKEN` environment variable, sent as `X-Admin-Token: <token>` or
  `Authorization: Bearer <token>`, compared with `hmac.compare_digest`.
  With no token configured the endpoints return `403` and are effectively off.
  `/health`, `/ready` and `/metrics` remain public so Kubernetes probes and
  Prometheus scraping keep working unauthenticated.
- **HIGH — `AuthManager` no longer fails open.** `_get_config()` returned
  `AuthScheme.NONE` for any agent URL that was not registered, so a request
  presenting an unregistered URL — a typo, a trailing-slash variant, or an
  attacker-chosen string — bypassed authentication entirely. Unregistered
  agents now raise the new `UnknownAgentError`.
- **MEDIUM — JWT `aud` claim is now actually validated.** `jwt_audience` was
  passed inside the `options` dict, where it is not a recognised key, so it was
  silently discarded and a token with no `aud` claim passed validation. It is
  now passed as the top-level `audience` parameter.
- **MEDIUM — Payload size is checked before parsing.** `validate_dict()` could
  only reject an oversized body after Pydantic had already deserialised it into
  memory, so the documented memory-exhaustion protection did not hold. New
  `PayloadValidator.validate_raw(body)` enforces `max_bytes` against the raw
  bytes first. Prefer it for anything arriving off the network.
- **MEDIUM — `RateLimiter` bucket map is bounded.** Buckets were created lazily
  per agent URL into an uncapped dict with no eviction, so attacker-influenced
  URLs could grow it without bound. Now an LRU `OrderedDict` capped by the new
  `max_tracked` argument (default 10,000).
- **MEDIUM — `TrustBoundary` matching is platform-independent.** Rules used
  `fnmatch.fnmatch()`, which applies `os.path.normcase()` and is therefore
  case-insensitive and slash-rewriting on Windows but case-sensitive on Linux —
  the same ACL produced different decisions per OS. Now `fnmatch.fnmatchcase()`.
- **Dependencies — dropped `python-jose` for `PyJWT`.** `python-jose` pulled in
  `ecdsa`, `rsa` and `pyasn1` transitively; `ecdsa` carries PYSEC-2026-1325
  (Minerva timing attack), which upstream has declared unfixable in pure Python.
  Only HS256 was ever used, so none of them were needed. Removes 5 of the 9
  advisories `pip-audit` reported against the dependency tree.

### Changed (breaking)
- `AuthManager.verify()` raises `UnknownAgentError` for unregistered agents.
  Pass `AuthManager(allow_unregistered=True)` for the pre-1.5.0 fail-open
  behaviour (logs a warning; development only).
  `build_auth_headers()` is unaffected and still returns `{}` for unknown
  agents — fail-closed applies to inbound verification, not outbound headers.
- `AgentServer` admin endpoints return `403` unless a token is configured.
- `google-adk` moved from a required dependency to the `adk` extra. It is
  imported lazily — exactly like the CrewAI, LangGraph and AutoGen adapters,
  none of which were ever required — but was pulling ~40 transitive packages
  into every install. Use `pip install nexus-a2a[adk]` (it is also in `[all]`).

### Fixed
- `TrustBoundary` skill ACLs are deterministic. `_check_skill()` returned on the
  first matching rule while iterating an insertion-ordered dict, so when a
  wildcard rule and a specific rule both matched, the outcome depended on
  registration order. All matching rules are now considered; since rules are
  additive grants, access is permitted if any matching rule grants the skill.
- `__version__` is derived from installed package metadata instead of being
  hardcoded. It had drifted to `1.2.0` while `pyproject.toml` said `1.4.1`,
  which would have failed the publish workflow's version-equality gate.
- `nexus run` no longer advertises a `/.well-known/agent-card.json` URL that
  the server does not serve.

### Added
- `AgentServer(admin_token=...)` and `NEXUS_ADMIN_TOKEN`.
- `AuthManager(allow_unregistered=...)` and `UnknownAgentError` (exported).
- `PayloadValidator.validate_raw()`.
- `RateLimiter(max_tracked=...)`.
- Tests covering admin-endpoint authorisation and auth fail-closed behaviour.

### Known limitations (as of this release)
- The security classes (`AuthManager`, `TrustBoundary`, `RateLimiter`,
  `PayloadValidator`, `CapabilityGuard`) are still opt-in building blocks that
  nothing in the library calls automatically, because there is no inbound A2A
  server to enforce them on.

  **Resolved in 1.6.0** by `A2AServer` + `SecurityMiddleware`. 1.5.0 was never
  published, so these changes ship inside 1.6.0 and this limitation never
  reached a release.

---

## [1.4.0] — Observability + CLI — Unreleased

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
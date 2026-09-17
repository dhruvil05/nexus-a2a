# Migrating

## 1.8 → 1.9

1.9.0 is the last minor release before 2.0. It changes no public signatures,
but several things now behave the way the documentation always said they did.
Check these before upgrading a running deployment.

### Behaviour that changed

**`nexus run` now applies `nexus.toml`.** Before 1.9.0, `nexus run` ignored
every section of the file, so `[security]`, `[storage]` and the rest had no
effect on a served agent. If your `nexus.toml` sets `auth_scheme`, the agent
now **requires** that credential. A client that was calling it without one
gets `401`.

**The `nexus.toml` auth secret is enforced.** `NexusConfig.build_auth_manager()`
registered the secret under the literal URL `"*"`, which never matched a
caller. From 1.5.0 that meant every caller was refused, and before 1.5.0 every
caller was let in without a check. The secret now applies to every caller that
isn't registered individually. `register_agent("*", config)` now works as
documented, and `AuthManager(default=...)` is the explicit form.

**The agent card advertises the auth that's enforced.** An agent secured through
`SecurityMiddleware` or `[security]` used to publish `"scheme": "none"` from
the decorator default. Clients that trust the card should now send
credentials.

**`InMemoryTaskStore` forgets finished tasks.** A task is evicted one hour after
it completes, fails or is cancelled, and the oldest finished tasks go first once
the store holds 10,000. Running and paused tasks are never evicted. Before this,
a default server kept every task it had handled until the process exited. To
keep the old behaviour:

```python
InMemoryTaskStore(retention_sec=None, max_tasks=None)
```

```toml
[storage]
task_retention_sec = 0   # 0 = keep forever
max_tasks          = 0   # 0 = no cap
```

**`nexus trace` and `nexus replay` look for the ops server first.** They resolve
their target from `--agent`, then `NEXUS_OPS_URL`, then `[ops].url`, and only
then `[agent].url`. They now send the admin token, from `--admin-token`,
`NEXUS_ADMIN_TOKEN` or `[ops].admin_token`. Before, they never sent one, so
both commands got `403` from every real server.

**`nexus status` shows `—` for values a server doesn't report.** It used to
show `0`, which looked like data. In JSON output these values are now `null`.

### New warnings

Each of these marks something that becomes an error in 2.0:

| Warning | Cause | Fix |
|---|---|---|
| `FutureWarning` | An `A2AServer` with no security binds a non-loopback address | Pass `security=`, bind `127.0.0.1`, or set `allow_insecure=True` |
| `FutureWarning` | An HS256 `jwt_secret` shorter than 32 bytes | Use `secrets.token_urlsafe(32)` |
| `DeprecationWarning` | `AuthManager(allow_unregistered=True)` | `AuthManager(default=AgentCredentialConfig(scheme=AuthScheme.NONE))` |
| `ConfigWarning` | A `nexus.toml` key or section that nothing reads | Fix the typo the message suggests |

The `ConfigWarning` may appear on configs that worked before, because some keys
were documented but never read:

- `[security] mtls_cert_file`, `mtls_key_file` and `mtls_ca_file`. Configure
  mTLS with `NEXUS_MTLS_CERT_FILE`, `NEXUS_MTLS_KEY_FILE` and
  `NEXUS_MTLS_CA_FILE` and `MutualTLSConfig.from_env()`.
- `[reliability] dlq_max_size` and `dlq_max_retries`. Pass
  `DeadLetterQueue(max_queue_size=..., max_retries=...)` in code.

Those keys never had any effect, so removing them changes nothing.

## Preparing for 2.0

2.0 turns the four warnings above into errors. There are no other planned
breaking changes.

| In 2.0 | Why |
|---|---|
| An unsecured `A2AServer` on a non-loopback address refuses to start unless `allow_insecure=True` | Anyone who can reach it can run your agent |
| `AuthManager(allow_unregistered=True)` is removed | It fails open |
| HS256 secrets shorter than 32 bytes are rejected | RFC 7518 §3.2 requires the key to be at least as long as the hash. A shorter secret can be brute-forced from one captured token |
| Unknown `nexus.toml` keys are an error | A typo like `auth_schem` silently leaves an agent unsecured |

### Check whether you're ready

Treat the warnings as errors in your test suite. Every 2.0 warning mentions
`nexus-a2a 2.0` in its message, so the filters can match on that text:

```toml
[tool.pytest.ini_options]
filterwarnings = [
  "error:.*nexus-a2a 2.0:FutureWarning",
  "error:.*nexus-a2a 2.0:DeprecationWarning",
  "error::nexus_a2a.config.ConfigWarning",
]
```

If this passes, your code is ready for 2.0.

> Match on the **message**, not the module. A filter like
> `error::FutureWarning:nexus_a2a` looks right but never fires. The warnings
> are reported against *your* code — the line that made the risky call — so a
> module filter set to `nexus_a2a` matches nothing, and the check passes
> without checking anything.

`nexus verify <url>` also checks a running agent for the problems 2.0 will
reject: whether its card matches the auth it enforces, and whether its webhook
registration refuses private addresses.

### Changes considered for 2.0 and dropped

Earlier roadmaps listed three other breaking changes. None of them is planned
any more:

- **`SecurityMiddleware` on by default.** Security can't be switched on without
  credentials to check. The real risk was an open agent reachable from the
  network, and the non-loopback rule above covers exactly that.
- **`caller_url` required on the client.** A shared default credential,
  added in 1.9, doesn't need to know who is calling. Per-caller credentials and
  trust rules already require the caller's identity, and say so when it's
  missing.
- **Durable stores by default.** A durable store needs a connection URL, so it
  can't be the default. The problem it was meant to solve, an in-memory store
  that grew forever, is fixed in 1.9 by retention.

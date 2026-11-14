# nexus-a2a dashboard — HTTP API (v1, nexus-a2a 2.1)

The dashboard is served by the ops server (`AgentServer`) under `/ui`.
Everything below is relative to the ops server's base URL, e.g.
`http://127.0.0.1:9100`.

## Access rules

- Only loopback clients are served unless `[dashboard] allow_remote = true`
  (others get `403 {"error": ...}`). The ops server never trusts
  `X-Forwarded-For`; the TCP peer address is what counts.
- Without `allow_remote`, the `Host` header must name this machine
  (`localhost`, `127.0.0.1`, `[::1]`), which defeats DNS rebinding. With
  `allow_remote`, `allowed_hosts` limits the accepted Host names, and the
  session cookie is marked `Secure` (use HTTPS).
- Disabled (404) unless the dashboard is enabled AND an admin token is set.
- Authentication: a session cookie `nexus_ui_session` (HttpOnly,
  SameSite=Strict, Path=/ui) obtained from `POST /ui/api/login`, or the header
  `X-Admin-Token: <token>` for scripts.
- Every state-changing request (POST) made with the cookie must send
  `X-CSRF-Token: <csrf>` (value from login / session) and a same-origin
  `Origin` header. Header-token (script) requests need no CSRF token.
- Unauthenticated API calls return `401 {"error": "..."}` — the UI shows the
  login screen.
- Failed logins and wrong `X-Admin-Token` values are both throttled (5 per
  minute per client address, then `429`) and recorded as security events.
  Successful logins never count.
- A live stream opened with the cookie ends when that session is logged out
  or expires.
- With `show_content = false`, task `error` fields (lists, detail, issues,
  stream, DLQ) are reduced to the exception type, e.g.
  `"RuntimeError (details hidden: show_content is off)"`, and log lines drop
  exception messages and task error text. Search matches only what is shown.
- Errors are always `{"error": "<message>"}` with a 4xx/5xx status.
- Times are ISO-8601 UTC strings (`"2026-11-11T10:15:30.123456Z"`).

## Pages and static files

| Path | What |
|---|---|
| `GET /ui` and `GET /ui/` | `index.html` (single-page app, hash routing) |
| `GET /ui/static/<file>` | `app.js`, `app.css` (and any other static file) |

Responses carry a strict CSP: `default-src 'self'; script-src 'self';
style-src 'self'; img-src 'self' data:; connect-src 'self';
frame-ancestors 'none'; base-uri 'none'; form-action 'self'`. So: no inline
`<script>`/`<style>`/`style=""` attributes, no CDNs, no `eval`.

## Shared shapes

```jsonc
// TaskSummary
{
  "id": "uuid",
  "state": "submitted|working|input_required|completed|failed|cancelled",
  "skill_id": "echo" | null,
  "context_id": "uuid",
  "owner": "http://caller:1" | null,
  "created_at": "ISO", "updated_at": "ISO",
  "duration_ms": 1234.5 | null,       // terminal tasks only
  "error": "RuntimeError: boom" | null,
  "preview": "first 120 chars of the first message" | null  // null when show_content=false
}

// Part (inside messages / artifacts)
{ "type": "text|data|file", "content": "..." | {...} | null, "length": 42, "redacted": false }

// SecurityEvent
{
  "at": "ISO",
  "kind": "auth_failed|forbidden|rate_limited|payload_too_large|bad_request|webhook_refused|ownership_denied|login_failed|admin_denied",
  "status": 401 | null,
  "caller": "http://claimed-caller:1" | null,
  "client": "127.0.0.1" | null,
  "detail": "human readable reason"
}

// LogLine
{ "at": "ISO", "level": "DEBUG|INFO|WARNING|ERROR|CRITICAL", "logger": "nexus_a2a.core.a2a_server", "message": "..." }

// WebhookEvent
{ "at": "ISO", "task_id": "uuid", "event": "task_completed", "url": "https://host/***", "succeeded": true, "status": 200 | 0, "attempts": 1, "error": null | "..." }
```

## Session

### `GET /ui/api/session`  (no auth needed)
```json
{ "authenticated": false, "csrf": null, "agent": "EchoAgent", "version": "2.1.1",
  "allow_replay": true, "show_content": false }
```

### `POST /ui/api/login`  body `{"token": "<admin token>"}`
- 200 `{"csrf": "<token>"}` + `Set-Cookie: nexus_ui_session=...`
- 401 wrong token, 429 too many attempts (5 per minute per client).

### `POST /ui/api/logout`  (CSRF) → 204

## Overview

### `GET /ui/api/overview`
```jsonc
{
  "agent": { "name": "EchoAgent", "url": "http://127.0.0.1:8001", "version": "1.0.0" },
  "nexus_version": "2.1.1",
  "uptime_seconds": 3600.5,
  "health": { "store": "ok|error", "registry": "ok|no_agents|degraded" },
  "tasks": {
    "by_state": { "submitted": 0, "working": 2, "input_required": 1,
                  "completed": 120, "failed": 3, "cancelled": 1 },
    "in_flight": 3,          // submitted + working + input_required
    "total_seen": 127
  },
  "throughput": {
    "per_minute": [ { "completed": 4, "failed": 0 }, ... ],  // last 30 minutes, oldest first, 30 entries
    "completed_last_5m": 18, "failed_last_5m": 1
  },
  "latency_ms": { "p50": 812.0, "p95": 2400.5, "samples": 120 },   // p50/p95 null when no samples
  "error_rate": 0.05,                                               // null when no terminal tasks in 5 min
  "dlq": { "pending": 2, "total": 3 },
  "security_events_last_hour": 4,
  "webhooks": { "delivered": 10, "failed": 1 }
}
```
The UI polls this every 5 s (and refreshes on stream events).

## Agents

### `GET /ui/api/agents`
```jsonc
{
  "self": {
    "name": "EchoAgent", "description": "...", "url": "http://127.0.0.1:8001", "version": "1.0.0",
    "skills": [ { "id": "echo", "name": "Echo", "description": "..." } ],
    "capabilities": { "streaming": false, "push_notifications": true, "multi_turn": true },
    "authentication": { "scheme": "api_key" } | null
  } | null,
  "peers": [
    { "name": "Summarizer", "url": "http://summ:8002", "healthy": true,
      "last_seen_seconds_ago": 12.5 | null, "skills": ["summarise"] }
  ]
}
```

### `POST /ui/api/agents/check`  (CSRF) → `{"results": {"http://summ:8002": true}}`

## Tasks

### `GET /ui/api/tasks?state=&skill=&q=&limit=100`
All filters optional. `q` matches id, owner, skill or error (case-insensitive).
`limit` max 500. Newest `updated_at` first.
```json
{ "tasks": [ TaskSummary ], "total": 127 }
```

### `GET /ui/api/tasks/{id}`
```jsonc
{
  "task": TaskSummary,
  "history":   [ { "role": "user|agent", "parts": [ Part ] } ],
  "artifacts": [ { "name": "result" | null, "parts": [ Part ] } ],
  "timeline":  [ { "at": "ISO", "event": "created|started|completed|failed|cancelled|input_required|resumed|timed_out|retried" } ],
  "trace": { "trace_id": "...", "hops": [ { "url": "...", "duration_ms": 12.3, "status": "ok|error", "error": null } ] } | null,
  "actions": { "can_cancel": true, "can_retry": false }
}
```
404 when unknown. With `show_content=false` every Part has `content: null,
redacted: true` (type and length kept).

### `POST /ui/api/tasks/{id}/cancel`  (CSRF)
200 `{"task": TaskSummary}`; 409 `{"error": ...}` when already finished.

### `POST /ui/api/tasks/{id}/retry`  (CSRF; needs `allow_replay`)
Starts a NEW task with the original task's first message, skill and context,
running in the background.
202 `{"task": TaskSummary}` (the new task); 403 when replay is disabled;
409 when the task cannot be retried.

## Issues

### `GET /ui/api/issues`
```jsonc
{
  "failed_tasks":     [ TaskSummary ],        // most recent first, max 100
  "dlq":              [ { "task_id": "...", "error": "...", "failed_at": 1731320000.0,
                          "agent_url": "..." | null, "skill_id": "..." | null,
                          "retry_count": 0, "last_retry_at": null, "replayed": false } ],
  "security_events":  [ SecurityEvent ],      // most recent first, max 200
  "webhook_failures": [ WebhookEvent ]        // most recent first, max 100
}
```

### `POST /ui/api/dlq/replay`  (CSRF; needs `allow_replay`)
Body `{"task_ids": ["..."]}` or `{"all": true}`. Duplicate ids run once, at
most 500 entries per request, and an entry already replayed is skipped with
`"error": "already replayed"`.
```json
{ "results": [ { "task_id": "...", "success": true, "error": null, "new_task_id": "..." } ] }
```

## Logs

### `GET /ui/api/logs?level=INFO&q=&limit=200`
`level` = minimum level. `q` substring match (case-insensitive) on message or
logger — paste a task id to see its lines. `limit` max 1000. Newest last.
```json
{ "logs": [ LogLine ] }
```

## Live stream

### `GET /ui/api/stream`  (Server-Sent Events, cookie auth)
```
event: task
data: {"event": "completed", "task": TaskSummary}

event: security
data: SecurityEvent

event: webhook
data: WebhookEvent

event: log
data: LogLine

: ping          (comment heartbeat every 15 s)
```
At most 20 concurrent streams; slow consumers are dropped (the UI should
reconnect with backoff and re-fetch the page's data).

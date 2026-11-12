/* nexus-a2a dashboard — single-page app (no build step, no dependencies).
 *
 * SECURITY: every value returned by the API is attacker-controlled (task
 * content, errors, agent names, log lines come from remote agents). The DOM is
 * only ever built with createElement/createElementNS + textContent via el()/svg()
 * below; never assign HTML strings. URLs from data are rendered as text.
 *
 * Layout of this file:
 *   1. DOM + formatting helpers     5. Router
 *   2. API client + session         6. Live stream (SSE)
 *   3. Shared UI (toast, dialog,    7. Pages: overview, agents, tasks,
 *      badges, tables)                 task detail, issues, logs
 *   4. Login / logout               8. Boot
 */
'use strict';

(function () {
  // ===================================================================
  // 1. DOM + formatting helpers
  // ===================================================================
  const $ = (id) => document.getElementById(id);
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const STATES = ['submitted', 'working', 'input_required', 'completed', 'failed', 'cancelled'];
  const LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];

  function appendChildren(node, children) {
    for (const c of children.flat(Infinity)) {
      if (c == null || c === false) continue;
      node.append(c instanceof Node ? c : String(c)); // strings become text nodes
    }
    return node;
  }

  /** createElement with attributes (set via setAttribute, never HTML) and children. */
  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') node.className = v;
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? '' : String(v));
    }
    return appendChildren(node, children);
  }

  function svg(tag, attrs, ...children) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, String(v));
    return appendChildren(node, children);
  }

  const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);
  const str = (v) => (v == null ? '' : String(v));
  const dash = (v) => (v == null || v === '' ? '—' : String(v));
  const shortId = (id) => str(id).slice(0, 8);
  const taskHref = (id) => '#/tasks/' + encodeURIComponent(str(id));
  const stateLabel = (s) => str(s).replace(/_/g, ' ');

  /** ISO string or epoch seconds (DLQ) -> Date | null */
  function parseTime(v) {
    if (v == null || v === '') return null;
    const d = typeof v === 'number' ? new Date(v * 1000) : new Date(String(v));
    return Number.isNaN(d.getTime()) ? null : d;
  }

  function fmtRel(ms) {
    const s = Math.round((Date.now() - ms) / 1000);
    if (s < 5) return 'just now';
    if (s < 60) return `${s} s ago`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m} min ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h} h ago`;
    return `${Math.floor(h / 24)} d ago`;
  }

  /** <time> element whose text is refreshed every 30 s by updateRelTimes(). */
  function relTime(v) {
    const d = v instanceof Date ? v : parseTime(v);
    if (!d) return el('span', { class: 'muted' }, '—');
    return el('time', { datetime: d.toISOString(), title: d.toLocaleString(), 'data-at': d.getTime() }, fmtRel(d.getTime()));
  }

  function updateRelTimes() {
    for (const t of document.querySelectorAll('time[data-at]')) t.textContent = fmtRel(Number(t.dataset.at));
  }

  function fmtMs(ms) {
    if (ms == null || !Number.isFinite(Number(ms))) return '—';
    ms = Number(ms);
    if (ms < 1000) return `${Math.round(ms)} ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)} s`;
    return `${Math.floor(ms / 60000)} min ${Math.round((ms % 60000) / 1000)} s`;
  }

  function fmtDuration(sec) {
    sec = Math.floor(num(sec));
    const d = Math.floor(sec / 86400), h = Math.floor((sec % 86400) / 3600), m = Math.floor((sec % 3600) / 60);
    if (d) return `${d} d ${h} h`;
    if (h) return `${h} h ${m} min`;
    if (m) return `${m} min`;
    return `${sec} s`;
  }

  const fmtNum = (v) => (v == null || !Number.isFinite(Number(v)) ? '—' : Number(v).toLocaleString());
  const fmtClock = (d) => d.toLocaleTimeString([], { hour12: false }) + '.' + String(d.getMilliseconds()).padStart(3, '0');

  function debounce(fn, ms) {
    let t = null;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  }

  function throttle(fn, ms) {
    let t = null;
    return () => { if (!t) t = setTimeout(() => { t = null; fn(); }, ms); };
  }

  // ===================================================================
  // 2. API client + session
  // ===================================================================
  const session = { authed: false, csrf: null, allowReplay: false, showContent: false, agent: '', version: '' };

  class ApiError extends Error {
    constructor(status, message) { super(message); this.status = status; this.handled = false; }
  }

  /**
   * fetch wrapper. 401 -> login screen; 403/409/429 -> toast with the server's
   * message (err.handled = true). opts.raw skips that global handling.
   */
  async function api(method, path, body, opts = {}) {
    const init = { method, credentials: 'same-origin', headers: { Accept: 'application/json' } };
    if (method !== 'GET') {
      init.headers['Content-Type'] = 'application/json';
      if (session.csrf) init.headers['X-CSRF-Token'] = session.csrf;
      init.body = JSON.stringify(body || {});
    }
    let res;
    try {
      res = await fetch(path, init);
    } catch (_) {
      throw new ApiError(0, 'Network error: the dashboard server is unreachable.');
    }
    let data = null;
    if (res.status !== 204) {
      try { data = await res.json(); } catch (_) { data = null; }
    }
    if (res.ok) return data || {};
    const message = data && typeof data.error === 'string' ? data.error : `Request failed (HTTP ${res.status})`;
    const err = new ApiError(res.status, message);
    if (!opts.raw) {
      if (res.status === 401) { err.handled = true; onUnauthorized(); }
      else if ([403, 409, 429].includes(res.status)) { err.handled = true; toast(message, 'error'); }
    }
    throw err;
  }
  const GET = (path) => api('GET', path);
  const POST = (path, body) => api('POST', path, body);

  function reportError(e) {
    if (!e.handled) toast(e.message, 'error');
  }

  function applySession(s) {
    if (s.csrf) session.csrf = s.csrf;
    session.allowReplay = !!s.allow_replay;
    session.showContent = !!s.show_content;
    if (s.agent) session.agent = str(s.agent);
    if (s.version) session.version = str(s.version);
    updateHeader();
  }

  function updateHeader() {
    $('agent-name').textContent = session.agent || 'nexus-a2a';
    $('nexus-version').textContent = session.version ? `nexus ${session.version}` : '';
    $('flag-content').hidden = session.showContent;
    $('flag-replay').hidden = session.allowReplay;
    document.title = session.agent ? `${session.agent} · nexus dashboard` : 'nexus-a2a dashboard';
  }

  // ===================================================================
  // 3. Shared UI
  // ===================================================================
  function toast(message, kind = 'info') {
    const box = $('toasts');
    const t = el('div', { class: `toast toast-${kind}`, role: kind === 'error' ? 'alert' : 'status' }, message);
    box.append(t);
    while (box.childElementCount > 4) box.firstElementChild.remove();
    setTimeout(() => t.remove(), kind === 'error' ? 7000 : 4000);
  }

  /** Modal confirmation via <dialog>.showModal() (native focus trap). Resolves true on confirm. */
  function confirmDialog(title, message, okLabel = 'Confirm', danger = false) {
    const dlg = $('confirm');
    if (dlg.open) return Promise.resolve(false);
    $('confirm-title').textContent = title;
    $('confirm-message').textContent = message;
    const ok = $('confirm-ok');
    ok.textContent = okLabel;
    ok.className = danger ? 'btn btn-danger' : 'btn btn-primary';
    dlg.returnValue = '';
    return new Promise((resolve) => {
      dlg.addEventListener('close', () => resolve(dlg.returnValue === 'ok'), { once: true });
      dlg.showModal();
    });
  }

  function badge(text, kind) {
    return el('span', { class: kind ? `badge badge-${kind}` : 'badge' }, text);
  }
  const stateBadge = (s) => badge(stateLabel(s) || 'unknown', STATES.includes(s) ? s : null);

  const empty = (text) => el('p', { class: 'empty' }, text);
  const loading = () => empty('Loading…');

  function errorState(e, retry) {
    return el('div', { class: 'error-state', role: 'alert' },
      `Could not load data: ${e.message} `,
      retry ? el('button', { type: 'button', class: 'btn btn-sm', onclick: retry }, 'Try again') : null);
  }

  function card(title, ...children) {
    return el('section', { class: 'card' }, el('h2', null, title), children);
  }

  /** headers: string[]; rows: <tr>[]. Wrapped so wide tables scroll horizontally. */
  function table(headers, rows, emptyText) {
    if (!rows.length) return el('div', { class: 'table-wrap' }, empty(emptyText));
    return el('div', { class: 'table-wrap' },
      el('table', null,
        el('thead', null, el('tr', null, headers.map((h) => el('th', { scope: 'col' }, h)))),
        el('tbody', null, rows)));
  }

  const td = (content, cls) => el('td', cls ? { class: cls } : null, content);
  const idLink = (id) => el('a', { href: taskHref(id), title: str(id), class: 'mono' }, shortId(id));
  const errSnippet = (e) => (e ? el('div', { class: 'snippet', title: str(e) }, str(e)) : '');

  /** Per-page lifetime: cancels intervals and lets async callbacks bail out after navigation. */
  function pageCtx() {
    const ctx = {
      alive: true,
      timers: [],
      every(fn, ms) { ctx.timers.push(setInterval(fn, ms)); },
      destroy() { ctx.alive = false; ctx.timers.forEach(clearInterval); },
    };
    return ctx;
  }

  function pageHead(title, ...extra) {
    return el('div', { class: 'page-head' }, el('h1', null, title), el('span', { class: 'spacer' }), extra);
  }

  async function retryTask(id) {
    const ok = await confirmDialog('Retry task?',
      `Start a new task with the first message, skill and context of task ${id}?`, 'Retry');
    if (!ok) return;
    try {
      const r = await POST(`/ui/api/tasks/${encodeURIComponent(id)}/retry`);
      toast('Retry started as a new task.', 'success');
      if (r.task && r.task.id) location.hash = taskHref(r.task.id);
    } catch (e) { reportError(e); }
  }

  // ===================================================================
  // 4. Login / logout
  // ===================================================================
  function showApp(on) {
    $('login').hidden = on;
    $('view').hidden = !on;
    $('nav').hidden = !on;
    $('app-controls').hidden = !on;
  }

  function enterApp() {
    session.authed = true;
    showApp(true);
    stream.start();
    route();
  }

  function showLogin(message) {
    session.authed = false;
    session.csrf = null;
    stream.stop();
    if (current) { current.destroy(); current = null; }
    $('view').replaceChildren();
    showApp(false);
    $('login-error').textContent = message || '';
    $('login-token').value = '';
    $('login-token').focus();
  }

  function onUnauthorized() {
    if (session.authed) showLogin('Your session has ended. Please sign in again.');
  }

  async function onLoginSubmit(ev) {
    ev.preventDefault();
    const input = $('login-token');
    const button = $('login-submit');
    if (!input.value) return;
    button.disabled = true;
    $('login-error').textContent = '';
    try {
      const r = await api('POST', '/ui/api/login', { token: input.value }, { raw: true });
      input.value = '';
      session.csrf = r.csrf || null;
      try { applySession(await api('GET', '/ui/api/session', null, { raw: true })); } catch (_) { /* flags stay as before */ }
      enterApp();
    } catch (e) {
      $('login-error').textContent = e.message;
      input.select();
    } finally {
      button.disabled = false;
    }
  }

  async function onLogout() {
    try {
      await POST('/ui/api/logout');
    } catch (e) {
      if (e.status !== 401) { reportError(e); return; }
    }
    showLogin('Signed out.');
  }

  // ===================================================================
  // 5. Router  (#/overview, #/agents, #/tasks, #/tasks/<id>, #/issues, #/logs)
  // ===================================================================
  let current = null; // { refresh, onEvent?, onVisible?, destroy }

  function route() {
    if (!session.authed) return;
    const parts = location.hash.replace(/^#\/?/, '').split('/');
    const name = parts[0] || 'overview';
    const pages = { overview: OverviewPage, agents: AgentsPage, tasks: TasksPage, issues: IssuesPage, logs: LogsPage };
    let factory = pages[name];
    let arg;
    if (name === 'tasks' && parts[1]) {
      factory = TaskDetailPage;
      try { arg = decodeURIComponent(parts.slice(1).join('/')); } catch (_) { arg = parts.slice(1).join('/'); }
    }
    if (!factory) { location.replace('#/overview'); return; }
    if (current) current.destroy();
    for (const a of $('nav').querySelectorAll('a')) {
      if (a.dataset.route === name) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    }
    const view = $('view');
    view.replaceChildren();
    current = factory(view, arg);
    window.scrollTo(0, 0);
  }

  // ===================================================================
  // 6. Live stream (SSE) with exponential backoff (cap 30 s)
  // ===================================================================
  const stream = (() => {
    let es = null;
    let timer = null;
    let attempt = 0;

    function indicator(state, text) {
      $('live').dataset.state = state;
      $('live-text').textContent = text;
    }

    function dispatch(type, raw) {
      let data;
      try { data = JSON.parse(raw); } catch (_) { return; }
      if (data && typeof data === 'object' && current && current.onEvent) current.onEvent(type, data);
    }

    function open() {
      es = new EventSource('/ui/api/stream');
      indicator('reconnecting', attempt ? 'Reconnecting…' : 'Connecting…');
      es.onopen = () => {
        const wasReconnect = attempt > 0;
        attempt = 0;
        indicator('live', 'Live');
        if (wasReconnect && current) current.refresh();
      };
      es.onerror = () => {
        if (!es) return;
        es.close();
        es = null;
        schedule();
      };
      for (const type of ['task', 'log', 'security', 'webhook']) {
        es.addEventListener(type, (ev) => dispatch(type, ev.data));
      }
    }

    function schedule() {
      const delay = Math.min(30000, 1000 * 2 ** attempt);
      attempt += 1;
      indicator('reconnecting', `Reconnecting in ${Math.round(delay / 1000)} s…`);
      timer = setTimeout(async () => {
        timer = null;
        // An EventSource cannot see HTTP status codes, so ask the session endpoint
        // whether we were logged out before opening a new stream.
        try {
          const s = await api('GET', '/ui/api/session', null, { raw: true });
          if (!session.authed) return;
          if (!s.authenticated) { onUnauthorized(); return; }
        } catch (_) {
          if (session.authed) schedule();
          return;
        }
        open();
      }, delay);
    }

    function stop() {
      if (es) { es.close(); es = null; }
      clearTimeout(timer);
      timer = null;
      attempt = 0;
      indicator('offline', 'Offline');
    }

    return { start() { stop(); open(); }, stop };
  })();

  // ===================================================================
  // 7. Pages
  // ===================================================================

  // ---------- Overview ----------
  function OverviewPage(view) {
    const ctx = pageCtx();
    const stamp = el('span', { class: 'muted' });
    const body = el('div', null, loading());
    view.append(pageHead('Overview', stamp), body);
    let busy = false;

    async function refresh() {
      if (busy || !ctx.alive) return;
      busy = true;
      try {
        const d = await GET('/ui/api/overview');
        if (ctx.alive) render(d);
      } catch (e) {
        if (ctx.alive && e.status !== 401) body.replaceChildren(errorState(e, refresh));
      } finally {
        busy = false;
      }
    }

    function render(d) {
      const agent = d.agent || {};
      if (agent.name) session.agent = str(agent.name);
      if (d.nexus_version) session.version = str(d.nexus_version);
      updateHeader();
      stamp.textContent = `Updated ${new Date().toLocaleTimeString()}`;

      const tasks = d.tasks || {};
      const tp = d.throughput || {};
      const lat = d.latency_ms || {};
      const dlq = d.dlq || {};
      const wh = d.webhooks || {};
      const health = d.health || {};
      const errRate = d.error_rate == null ? '—' : `${(num(d.error_rate) * 100).toFixed(1)}%`;

      const kpi = (label, value, sub, alert) => el('div', { class: alert ? 'kpi kpi-alert' : 'kpi' },
        el('div', { class: 'kpi-label' }, label), el('div', { class: 'kpi-value' }, value),
        sub ? el('div', { class: 'kpi-sub' }, sub) : null);
      const healthBadge = (v) => badge(dash(v), ['ok', 'error', 'degraded', 'no_agents'].includes(v) ? v : null);

      body.replaceChildren(
        el('div', { class: 'health' },
          el('span', null, 'Store ', healthBadge(health.store)),
          el('span', null, 'Registry ', healthBadge(health.registry)),
          el('span', { class: 'muted' }, `Agent ${dash(agent.name)} v${dash(agent.version)} at `, el('code', null, dash(agent.url)))),
        el('div', { class: 'kpis' },
          kpi('In flight', fmtNum(tasks.in_flight), `${fmtNum(tasks.total_seen)} seen in total`),
          kpi('Completed (5 min)', fmtNum(tp.completed_last_5m)),
          kpi('Failed (5 min)', fmtNum(tp.failed_last_5m), null, num(tp.failed_last_5m) > 0),
          kpi('Error rate (5 min)', errRate, null, num(d.error_rate) >= 0.1),
          kpi('Latency p50', fmtMs(lat.p50), `${fmtNum(lat.samples)} samples`),
          kpi('Latency p95', fmtMs(lat.p95)),
          kpi('DLQ pending', fmtNum(dlq.pending), `${fmtNum(dlq.total)} total`, num(dlq.pending) > 0),
          kpi('Security events (1 h)', fmtNum(d.security_events_last_hour), null, num(d.security_events_last_hour) > 0),
          kpi('Webhooks', `${fmtNum(wh.delivered)} / ${fmtNum(wh.failed)}`, 'delivered / failed', num(wh.failed) > 0),
          kpi('Uptime', fmtDuration(d.uptime_seconds))),
        el('div', { class: 'grid-2' },
          card('Throughput — last 30 minutes', throughputChart(tp.per_minute)),
          card('Tasks by state', stateBreakdown(tasks.by_state || {}, tasks.total_seen))));
    }

    ctx.every(() => { if (!document.hidden) refresh(); }, 5000);
    const soon = throttle(refresh, 2000);
    refresh();
    return { refresh, onVisible: refresh, onEvent: (type) => { if (type !== 'log' && !document.hidden) soon(); }, destroy: ctx.destroy };
  }

  function throughputChart(perMinute) {
    const data = Array.isArray(perMinute) ? perMinute : [];
    if (!data.length) return empty('No throughput data yet.');
    const W = 300, H = 100;
    const n = data.length;
    const slot = W / n, bw = slot * 0.72;
    const max = Math.max(1, ...data.map((p) => num(p && p.completed) + num(p && p.failed)));
    let sumC = 0, sumF = 0;
    const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none', role: 'img' });
    for (const f of [0.25, 0.5, 0.75]) {
      root.append(svg('line', { class: 'gridline', x1: 0, x2: W, y1: H * f, y2: H * f, 'vector-effect': 'non-scaling-stroke' }));
    }
    data.forEach((p, i) => {
      const c = num(p && p.completed), f = num(p && p.failed);
      sumC += c; sumF += f;
      const x = i * slot + (slot - bw) / 2;
      const hc = (c / max) * H, hf = (f / max) * H;
      const g = svg('g', null, svg('title', null, `${n - i} min ago: ${c} completed, ${f} failed`));
      // Transparent full-height hit area so the tooltip works on empty minutes too.
      g.append(svg('rect', { x: i * slot, y: 0, width: slot, height: H, fill: 'transparent' }));
      if (c) g.append(svg('rect', { class: 'bar-ok', x, y: H - hc, width: bw, height: hc }));
      if (f) g.append(svg('rect', { class: 'bar-err', x, y: H - hc - hf, width: bw, height: hf }));
      root.append(g);
    });
    root.setAttribute('aria-label', `Last ${n} minutes: ${sumC} completed, ${sumF} failed; busiest minute ${max} tasks`);
    return el('div', { class: 'chart' },
      el('div', { class: 'toolbar' },
        el('div', { class: 'legend' },
          el('span', null, el('span', { class: 'swatch swatch-ok' }), `Completed (${sumC})`),
          el('span', null, el('span', { class: 'swatch swatch-err' }), `Failed (${sumF})`)),
        el('span', { class: 'spacer' }),
        el('span', { class: 'chart-ymax' }, `peak ${max}/min`)),
      root,
      el('div', { class: 'chart-axis', 'aria-hidden': 'true' },
        el('span', null, `${n} min ago`), el('span', null, `${Math.round(n / 2)} min ago`), el('span', null, 'now')));
  }

  function stateBreakdown(byState, totalSeen) {
    const total = STATES.reduce((a, s) => a + num(byState[s]), 0);
    if (!total) return empty('No tasks seen yet.');
    return el('div', null,
      el('ul', { class: 'breakdown' }, STATES.map((s) => {
        const fill = el('div', { class: `meter-fill s-${s}` });
        fill.style.width = `${(num(byState[s]) / total) * 100}%`;
        return el('li', null, el('span', null, stateBadge(s)), el('div', { class: 'meter' }, fill),
          el('span', { class: 'count' }, fmtNum(byState[s])));
      })),
      el('p', { class: 'table-meta' }, `${fmtNum(totalSeen)} tasks seen since start.`));
  }

  // ---------- Agents ----------
  function AgentsPage(view) {
    const ctx = pageCtx();
    const body = el('div', null, loading());
    view.append(pageHead('Agents'), body);

    async function refresh() {
      try {
        const d = await GET('/ui/api/agents');
        if (ctx.alive) render(d);
      } catch (e) {
        if (ctx.alive && e.status !== 401) body.replaceChildren(errorState(e, refresh));
      }
    }

    async function checkHealth(ev) {
      const btn = ev.currentTarget;
      btn.disabled = true;
      try {
        const r = await POST('/ui/api/agents/check');
        const results = Object.values((r && r.results) || {});
        toast(`Checked ${results.length} peer(s): ${results.filter(Boolean).length} healthy.`, 'success');
        await refresh();
      } catch (e) {
        reportError(e);
      } finally {
        btn.disabled = false;
      }
    }

    function selfCard(a) {
      if (!a) return card('This agent', empty('No agent card is registered with this server.'));
      const caps = Object.entries(a.capabilities || {});
      const skills = Array.isArray(a.skills) ? a.skills : [];
      return el('section', { class: 'card' },
        el('h2', null, dash(a.name), badge(`v${dash(a.version)}`)),
        el('dl', { class: 'kv' },
          el('dt', null, 'URL'), el('dd', null, el('code', null, dash(a.url))),
          el('dt', null, 'Description'), el('dd', null, dash(a.description)),
          el('dt', null, 'Authentication'), el('dd', null, a.authentication ? dash(a.authentication.scheme) : 'none'),
          el('dt', null, 'Capabilities'),
          el('dd', null, caps.length ? caps.map(([k, v]) => [badge(`${k.replace(/_/g, ' ')}: ${v ? 'yes' : 'no'}`, v ? 'yes' : 'no'), ' ']) : '—')),
        el('h3', { class: 'section-title' }, 'Skills'),
        table(['ID', 'Name', 'Description'],
          skills.map((s) => el('tr', null, td(el('code', null, dash(s.id))), td(dash(s.name)), td(dash(s.description), 'cell-wrap'))),
          'This agent advertises no skills.'));
    }

    function peersCard(peers) {
      const rows = peers.map((p) => {
        const seen = p.last_seen_seconds_ago == null ? null : new Date(Date.now() - num(p.last_seen_seconds_ago) * 1000);
        const health = p.healthy === true ? badge('healthy', 'ok') : p.healthy === false ? badge('unhealthy', 'bad') : badge('unknown');
        return el('tr', null,
          td(dash(p.name)), td(el('code', null, dash(p.url))), td(health),
          td(seen ? relTime(seen) : el('span', { class: 'muted' }, 'never'), 'nowrap'),
          td(Array.isArray(p.skills) && p.skills.length ? p.skills.map(str).join(', ') : '—', 'cell-wrap'));
      });
      return el('section', { class: 'card' },
        el('div', { class: 'page-head' },
          el('h2', null, 'Peers'), el('span', { class: 'spacer' }),
          el('button', { type: 'button', class: 'btn', onclick: checkHealth, disabled: !peers.length }, 'Check health')),
        table(['Name', 'URL', 'Health', 'Last seen', 'Skills'], rows, 'No peer agents are registered.'));
    }

    function render(d) {
      body.replaceChildren(selfCard(d.self), peersCard(Array.isArray(d.peers) ? d.peers : []));
    }

    refresh();
    return { refresh, destroy: ctx.destroy };
  }

  // ---------- Tasks ----------
  const taskFilters = { state: '', skill: '', q: '' }; // survives navigation to a detail page and back
  const TASK_LIMIT = 100;

  function TasksPage(view) {
    const ctx = pageCtx();
    const rows = new Map(); // task id -> <tr>
    let total = 0;
    let seq = 0;

    const stateSel = el('select', { id: 'f-state' }, el('option', { value: '' }, 'All states'),
      STATES.map((s) => el('option', { value: s }, stateLabel(s))));
    const skillIn = el('input', { id: 'f-skill', type: 'text', placeholder: 'exact skill id', autocomplete: 'off' });
    const qIn = el('input', { id: 'f-q', type: 'search', placeholder: 'id, owner, skill or error', autocomplete: 'off' });
    stateSel.value = taskFilters.state;
    skillIn.value = taskFilters.skill;
    qIn.value = taskFilters.q;

    const tbody = el('tbody');
    const tableWrap = el('div', { class: 'table-wrap' }, el('table', null,
      el('thead', null, el('tr', null, ['ID', 'State', 'Skill', 'Owner', 'Updated', 'Duration', 'Error'].map((h) => el('th', { scope: 'col' }, h)))),
      tbody));
    const emptyBox = el('div', { class: 'table-wrap' }, empty('No tasks match these filters.'));
    const meta = el('p', { class: 'table-meta', 'aria-live': 'polite' });
    const errBox = el('div');

    const onFilter = () => {
      taskFilters.state = stateSel.value;
      taskFilters.skill = skillIn.value.trim();
      taskFilters.q = qIn.value.trim();
      refresh();
    };
    const onType = debounce(onFilter, 300);
    stateSel.addEventListener('change', onFilter);
    skillIn.addEventListener('input', onType);
    qIn.addEventListener('input', onType);

    view.append(pageHead('Tasks'),
      el('div', { class: 'filters' },
        el('div', { class: 'field' }, el('label', { for: 'f-state' }, 'State'), stateSel),
        el('div', { class: 'field' }, el('label', { for: 'f-skill' }, 'Skill'), skillIn),
        el('div', { class: 'field field-grow' }, el('label', { for: 'f-q' }, 'Search'), qIn),
        el('button', { type: 'button', class: 'btn', onclick: () => refresh() }, 'Refresh')),
      errBox, tableWrap, emptyBox, meta);
    tableWrap.hidden = true;
    emptyBox.replaceChildren(loading());

    function row(t) {
      const tr = el('tr', { class: 'clickable', 'data-id': str(t.id) },
        td(idLink(t.id)), td(stateBadge(t.state)), td(dash(t.skill_id)), td(dash(t.owner), 'cell-wrap'),
        td(relTime(t.updated_at), 'nowrap'), td(fmtMs(t.duration_ms), 'num'), td(errSnippet(t.error)));
      tr.addEventListener('click', (ev) => {
        if (!ev.target.closest('a, button') && !window.getSelection().toString()) location.hash = taskHref(t.id);
      });
      return tr;
    }

    function sync() {
      tableWrap.hidden = rows.size === 0;
      emptyBox.hidden = rows.size > 0;
      emptyBox.replaceChildren(empty('No tasks match these filters.'));
      meta.textContent = `Showing ${rows.size} of ${fmtNum(total)} matching task(s), newest first. Live updates are applied as they arrive.`;
    }

    function matches(t) {
      const f = taskFilters;
      if (f.state && t.state !== f.state) return false;
      if (f.skill && t.skill_id !== f.skill) return false;
      if (!f.q) return true;
      const q = f.q.toLowerCase();
      return [t.id, t.owner, t.skill_id, t.error].some((v) => typeof v === 'string' && v.toLowerCase().includes(q));
    }

    async function refresh() {
      const mine = ++seq;
      const qs = new URLSearchParams({ limit: String(TASK_LIMIT) });
      for (const k of ['state', 'skill', 'q']) if (taskFilters[k]) qs.set(k, taskFilters[k]);
      try {
        const d = await GET(`/ui/api/tasks?${qs}`);
        if (!ctx.alive || mine !== seq) return;
        rows.clear();
        tbody.replaceChildren();
        for (const t of Array.isArray(d.tasks) ? d.tasks : []) {
          if (!t || rows.has(t.id)) continue;
          const tr = row(t);
          rows.set(t.id, tr);
          tbody.append(tr);
        }
        total = num(d.total);
        errBox.replaceChildren();
        sync();
      } catch (e) {
        if (ctx.alive && mine === seq && e.status !== 401) errBox.replaceChildren(errorState(e, refresh));
      }
    }

    function onEvent(type, data) {
      const t = data.task;
      if (type !== 'task' || !t || typeof t.id !== 'string') return;
      const existing = rows.get(t.id);
      if (existing) { existing.remove(); rows.delete(t.id); }
      if (matches(t)) {
        const tr = row(t);
        tr.classList.add('flash');
        tbody.prepend(tr);
        rows.set(t.id, tr);
        if (!existing) total += 1;
        while (rows.size > TASK_LIMIT) {
          const last = tbody.lastElementChild;
          rows.delete(last.dataset.id);
          last.remove();
        }
      } else if (existing) {
        total = Math.max(0, total - 1);
      }
      sync();
    }

    refresh();
    return { refresh, onEvent, destroy: ctx.destroy };
  }

  // ---------- Task detail ----------
  function TaskDetailPage(view, id) {
    const ctx = pageCtx();
    const actions = el('div', { class: 'toolbar' });
    const body = el('div', null, loading());
    view.append(
      el('p', null, el('a', { href: '#/tasks' }, '← All tasks')),
      pageHead(['Task ', el('code', { class: 'muted title-id' }, id)], actions),
      body);

    async function refresh() {
      try {
        const d = await GET(`/ui/api/tasks/${encodeURIComponent(id)}`);
        if (ctx.alive) render(d);
      } catch (e) {
        if (!ctx.alive || e.status === 401) return;
        actions.replaceChildren();
        body.replaceChildren(e.status === 404
          ? el('div', { class: 'card' }, empty('Task not found. It may never have existed or has expired from the store.'))
          : errorState(e, refresh));
      }
    }

    async function cancel(ev) {
      const ok = await confirmDialog('Cancel task?', `Cancel task ${id}? The agent will stop working on it.`, 'Cancel task', true);
      if (!ok) return;
      const btn = ev.target;
      btn.disabled = true;
      try {
        await POST(`/ui/api/tasks/${encodeURIComponent(id)}/cancel`);
        toast('Task cancelled.', 'success');
        await refresh();
      } catch (e) {
        reportError(e);
        btn.disabled = false;
      }
    }

    function render(d) {
      const t = d.task || {};
      const acts = d.actions || {};
      actions.replaceChildren();
      appendChildren(actions, [
        acts.can_cancel ? el('button', { type: 'button', class: 'btn btn-danger', onclick: cancel }, 'Cancel task') : null,
        acts.can_retry && session.allowReplay ? el('button', { type: 'button', class: 'btn btn-primary', onclick: () => retryTask(id) }, 'Retry') : null]);

      const history = Array.isArray(d.history) ? d.history : [];
      const artifacts = Array.isArray(d.artifacts) ? d.artifacts : [];
      const anyRedacted = [...history, ...artifacts].some((m) => Array.isArray(m.parts) && m.parts.some((p) => p && p.redacted));
      const hiddenNote = anyRedacted
        ? el('p', { class: 'note' }, 'Message content is hidden because show_content is off in the dashboard configuration. Only part types and sizes are shown.')
        : null;

      body.replaceChildren(el('div', { class: 'grid-2' },
        el('div', { class: 'stack' },
          card('Summary', summary(t)),
          card('History', hiddenNote, history.length
            ? history.map((m) => message(badge(dash(m.role), m.role === 'agent' ? 'working' : null), m.parts))
            : empty('No messages recorded.')),
          card('Artifacts', artifacts.length
            ? artifacts.map((a) => message(el('strong', null, a.name == null ? 'unnamed artifact' : str(a.name)), a.parts))
            : empty('No artifacts produced.'))),
        el('div', { class: 'stack' },
          card('Timeline', timeline(d.timeline)),
          card('Trace', trace(d.trace)))));
    }

    function summary(t) {
      const created = parseTime(t.created_at), updated = parseTime(t.updated_at);
      const when = (d) => (d ? [d.toLocaleString(), ' (', relTime(d), ')'] : '—');
      return el('dl', { class: 'kv' },
        el('dt', null, 'ID'), el('dd', null, el('code', null, dash(t.id))),
        el('dt', null, 'State'), el('dd', null, stateBadge(t.state)),
        el('dt', null, 'Skill'), el('dd', null, dash(t.skill_id)),
        el('dt', null, 'Context'), el('dd', null, el('code', null, dash(t.context_id))),
        el('dt', null, 'Owner'), el('dd', null, dash(t.owner)),
        el('dt', null, 'Created'), el('dd', null, when(created)),
        el('dt', null, 'Updated'), el('dd', null, when(updated)),
        el('dt', null, 'Duration'), el('dd', null, fmtMs(t.duration_ms)),
        t.error ? [el('dt', null, 'Error'), el('dd', null, el('pre', { class: 'hop-error' }, str(t.error)))] : null);
    }

    function timeline(items) {
      if (!Array.isArray(items) || !items.length) return empty('No timeline events.');
      const start = parseTime(items[0].at);
      return el('ol', { class: 'timeline' }, items.map((ev) => {
        const at = parseTime(ev.at);
        const offset = at && start ? ` (+${fmtMs(at - start)})` : '';
        return el('li', null, el('div', { class: 'ev' }, stateLabel(ev.event)),
          el('div', { class: 'when' }, at ? at.toLocaleString() : '—', offset));
      }));
    }

    function trace(tr) {
      const hops = tr && Array.isArray(tr.hops) ? tr.hops : [];
      if (!hops.length) return empty('No trace recorded for this task.');
      const max = Math.max(1, ...hops.map((h) => num(h.duration_ms)));
      return el('div', null,
        el('p', { class: 'muted' }, 'Trace ', el('code', null, dash(tr.trace_id))),
        el('ul', { class: 'hops' }, hops.map((h) => {
          const bar = el('div', { class: h.status === 'error' ? 'hop-bar err' : 'hop-bar' });
          bar.style.width = `${(num(h.duration_ms) / max) * 100}%`;
          return el('li', { class: 'hop' },
            el('div', { class: 'hop-head' },
              el('code', { class: 'hop-url' }, dash(h.url)),
              el('span', { class: 'nowrap' }, badge(dash(h.status), h.status === 'error' ? 'bad' : 'ok'), ' ', fmtMs(h.duration_ms))),
            bar,
            h.error ? el('div', { class: 'hop-error' }, str(h.error)) : null);
        })));
    }

    function message(head, parts) {
      const list = Array.isArray(parts) ? parts : [];
      return el('div', { class: 'msg' }, el('div', { class: 'msg-head' }, head),
        list.length ? list.map(part) : el('p', { class: 'muted' }, 'No parts.'));
    }

    function part(p) {
      p = p || {};
      let content;
      if (p.redacted) content = el('div', { class: 'part-hidden' }, `content hidden (${fmtNum(p.length)} chars)`);
      else if (p.content == null) content = el('div', { class: 'part-hidden' }, 'empty');
      else content = el('pre', null, typeof p.content === 'string' ? p.content : JSON.stringify(p.content, null, 2));
      return el('div', { class: 'part' }, el('div', { class: 'part-type' }, dash(p.type)), content);
    }

    const soon = throttle(refresh, 1000);
    refresh();
    return {
      refresh,
      onEvent: (type, data) => { if (type === 'task' && data.task && data.task.id === id) soon(); },
      destroy: ctx.destroy,
    };
  }

  // ---------- Issues ----------
  const ISSUE_TABS = [['failed', 'Failed tasks'], ['dlq', 'Dead-letter queue'], ['security', 'Security events'], ['webhooks', 'Webhook failures']];
  let issuesTab = 'failed';

  function IssuesPage(view) {
    const ctx = pageCtx();
    const selected = new Set();
    let data = null;
    let replayResults = null;

    const tabBtns = ISSUE_TABS.map(([key, label]) => {
      const count = badge('…');
      const b = el('button', { type: 'button', class: 'tab', role: 'tab', id: `tab-${key}`, 'aria-controls': 'issues-panel', onclick: () => select(key) }, label, count);
      b.count = count;
      b.key = key;
      return b;
    });
    const tablist = el('div', { class: 'tabs', role: 'tablist', 'aria-label': 'Issue categories' }, tabBtns);
    tablist.addEventListener('keydown', (ev) => {
      if (ev.key !== 'ArrowRight' && ev.key !== 'ArrowLeft') return;
      const i = ISSUE_TABS.findIndex(([k]) => k === issuesTab);
      const n = (i + (ev.key === 'ArrowRight' ? 1 : -1) + ISSUE_TABS.length) % ISSUE_TABS.length;
      select(ISSUE_TABS[n][0]);
      tabBtns[n].focus();
    });
    const panel = el('div', { id: 'issues-panel', role: 'tabpanel', tabindex: '0' }, loading());
    view.append(pageHead('Issues', el('button', { type: 'button', class: 'btn', onclick: () => refresh() }, 'Refresh')), tablist, panel);

    function select(key) {
      issuesTab = key;
      renderTabs();
      renderPanel();
    }

    function renderTabs() {
      for (const b of tabBtns) {
        const on = b.key === issuesTab;
        b.setAttribute('aria-selected', on ? 'true' : 'false');
        b.tabIndex = on ? 0 : -1;
        if (data) {
          const n = b.key === 'failed' ? data.failed_tasks.length
            : b.key === 'dlq' ? data.dlq.filter((x) => !x.replayed).length
              : b.key === 'security' ? data.security_events.length : data.webhook_failures.length;
          b.count.textContent = String(n);
          b.count.className = n ? 'badge badge-bad' : 'badge';
        }
      }
      panel.setAttribute('aria-labelledby', `tab-${issuesTab}`);
    }

    async function refresh() {
      try {
        const d = await GET('/ui/api/issues');
        if (!ctx.alive) return;
        const arr = (v) => (Array.isArray(v) ? v.filter((x) => x && typeof x === 'object') : []);
        data = { failed_tasks: arr(d.failed_tasks), dlq: arr(d.dlq), security_events: arr(d.security_events), webhook_failures: arr(d.webhook_failures) };
        for (const id of [...selected]) if (!data.dlq.some((x) => x.task_id === id && !x.replayed)) selected.delete(id);
        renderTabs();
        renderPanel();
      } catch (e) {
        if (ctx.alive && e.status !== 401) panel.replaceChildren(errorState(e, refresh));
      }
    }

    function renderPanel() {
      if (!data) return;
      const render = { failed: failedPanel, dlq: dlqPanel, security: securityPanel, webhooks: webhookPanel }[issuesTab];
      panel.replaceChildren(render());
    }

    function failedPanel() {
      const rows = data.failed_tasks.map((t) => el('tr', null,
        td(idLink(t.id)), td(dash(t.skill_id)), td(dash(t.owner), 'cell-wrap'), td(relTime(t.updated_at), 'nowrap'),
        td(fmtMs(t.duration_ms), 'num'), td(errSnippet(t.error)),
        session.allowReplay ? td(el('button', { type: 'button', class: 'btn btn-sm', onclick: () => retryTask(t.id) }, 'Retry')) : null));
      const headers = ['ID', 'Skill', 'Owner', 'Updated', 'Duration', 'Error'];
      if (session.allowReplay) headers.push('Actions');
      return table(headers, rows, 'No failed tasks.');
    }

    function dlqPanel() {
      const pending = data.dlq.filter((x) => !x.replayed);
      const replay = session.allowReplay;
      const replaySel = el('button', { type: 'button', class: 'btn btn-primary', disabled: !selected.size, onclick: () => doReplay({ task_ids: [...selected] }, selected.size) }, `Replay selected (${selected.size})`);
      const updateSel = () => {
        replaySel.textContent = `Replay selected (${selected.size})`;
        replaySel.disabled = !selected.size;
      };
      const boxes = [];
      const rows = data.dlq.map((x) => {
        const box = el('input', { type: 'checkbox', 'aria-label': `Select ${str(x.task_id)}`, disabled: !!x.replayed });
        box.checked = selected.has(x.task_id);
        box.addEventListener('change', () => {
          if (box.checked) selected.add(x.task_id); else selected.delete(x.task_id);
          updateSel();
        });
        if (!x.replayed) boxes.push([box, x.task_id]);
        return el('tr', null,
          replay ? td(box) : null,
          td(idLink(x.task_id)), td(dash(x.error), 'cell-wrap'), td(relTime(parseTime(x.failed_at)), 'nowrap'),
          td(el('code', null, dash(x.agent_url))), td(dash(x.skill_id)), td(fmtNum(x.retry_count), 'num'),
          td(relTime(parseTime(x.last_retry_at)), 'nowrap'),
          td(x.replayed ? badge('replayed', 'ok') : badge('pending', 'warn')));
      });
      const all = el('input', { type: 'checkbox', 'aria-label': 'Select all pending entries', disabled: !boxes.length });
      all.checked = boxes.length > 0 && boxes.every(([, id]) => selected.has(id));
      all.addEventListener('change', () => {
        for (const [box, id] of boxes) { box.checked = all.checked; if (all.checked) selected.add(id); else selected.delete(id); }
        updateSel();
      });
      const headers = ['Task', 'Error', 'Failed', 'Agent', 'Skill', 'Retries', 'Last retry', 'Status'];
      if (replay) headers.unshift(all);

      return el('div', null,
        replay
          ? el('div', { class: 'toolbar' }, replaySel,
            el('button', { type: 'button', class: 'btn', disabled: !pending.length, onclick: () => doReplay({ all: true }, pending.length) }, 'Replay all'))
          : el('p', { class: 'note' }, 'Replay is disabled on this server (allow_replay = false).'),
        table(headers, rows, 'The dead-letter queue is empty.'),
        replayResults ? resultsList(replayResults) : null);
    }

    function resultsList(results) {
      return el('section', { class: 'card', 'aria-live': 'polite' },
        el('h3', null, 'Replay results'),
        results.length ? el('ul', { class: 'result-list' }, results.map((r) => el('li', null,
          el('code', { title: str(r.task_id) }, shortId(r.task_id)),
          r.success ? badge('replayed', 'ok') : badge('failed', 'bad'),
          r.error ? el('span', { class: 'hop-error' }, str(r.error)) : null,
          r.new_task_id ? ['new task ', idLink(r.new_task_id)] : null))) : empty('Nothing was replayed.'));
    }

    async function doReplay(body, count) {
      const msg = body.all ? `Replay all ${count} pending dead-letter entries?` : `Replay ${count} selected dead-letter ${count === 1 ? 'entry' : 'entries'}?`;
      if (!(await confirmDialog('Replay from DLQ?', msg, 'Replay'))) return;
      try {
        const r = await POST('/ui/api/dlq/replay', body);
        replayResults = Array.isArray(r.results) ? r.results.filter((x) => x && typeof x === 'object') : [];
        selected.clear();
        const ok = replayResults.filter((x) => x.success).length;
        toast(`Replayed ${ok} of ${replayResults.length} entr${replayResults.length === 1 ? 'y' : 'ies'}.`, ok === replayResults.length ? 'success' : 'error');
        await refresh();
      } catch (e) { reportError(e); }
    }

    function securityPanel() {
      const kindClass = (k) => (['rate_limited', 'payload_too_large', 'bad_request'].includes(k) ? 'warn' : 'bad');
      const rows = data.security_events.map((e) => el('tr', null,
        td(relTime(e.at), 'nowrap'), td(badge(dash(e.kind), kindClass(e.kind))), td(dash(e.status), 'num'),
        td(dash(e.caller), 'cell-wrap'), td(dash(e.client)), td(dash(e.detail), 'cell-wrap')));
      return table(['Time', 'Kind', 'Status', 'Caller', 'Client', 'Detail'], rows, 'No security events recorded.');
    }

    function webhookPanel() {
      const rows = data.webhook_failures.map((w) => el('tr', null,
        td(relTime(w.at), 'nowrap'), td(idLink(w.task_id)), td(dash(w.event)), td(el('code', null, dash(w.url)), 'cell-wrap'),
        td(dash(w.status), 'num'), td(fmtNum(w.attempts), 'num'), td(dash(w.error), 'cell-wrap')));
      return table(['Time', 'Task', 'Event', 'URL', 'Status', 'Attempts', 'Error'], rows, 'No webhook delivery failures.');
    }

    function onEvent(type, d) {
      if (!data) return;
      let tab;
      if (type === 'security') {
        data.security_events.unshift(d);
        data.security_events.length = Math.min(data.security_events.length, 200);
        tab = 'security';
      } else if (type === 'webhook' && d.succeeded === false) {
        data.webhook_failures.unshift(d);
        data.webhook_failures.length = Math.min(data.webhook_failures.length, 100);
        tab = 'webhooks';
      } else if (type === 'task' && d.task && typeof d.task.id === 'string') {
        const i = data.failed_tasks.findIndex((x) => x.id === d.task.id);
        if (i < 0 && d.task.state !== 'failed') return;
        if (i >= 0) data.failed_tasks.splice(i, 1);
        if (d.task.state === 'failed') data.failed_tasks.unshift(d.task);
        tab = 'failed';
      } else {
        return;
      }
      renderTabs();
      // The DLQ tab is only re-rendered on explicit refresh so selections are never disturbed.
      if (tab === issuesTab) renderPanel();
    }

    renderTabs();
    refresh();
    return { refresh, onEvent, destroy: ctx.destroy };
  }

  // ---------- Logs ----------
  const logFilters = { level: 'INFO', q: '' };
  const MAX_LOG_ROWS = 2000;

  function LogsPage(view) {
    const ctx = pageCtx();
    let paused = false;
    let seq = 0;
    const buffer = [];

    const levelSel = el('select', { id: 'l-level' }, LEVELS.map((l) => el('option', { value: l }, l)));
    const qIn = el('input', { id: 'l-q', type: 'search', placeholder: 'message or logger (paste a task id)', autocomplete: 'off' });
    levelSel.value = logFilters.level;
    qIn.value = logFilters.q;
    const pauseBtn = el('button', { type: 'button', class: 'btn', 'aria-pressed': 'false' }, 'Pause');
    const status = el('span', { class: 'muted', 'aria-live': 'polite' });
    const box = el('div', { class: 'logbox', role: 'log', 'aria-label': 'Log lines', tabindex: '0' }, loading());

    const onFilter = () => {
      logFilters.level = levelSel.value;
      logFilters.q = qIn.value.trim();
      refresh();
    };
    levelSel.addEventListener('change', onFilter);
    qIn.addEventListener('input', debounce(onFilter, 300));
    pauseBtn.addEventListener('click', () => {
      paused = !paused;
      pauseBtn.textContent = paused ? 'Resume' : 'Pause';
      pauseBtn.setAttribute('aria-pressed', String(paused));
      if (!paused && buffer.length) append(buffer.splice(0), true);
      updateStatus();
    });

    view.append(pageHead('Logs', status),
      el('div', { class: 'filters' },
        el('div', { class: 'field' }, el('label', { for: 'l-level' }, 'Minimum level'), levelSel),
        el('div', { class: 'field field-grow' }, el('label', { for: 'l-q' }, 'Search'), qIn),
        pauseBtn),
      box);

    const rank = (lv) => Math.max(0, LEVELS.indexOf(lv));
    function matches(l) {
      if (rank(l.level) < rank(logFilters.level)) return false;
      const q = logFilters.q.toLowerCase();
      return !q || str(l.message).toLowerCase().includes(q) || str(l.logger).toLowerCase().includes(q);
    }

    function row(l) {
      const d = parseTime(l.at);
      const lv = LEVELS.includes(l.level) ? l.level : 'INFO';
      return el('div', { class: `logrow r-${lv}` },
        el('span', { class: 't', title: d ? d.toLocaleString() : '' }, d ? fmtClock(d) : '—'),
        el('span', { class: `lv lv-${lv}` }, dash(l.level)),
        el('span', { class: 'lg', title: str(l.logger) }, dash(l.logger)),
        el('span', { class: 'm' }, str(l.message)));
    }

    function rowCount() { return box.querySelectorAll('.logrow').length; }

    function updateStatus() {
      const n = rowCount();
      status.textContent = `${fmtNum(n)} line(s)` + (paused ? ` · paused, ${fmtNum(buffer.length)} buffered` : ' · live');
    }

    function append(lines, forceScroll) {
      const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 30;
      if (!rowCount()) box.replaceChildren();
      const frag = document.createDocumentFragment();
      for (const l of lines) frag.append(row(l));
      box.append(frag);
      let excess = rowCount() - MAX_LOG_ROWS;
      while (excess-- > 0) box.firstElementChild.remove();
      if (!rowCount()) box.replaceChildren(empty('No log lines match these filters.'));
      if (atBottom || forceScroll) box.scrollTop = box.scrollHeight;
      updateStatus();
    }

    async function refresh() {
      const mine = ++seq;
      const qs = new URLSearchParams({ level: logFilters.level, limit: '500' });
      if (logFilters.q) qs.set('q', logFilters.q);
      try {
        const d = await GET(`/ui/api/logs?${qs}`);
        if (!ctx.alive || mine !== seq) return;
        buffer.length = 0;
        box.replaceChildren();
        append((Array.isArray(d.logs) ? d.logs : []).filter((l) => l && typeof l === 'object'), true);
      } catch (e) {
        if (ctx.alive && mine === seq && e.status !== 401) box.replaceChildren(errorState(e, refresh));
      }
    }

    function onEvent(type, l) {
      if (type !== 'log' || !matches(l)) return;
      if (paused) {
        buffer.push(l);
        if (buffer.length > MAX_LOG_ROWS) buffer.shift();
        updateStatus();
      } else {
        append([l], false);
      }
    }

    refresh();
    return { refresh, onEvent, destroy: ctx.destroy };
  }

  // ===================================================================
  // 8. Boot
  // ===================================================================
  async function boot() {
    $('login-form').addEventListener('submit', onLoginSubmit);
    $('logout').addEventListener('click', onLogout);
    window.addEventListener('hashchange', () => {
      route();
      if (session.authed) $('view').focus({ preventScroll: true });
    });
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden && current && current.onVisible) current.onVisible();
    });
    setInterval(updateRelTimes, 30000);

    try {
      const s = await api('GET', '/ui/api/session', null, { raw: true });
      applySession(s);
      if (s.authenticated) enterApp(); else showLogin();
    } catch (e) {
      updateHeader();
      showLogin(e.message);
    }
  }

  document.addEventListener('DOMContentLoaded', boot);
})();

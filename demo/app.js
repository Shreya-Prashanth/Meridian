const $ = (id) => document.getElementById(id);
const conversation = $("conversation");
const activity = $("activity");
const tools = $("tools");
const status = $("status");
const activeTools = new Map();
let lastState = { version: 0, epoch: 0, intent: null, slots: {}, pending_calls: [], completed_calls: [] };

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}

function addBubble(kind, label, text) {
  const div = document.createElement("div");
  div.className = `bubble ${kind}`;
  div.innerHTML = `<span class="label">${escapeHtml(label)}</span>${escapeHtml(text)}`;
  conversation.appendChild(div);
  conversation.scrollTop = conversation.scrollHeight;
}

function setStatus(kind, text) {
  status.className = `status ${kind}`;
  status.innerHTML = `<span></span> ${escapeHtml(text)}`;
}

function activityRow(kind, detail, timestamp = null) {
  const row = document.createElement("div");
  row.className = "activity-row";
  const time = timestamp == null ? "--" : `${(Number(timestamp) / 1000).toFixed(2)}s`;
  row.innerHTML = `<span class="time">${escapeHtml(time)}</span><span class="kind">${escapeHtml(kind)}</span><span class="detail">${escapeHtml(detail)}</span>`;
  activity.appendChild(row);
  activity.scrollTop = activity.scrollHeight;
}

function renderState(state) {
  lastState = state;
  $("intent").textContent = state.intent || "—";
  $("version").textContent = state.version ?? 0;
  $("epoch").textContent = state.epoch ?? 0;
  const entries = Object.entries(state.slots || {});
  if (!entries.length) {
    $("slots").className = "slots empty";
    $("slots").textContent = "No slots yet";
  } else {
    $("slots").className = "slots";
    $("slots").innerHTML = entries.map(([k, v]) => `<div class="slot"><b>${escapeHtml(k)}</b><span>${escapeHtml(typeof v === "object" ? JSON.stringify(v) : v)}</span></div>`).join("");
  }
}

function renderTools() {
  if (!activeTools.size) {
    tools.className = "tools empty";
    tools.textContent = "No active operations";
    return;
  }
  tools.className = "tools";
  tools.innerHTML = [...activeTools.values()].reverse().map((t) => `
    <div class="tool ${escapeHtml(t.status)}">
      <div class="tool-top"><span class="tool-name">${escapeHtml(t.name)}</span><span class="tool-status">${escapeHtml(t.status.toUpperCase())}</span></div>
      <div class="tool-args">${escapeHtml(JSON.stringify(t.args, null, 2))}</div>
    </div>`).join("");
}

async function refreshState() {
  try {
    const res = await fetch("/api/state", { cache: "no-store" });
    if (res.ok) renderState(await res.json());
  } catch (_) {}
}

function handleAction(action) {
  const type = action.type;
  const p = action.payload || {};
  activityRow(type.toUpperCase(), describeAction(type, p), action.timestamp_ms);

  if (type === "ack") {
    addBubble("agent", "AGENT", p.text || "Working on it...");
    setStatus("working", "WORKING");
  } else if (type === "progress") {
    addBubble("agent", "PROGRESS", p.text || "Working...");
    setStatus("working", "WORKING");
  } else if (type === "tool_call") {
    activeTools.set(p.call_id, { name: p.tool, args: p.arguments || {}, status: "running" });
    setStatus("working", "WORKING");
    renderTools();
  } else if (type === "cancel") {
    const t = activeTools.get(p.call_id);
    if (t) { t.status = "cancelled"; activeTools.set(p.call_id, t); }
    setStatus("interrupted", "INTERRUPTED");
    renderTools();
    addBubble("system", "INTERRUPTION", `Cancelled ${p.call_id || "the previous operation"}. New epoch: ${p.new_epoch ?? "—"}.`);
  } else if (type === "state_snapshot") {
    const s = p.state || {};
    renderState({ ...lastState, ...s, epoch: s.epoch ?? lastState.epoch });
  } else if (type === "final") {
    addBubble("agent", "FINAL", p.text || "Done.");
    setStatus("idle", "READY");
    for (const [id, t] of activeTools) {
      if (t.status === "running") { t.status = "done"; activeTools.set(id, t); }
    }
    renderTools();
  } else if (type === "clarification") {
    addBubble("agent", "CLARIFICATION", p.question || "Could you clarify?");
    setStatus("idle", "WAITING");
  }
}

function describeAction(type, p) {
  if (type === "ack") return p.text || "Acknowledged";
  if (type === "tool_call") return `${p.tool || "tool"}(${JSON.stringify(p.arguments || {})})`;
  if (type === "cancel") return `${p.call_id || "call"} invalidated · epoch ${p.old_epoch ?? "?"} → ${p.new_epoch ?? "?"}`;
  if (type === "final") return p.text || "Completed";
  if (type === "state_snapshot") return `state v${p.state?.version ?? "?"} · intent=${p.state?.intent || "—"}`;
  if (type === "clarification") return p.question || "Clarification requested";
  return p.text || JSON.stringify(p);
}

function handleTrace(t) {
  const important = new Set(["tool_started", "tool_result_accepted", "tool_cancelled", "stale_result_rejected", "tool_failed", "interruption_replan"]);
  if (!important.has(t.kind)) return;
  let detail = t.note || t.tool || "";
  if (t.kind === "stale_result_rejected") {
    detail = `STALE RESULT REJECTED · ${t.tool || "tool"} · ${t.note || "old operation"}`;
    setStatus("interrupted", "STALE RESULT REJECTED");
    addBubble("system", "STATE GUARD", "A late result from the old request was rejected.");
  }
  if (t.kind === "interruption_replan") {
    detail = `Replanning after interruption · ${t.note || ""}`;
  }
  activityRow(t.kind.replaceAll("_", " ").toUpperCase(), detail, t.timestamp_ms);
}

const source = new EventSource("/api/events");
source.onmessage = (event) => {
  try {
    const msg = JSON.parse(event.data);
    if (msg.kind === "connected") renderState(msg.state);
    if (msg.kind === "action") handleAction(msg.action);
    if (msg.kind === "trace") handleTrace(msg.trace);
    if (msg.kind === "user") {
      addBubble(msg.mode === "interrupt" ? "interrupt" : "user", msg.mode === "interrupt" ? "USER · INTERRUPT" : "USER", msg.text);
      activityRow(msg.mode === "interrupt" ? "INTERRUPT" : "USER", msg.text);
      if (msg.mode === "interrupt") setStatus("interrupted", "INTERRUPTED");
    }
    if (msg.kind === "system") activityRow("SYSTEM", msg.text);
  } catch (_) {}
};
source.onerror = () => setStatus("idle", "RECONNECTING");

$("messageForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("messageInput");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  await fetch("/api/message", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
});

$("interruptButton").addEventListener("click", async () => {
  const input = $("interruptInput");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  await fetch("/api/interrupt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
});

$("resetButton").addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  conversation.innerHTML = "";
  activity.innerHTML = "";
  activeTools.clear();
  renderTools();
  await refreshState();
  setStatus("idle", "READY");
});

setInterval(refreshState, 500);
refreshState();

const els = {
  thread: document.getElementById("thread"),
  emptyState: document.getElementById("emptyState"),
  form: document.getElementById("composerForm"),
  input: document.getElementById("questionInput"),
  sendBtn: document.getElementById("sendBtn"),
  sessionsList: document.getElementById("sessionsList"),
  newChatBtn: document.getElementById("newChatBtn"),
  chatTitle: document.getElementById("chatTitle"),
  chatMeta: document.getElementById("chatMeta"),
  uploadZone: document.getElementById("uploadZone"),
  fileInput: document.getElementById("fileInput"),
  uploadStatus: document.getElementById("uploadStatus"),
  userTpl: document.getElementById("msgUserTpl"),
  assistantTpl: document.getElementById("msgAssistantTpl"),
};

let state = {
  sessionId: localStorage.getItem("folio_session_id") || null,
  sending: false,
  uploading: false,
};

// ---------- Sessions ----------

async function loadSessions() {
  const res = await fetch("/sessions");
  const sessions = await res.json();
  renderSessions(sessions);
  return sessions;
}

function renderSessions(sessions) {
  els.sessionsList.innerHTML = "";
  for (const s of sessions) {
    const item = document.createElement("div");
    item.className = "session-item" + (s.id === state.sessionId ? " active" : "");
    item.innerHTML = `<span class="title"></span><button class="del" title="Delete">×</button>`;
    item.querySelector(".title").textContent = s.title || "New chat";
    item.addEventListener("click", (e) => {
      if (e.target.closest(".del")) return;
      openSession(s.id, s.title);
    });
    item.querySelector(".del").addEventListener("click", async (e) => {
      e.stopPropagation();
      await fetch(`/sessions/${s.id}`, { method: "DELETE" });
      if (state.sessionId === s.id) startNewChat();
      loadSessions();
    });
    els.sessionsList.appendChild(item);
  }
}

async function openSession(id, title) {
  state.sessionId = id;
  localStorage.setItem("folio_session_id", id);
  els.chatTitle.textContent = title || "Conversation";
  els.chatMeta.textContent = "Continuing this conversation — Folio remembers earlier turns";
  els.thread.innerHTML = "";
  els.thread.appendChild(els.emptyState);
  els.emptyState.hidden = true;

  const res = await fetch(`/sessions/${id}/messages`);
  if (!res.ok) return;
  const messages = await res.json();
  if (messages.length === 0) {
    els.emptyState.hidden = false;
  } else {
    for (const m of messages) {
      if (m.role === "user") appendUserMessage(m.content);
      else appendAssistantMessage(m.content, m.sources || []);
    }
    scrollToBottom();
  }
  loadSessions();
}

function startNewChat() {
  state.sessionId = null;
  localStorage.removeItem("folio_session_id");
  els.chatTitle.textContent = "New conversation";
  els.chatMeta.textContent = "Ask about the tables, figures, or text in an ingested PDF";
  els.thread.innerHTML = "";
  els.thread.appendChild(els.emptyState);
  els.emptyState.hidden = false;
  loadSessions();
}

els.newChatBtn.addEventListener("click", startNewChat);

// ---------- Message rendering ----------

function scrollToBottom() {
  els.thread.scrollTop = els.thread.scrollHeight;
}

function appendUserMessage(text) {
  els.emptyState.hidden = true;
  const node = els.userTpl.content.cloneNode(true);
  node.querySelector(".bubble").textContent = text;
  els.thread.appendChild(node);
  scrollToBottom();
}

function appendAssistantMessage(text, sources) {
  els.emptyState.hidden = true;
  const node = els.assistantTpl.content.cloneNode(true);
  const wrap = node.querySelector(".msg-assistant");
  wrap.querySelector(".page-content").textContent = stripSourcesLine(text);
  renderCitations(wrap.querySelector(".citations"), sources);
  els.thread.appendChild(node);
  scrollToBottom();
  return wrap;
}

function appendPendingAssistantMessage() {
  els.emptyState.hidden = true;
  const node = els.assistantTpl.content.cloneNode(true);
  const wrap = node.querySelector(".msg-assistant");
  const content = wrap.querySelector(".page-content");
  content.innerHTML = `<span class="typing-dots"><span></span><span></span><span></span></span>`;
  els.thread.appendChild(node);
  scrollToBottom();
  return wrap;
}

function renderCitations(container, sources) {
  container.innerHTML = "";
  const seen = new Set();
  for (const s of sources || []) {
    const key = `${s.kind}-${s.page}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const tab = document.createElement("span");
    tab.className = "citation-tab";
    tab.innerHTML = `<span class="kind"></span><span class="page"></span>`;
    tab.querySelector(".kind").textContent = s.kind || "text";
    tab.querySelector(".page").textContent = `p.${s.page ?? "?"}`;
    container.appendChild(tab);
  }
}

// The model is asked to append a "Sources:" line itself; we render
// citations as tabs instead, so strip that trailing line if present.
function stripSourcesLine(text) {
  return text.replace(/\n?Sources:.*$/s, "").trim();
}

// ---------- Chat submit (SSE) ----------

els.input.addEventListener("input", () => {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 160) + "px";
});

els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.form.requestSubmit();
  }
});

els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = els.input.value.trim();
  if (!question || state.sending) return;

  els.input.value = "";
  els.input.style.height = "auto";
  appendUserMessage(question);
  await sendQuestion(question);
});

async function sendQuestion(question) {
  state.sending = true;
  els.sendBtn.disabled = true;
  const assistantWrap = appendPendingAssistantMessage();
  const contentEl = assistantWrap.querySelector(".page-content");
  const citationsEl = assistantWrap.querySelector(".citations");

  try {
    const res = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: state.sessionId }),
    });
    if (!res.ok || !res.body) throw new Error(`Server responded ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let started = false;
    let full = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const rawEvent = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const { event, data } = parseSSE(rawEvent);
        if (!event) continue;
        const payload = data ? JSON.parse(data) : {};

        if (event === "session") {
          state.sessionId = payload.session_id;
          localStorage.setItem("folio_session_id", payload.session_id);
        } else if (event === "chunk") {
          if (!started) { contentEl.textContent = ""; started = true; }
          full += payload.text;
          contentEl.textContent = full;
          scrollToBottom();
        } else if (event === "done") {
          renderCitations(citationsEl, payload.sources);
          loadSessions();
        } else if (event === "error") {
          throw new Error(payload.error || "Unknown error");
        }
      }
    }
    if (!started) contentEl.textContent = "(No answer returned.)";
  } catch (err) {
    contentEl.textContent = `Something went wrong: ${err.message}`;
    assistantWrap.closest(".msg").classList.add("msg-error");
  } finally {
    state.sending = false;
    els.sendBtn.disabled = false;
    els.input.focus();
  }
}

function parseSSE(raw) {
  let event = null, data = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  return { event, data };
}

// ---------- Upload ----------

els.uploadZone.addEventListener("click", () => { if (!state.uploading) els.fileInput.click(); });
els.fileInput.addEventListener("change", () => {
  if (els.fileInput.files[0]) handleUpload(els.fileInput.files[0]);
});
["dragover", "dragenter"].forEach((evt) =>
  els.uploadZone.addEventListener(evt, (e) => {
    e.preventDefault();
    if (!state.uploading) els.uploadZone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  els.uploadZone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.uploadZone.classList.remove("dragover");
  })
);
els.uploadZone.addEventListener("drop", (e) => {
  if (state.uploading) return;
  const file = e.dataTransfer.files[0];
  if (file) handleUpload(file);
});

function setUploadStatus(html, cls) {
  els.uploadStatus.hidden = false;
  els.uploadStatus.className = "upload-status" + (cls ? " " + cls : "");
  els.uploadStatus.innerHTML = html;
}

async function handleUpload(file) {
  if (state.uploading) return; // ingestion is synchronous and can take a while — avoid overlapping uploads
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    setUploadStatus(`<span class="fname">${escapeHtml(file.name)}</span>Only PDF files are supported.`, "error");
    return;
  }
  state.uploading = true;
  els.uploadZone.classList.add("uploading");
  setUploadStatus(
    `<span class="fname">${escapeHtml(file.name)}</span>` +
    `Replacing current document & indexing (text, tables, images)… this can take a moment.`
  );

  const form = new FormData();
  form.append("file", file);

  try {
    // /upload is synchronous: it clears the previous document, resets
    // the index, and only responds once ingestion is fully done — so
    // there's no job to poll, just one request/response.
    const res = await fetch("/upload", { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Upload failed (${res.status})`);
    }
    const result = await res.json();
    setUploadStatus(
      `<span class="fname">${escapeHtml(result.filename)}</span>` +
      `Indexed — ${result.texts_indexed ?? 0} text chunks, ${result.tables_indexed ?? 0} tables, ${result.images_indexed ?? 0} images.`,
      "ok"
    );
  } catch (err) {
    setUploadStatus(`<span class="fname">${escapeHtml(file.name)}</span>${escapeHtml(err.message)}`, "error");
  } finally {
    state.uploading = false;
    els.uploadZone.classList.remove("uploading");
  }
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------- Init ----------

(async function init() {
  const sessions = await loadSessions();
  if (state.sessionId && sessions.some((s) => s.id === state.sessionId)) {
    const s = sessions.find((s) => s.id === state.sessionId);
    openSession(s.id, s.title);
  } else {
    startNewChat();
  }
})();

// All calls hit the same public API any client uses; the UI holds no privileged path.

async function jsonOrThrow(res) {
  if (!res.ok) {
    let msg = res.statusText;
    try {
      msg = (await res.json()).error?.message || msg;
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

const postJSON = (path, body) =>
  fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

export const health = () => fetch("/healthz").then(jsonOrThrow);
export const listModels = () => fetch("/v1/models").then(jsonOrThrow).then((d) => d.data);
export const metricsSummary = () => fetch("/metrics/summary").then(jsonOrThrow);

export const loadModel = (name) => postJSON("/admin/models/load", { served_name: name }).then(jsonOrThrow);
export const unloadModel = (name) => postJSON("/admin/models/unload", { served_name: name }).then(jsonOrThrow);
export const pinModel = (name, pinned) =>
  postJSON("/admin/models/pin", { served_name: name, pinned }).then(jsonOrThrow);

export const ragQuery = (query) => postJSON("/rag/query", { query }).then(jsonOrThrow);
export const ragDocuments = () => fetch("/rag/documents").then(jsonOrThrow).then((d) => d.documents);

export async function uploadDocs(files) {
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  return jsonOrThrow(await fetch("/rag/upload", { method: "POST", body: form }));
}

// Stream a chat completion, calling onDelta(text) for each content chunk.
export async function streamChat({ model, messages, signal, onDelta }) {
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model, messages, stream: true, max_tokens: 1024 }),
    signal,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      msg = (await res.json()).error?.message || msg;
    } catch {}
    throw new Error(msg);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") return;
      try {
        const delta = JSON.parse(data).choices?.[0]?.delta?.content;
        if (delta) onDelta(delta);
      } catch {}
    }
  }
}

// Read a File as a base64 data URL for inline image messages (routed to the VLM).
export const fileToDataUrl = (file) =>
  new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });

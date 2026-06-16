import React, { useEffect, useRef, useState } from "react";
import { fileToDataUrl, ragQuery, streamChat, uploadDocs } from "./api.js";

const isImage = (f) => f.type.startsWith("image/");

export default function ChatView({ models, hasVision, hasRag }) {
  const [model, setModel] = useState("");
  const [grounded, setGrounded] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [images, setImages] = useState([]); // {name, url}
  const [docs, setDocs] = useState([]); // indexed doc names
  const [busy, setBusy] = useState(false);
  const fileRef = useRef(null);
  const endRef = useRef(null);

  useEffect(() => {
    if (!model && models.length) setModel(models[0].id);
  }, [models, model]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const openaiMessages = (history) =>
    history.map((m) => {
      if (m.role === "user" && m.images?.length) {
        return {
          role: "user",
          content: [
            { type: "text", text: m.text },
            ...m.images.map((u) => ({ type: "image_url", image_url: { url: u } })),
          ],
        };
      }
      return { role: m.role, content: m.text };
    });

  async function send() {
    const text = input.trim();
    if ((!text && !images.length) || busy || !model) return;
    const user = { role: "user", text, images: images.map((i) => i.url) };
    const history = [...messages, user];
    setMessages([...history, { role: "assistant", text: "", sources: null }]);
    setInput("");
    setImages([]);
    setBusy(true);

    const setAssistant = (fn) =>
      setMessages((cur) => {
        const next = cur.slice();
        next[next.length - 1] = fn(next[next.length - 1]);
        return next;
      });

    try {
      if (grounded && hasRag) {
        const res = await ragQuery(text);
        setAssistant((a) => ({ ...a, text: res.answer, sources: res.sources }));
      } else {
        await streamChat({
          model,
          thinking,
          messages: openaiMessages(history),
          onDelta: (chunk) => setAssistant((a) => ({ ...a, text: a.text + chunk })),
        });
      }
    } catch (e) {
      setAssistant((a) => ({ ...a, text: (a.text || "") + `\n\n⚠ ${e.message}` }));
    } finally {
      setBusy(false);
    }
  }

  async function onFiles(fileList) {
    const files = [...fileList];
    const imgs = files.filter(isImage);
    const documents = files.filter((f) => !isImage(f));
    for (const f of imgs) {
      const url = await fileToDataUrl(f);
      setImages((cur) => [...cur, { name: f.name, url }]);
    }
    if (documents.length) {
      try {
        const res = await uploadDocs(documents);
        const names = res.documents.map((d) => d.source.split("/").pop());
        setDocs((cur) => [...new Set([...cur, ...names])]); // dedupe re-uploads
        setGrounded(true);
        if (names.length) {
          setMessages((cur) => [
            ...cur,
            { role: "note", text: `📎 Indexed ${names.join(", ")} — ask with Grounded on.` },
          ]);
        }
      } catch (e) {
        setMessages((cur) => [...cur, { role: "assistant", text: `⚠ upload failed: ${e.message}` }]);
      }
    }
  }

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">Ask anything. Attach an image to use vision, or a document to ground answers.</div>
        )}
        {messages.map((m, i) => (
          <Message key={i} m={m} />
        ))}
        <div ref={endRef} />
      </div>

      <div className="composer">
        <div className="composer-row">
          <select value={model} onChange={(e) => setModel(e.target.value)} title="Model">
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.id}
                {m.type === "vlm" ? " (vision)" : ""}
              </option>
            ))}
          </select>
          <label className="toggle" title="Show the model's reasoning (<think>) — for reasoning models like Qwen3">
            <input type="checkbox" checked={thinking} onChange={(e) => setThinking(e.target.checked)} />
            Thinking
          </label>
          {hasRag && (
            <label className="toggle" title="Answer from your uploaded documents with citations">
              <input type="checkbox" checked={grounded} onChange={(e) => setGrounded(e.target.checked)} />
              Grounded
            </label>
          )}
          {docs.length > 0 && (
            <span className="chips" title={docs.join(", ")}>
              📎 indexed: {docs.join(", ")}
            </span>
          )}
        </div>

        {images.length > 0 && (
          <div className="thumbs">
            {images.map((im, i) => (
              <span key={i} className="thumb">
                <img src={im.url} alt={im.name} />
                <button onClick={() => setImages(images.filter((_, j) => j !== i))}>×</button>
              </span>
            ))}
          </div>
        )}

        <div className="composer-row">
          <button className="ghost" onClick={() => fileRef.current?.click()} title="Attach image or document">
            📎
          </button>
          <input
            ref={fileRef}
            type="file"
            multiple
            accept={(hasVision ? "image/*," : "") + ".pdf,.txt,.md,.markdown"}
            style={{ display: "none" }}
            onChange={(e) => {
              onFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            placeholder={grounded ? "Ask about your documents…" : "Type a message…"}
            rows={1}
          />
          <button className="send" disabled={busy} onClick={send}>
            {busy ? "…" : "▷"}
          </button>
        </div>
      </div>
    </div>
  );
}

function splitThink(text) {
  if (!text.includes("<think>")) return { think: null, answer: text };
  const start = text.indexOf("<think>") + "<think>".length;
  const end = text.indexOf("</think>");
  if (end === -1) return { think: text.slice(start), answer: "", open: true }; // still thinking
  return { think: text.slice(start, end).trim(), answer: text.slice(end + 8).trimStart() };
}

function Message({ m }) {
  if (m.role === "note") {
    return <div className="note">{m.text}</div>;
  }
  const { think, answer, open } = m.role === "assistant" ? splitThink(m.text) : { answer: m.text };
  return (
    <div className={`msg ${m.role}`}>
      <div className="role">{m.role === "user" ? "you" : "assistant"}</div>
      <div className="bubble">
        {m.images?.map((u, i) => <img key={i} className="msg-img" src={u} alt="" />)}
        {think != null && (
          <details className="think" open={open || !answer}>
            <summary>💭 Reasoning</summary>
            <div className="think-text">{think}</div>
          </details>
        )}
        <div className="text">{answer}</div>
        {m.sources?.length > 0 && (
          <div className="sources">
            <div className="sources-title">Sources</div>
            {m.sources.map((s) => (
              <details key={s.n}>
                <summary>
                  [{s.n}] {s.source.split("/").pop()} · {Math.round(s.score * 100) / 100}
                </summary>
                <div className="src-text">{s.text}</div>
              </details>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

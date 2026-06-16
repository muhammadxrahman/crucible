import React, { useEffect, useState } from "react";
import {
  addModel,
  availableModels,
  loadModel,
  metricsSummary,
  pinModel,
  shutdownServer,
  unloadModel,
} from "./api.js";

const basename = (p) => (p ? p.split("/").pop() : "");
const slug = (p) =>
  basename(p)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

export default function SidePanel({ models, hw, onRefresh, onCollapse, onShutdown }) {
  const [metrics, setMetrics] = useState(null);
  const [acting, setActing] = useState("");
  const [showAdd, setShowAdd] = useState(false);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const m = await metricsSummary();
        if (alive) setMetrics(m);
      } catch {}
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const act = async (fn, name) => {
    setActing(name);
    try {
      await fn();
      await onRefresh();
    } finally {
      setActing("");
    }
  };

  const cur = metrics?.current || {};
  const ceiling = hw?.memory_ceiling_gb || 0;
  const resident = hw?.resident_gb || 0;
  const pct = ceiling ? Math.min(100, (resident / ceiling) * 100) : 0;

  return (
    <aside className="side">
      <div className="side-head">
        <span>models</span>
        <span className="side-head-actions">
          <button className="ghost" onClick={() => setShowAdd((v) => !v)} title="Add a downloaded model">
            {showAdd ? "×" : "＋"}
          </button>
          <button className="ghost" onClick={onCollapse} title="Hide panel">
            ‹
          </button>
        </span>
      </div>

      {showAdd && (
        <AddModelPanel
          existing={models}
          onAdded={async () => {
            await onRefresh();
            setShowAdd(false);
          }}
        />
      )}

      <div className="models">
        {models.map((m) => (
          <div key={m.id} className="model-row">
            <span className={`dot ${m.state}`} title={m.error || m.state} />
            <span className="model-name" title={`${m.type} · ${m.path || ""}`}>
              {m.id}
              {m.path && <span className="model-path">{basename(m.path)}</span>}
            </span>
            <span className="model-actions">
              {m.state === "loading" ? (
                <span className="loading-tag">loading…</span>
              ) : m.state === "resident" ? (
                <button disabled={acting === m.id} onClick={() => act(() => unloadModel(m.id), m.id)}>
                  unload
                </button>
              ) : (
                <button disabled={acting === m.id} onClick={() => act(() => loadModel(m.id), m.id)}>
                  load
                </button>
              )}
              <button
                className={m.pinned ? "pinned" : ""}
                disabled={acting === m.id || m.state === "loading"}
                onClick={() => act(() => pinModel(m.id, !m.pinned), m.id)}
                title={m.pinned ? "unpin" : "pin"}
              >
                ★
              </button>
            </span>
          </div>
        ))}
      </div>

      <div className="side-section">
        <div className="label">profile</div>
        <div className="value">{hw?.profile || "…"}</div>
        <div className="bar">
          <div style={{ width: `${pct}%` }} />
        </div>
        <div className="muted">
          {resident.toFixed(1)} / {ceiling.toFixed(0)} GB resident
        </div>
      </div>

      <div className="side-section">
        <div className="label">throughput</div>
        <div className="stat">
          <span>decode</span>
          <span>{(cur.decode_tps || 0).toFixed(0)} tok/s</span>
        </div>
        <div className="stat">
          <span>prefill</span>
          <span>{(cur.prefill_tps || 0).toFixed(0)} tok/s</span>
        </div>
        <div className="stat">
          <span>TTFT</span>
          <span>{(cur.ttft_ms || 0).toFixed(0)} ms</span>
        </div>
        <div className="stat">
          <span>batch</span>
          <span>{cur.batch_size || 0}</span>
        </div>
      </div>

      <a className="side-link" href="/observability" target="_blank" rel="noreferrer">
        full dashboard →
      </a>

      <button
        className="shutdown"
        title="Gracefully stop the server"
        onClick={async () => {
          if (!confirm("Shut down the Crucible server?")) return;
          try {
            await shutdownServer();
          } catch {
            // the server may drop the connection as it stops — treat as success
          }
          onShutdown?.();
        }}
      >
        ⏻ Shut down server
      </button>
    </aside>
  );
}

function AddModelPanel({ existing, onAdded }) {
  const [cache, setCache] = useState(null);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  // Per-row editable type / served_name keyed by repo_id.
  const [edits, setEdits] = useState({});

  useEffect(() => {
    availableModels()
      .then(setCache)
      .catch((e) => setErr(e.message));
  }, []);

  const usedNames = new Set(existing.map((m) => m.id));

  const add = async (repo) => {
    const e = edits[repo.repo_id] || {};
    const served_name = (e.served_name ?? slug(repo.repo_id)).trim();
    const type = e.type ?? repo.guessed_type;
    if (!served_name) return setErr("name required");
    if (usedNames.has(served_name)) return setErr(`name "${served_name}" already in use`);
    setErr("");
    setBusy(repo.repo_id);
    try {
      await addModel({ path: repo.repo_id, type, served_name });
      await onAdded();
    } catch (ex) {
      setErr(ex.message);
    } finally {
      setBusy("");
    }
  };

  if (err && !cache) return <div className="add-panel error">{err}</div>;
  if (!cache) return <div className="add-panel muted">scanning cache…</div>;
  if (cache.length === 0)
    return <div className="add-panel muted">no downloaded MLX models found</div>;

  return (
    <div className="add-panel">
      <div className="add-hint">downloaded models — pick a type and add</div>
      {err && <div className="add-err">{err}</div>}
      {cache.map((r) => {
        const e = edits[r.repo_id] || {};
        const set = (patch) => setEdits((s) => ({ ...s, [r.repo_id]: { ...e, ...patch } }));
        return (
          <div key={r.repo_id} className="add-row">
            <div className="add-top">
              <span className="add-repo" title={r.repo_id}>
                {r.repo_id}
              </span>
              <span className="add-size">{r.size_str}</span>
            </div>
            {r.registered ? (
              <div className="add-bottom muted">already registered</div>
            ) : (
              <div className="add-bottom">
                <select value={e.type ?? r.guessed_type} onChange={(ev) => set({ type: ev.target.value })}>
                  <option value="lm">lm</option>
                  <option value="vlm">vlm</option>
                  <option value="embedding">embedding</option>
                  <option value="rerank">rerank</option>
                </select>
                <input
                  className="add-name"
                  value={e.served_name ?? slug(r.repo_id)}
                  onChange={(ev) => set({ served_name: ev.target.value })}
                  spellCheck={false}
                />
                <button disabled={busy === r.repo_id} onClick={() => add(r)}>
                  {busy === r.repo_id ? "…" : "add"}
                </button>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

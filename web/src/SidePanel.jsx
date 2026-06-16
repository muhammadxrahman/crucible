import React, { useEffect, useState } from "react";
import { loadModel, metricsSummary, pinModel, unloadModel } from "./api.js";

export default function SidePanel({ models, hw, onRefresh, onCollapse }) {
  const [metrics, setMetrics] = useState(null);
  const [acting, setActing] = useState("");

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
        <button className="ghost" onClick={onCollapse} title="Hide panel">
          ‹
        </button>
      </div>

      <div className="models">
        {models.map((m) => (
          <div key={m.id} className="model-row">
            <span className={`dot ${m.state}`} />
            <span className="model-name" title={m.type}>
              {m.id}
            </span>
            <span className="model-actions">
              {m.state === "resident" ? (
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
                disabled={acting === m.id}
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
    </aside>
  );
}

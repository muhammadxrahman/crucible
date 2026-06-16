import React, { useCallback, useEffect, useState } from "react";
import ChatView from "./ChatView.jsx";
import SidePanel from "./SidePanel.jsx";
import { health, listModels } from "./api.js";

export default function App() {
  const [models, setModels] = useState([]);
  const [hw, setHw] = useState(null);
  const [collapsed, setCollapsed] = useState(false);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const [m, h] = await Promise.all([listModels(), health()]);
      setModels(m);
      setHw(h);
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const chatModels = models.filter((m) => m.type === "lm" || m.type === "vlm");
  const hasVision = models.some((m) => m.type === "vlm");
  const hasRag = models.some((m) => m.type === "embedding");

  return (
    <div className="app">
      {!collapsed && (
        <SidePanel models={models} hw={hw} onRefresh={refresh} onCollapse={() => setCollapsed(true)} />
      )}
      <main className="main">
        <header className="topbar">
          {collapsed && (
            <button className="ghost" onClick={() => setCollapsed(false)} title="Show panel">
              ☰
            </button>
          )}
          <span className="brand">CRUCIBLE</span>
          {error && <span className="err">· {error}</span>}
        </header>
        <ChatView models={chatModels} hasVision={hasVision} hasRag={hasRag} />
      </main>
    </div>
  );
}

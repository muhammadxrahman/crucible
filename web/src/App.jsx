import React, { useCallback, useEffect, useState } from "react";
import ChatView from "./ChatView.jsx";
import SidePanel from "./SidePanel.jsx";
import { deleteSession, getSession, health, listModels, listSessions } from "./api.js";

export default function App() {
  const [models, setModels] = useState([]);
  const [hw, setHw] = useState(null);
  const [collapsed, setCollapsed] = useState(false);
  const [error, setError] = useState(null);
  const [stopped, setStopped] = useState(false);

  const [historyEnabled, setHistoryEnabled] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [active, setActive] = useState(null); // { id, messages: [{role,text}] } | null

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

  const reloadSessions = useCallback(async () => {
    try {
      setSessions(await listSessions());
      setHistoryEnabled(true);
    } catch {
      setHistoryEnabled(false); // server has history disabled (503) — hide the Chats UI
    }
  }, []);

  useEffect(() => {
    refresh();
    reloadSessions();
  }, [refresh, reloadSessions]);

  const newChat = () => setActive(null);

  const openSession = async (id) => {
    const s = await getSession(id);
    setActive({ id: s.id, messages: s.messages.map((m) => ({ role: m.role, text: m.content })) });
  };

  const removeSession = async (id) => {
    await deleteSession(id);
    if (active?.id === id) setActive(null);
    reloadSessions();
  };

  const onSessionCreated = (s) => {
    setActive({ id: s.id, messages: null }); // ChatView already holds the live messages
    reloadSessions();
  };

  const chatModels = models.filter((m) => m.type === "lm" || m.type === "vlm");
  const hasVision = models.some((m) => m.type === "vlm");
  const hasRag = models.some((m) => m.type === "embedding");

  if (stopped) {
    return (
      <div className="stopped-overlay">
        <div>
          <h1>Crucible server stopped</h1>
          <p>You can close this tab. Restart with `uv run mlxd serve` in your terminal.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      {!collapsed && (
        <SidePanel
          models={models}
          hw={hw}
          historyEnabled={historyEnabled}
          sessions={sessions}
          activeSessionId={active?.id}
          onNewChat={newChat}
          onSelectSession={openSession}
          onDeleteSession={removeSession}
          onRefresh={refresh}
          onCollapse={() => setCollapsed(true)}
          onShutdown={() => setStopped(true)}
        />
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
        <ChatView
          models={chatModels}
          hasVision={hasVision}
          hasRag={hasRag}
          historyEnabled={historyEnabled}
          sessionId={active?.id}
          initialMessages={active?.messages}
          onSessionCreated={onSessionCreated}
          onActivity={reloadSessions}
        />
      </main>
    </div>
  );
}

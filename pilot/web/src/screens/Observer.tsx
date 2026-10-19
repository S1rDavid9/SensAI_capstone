import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, load, researcherHeaders, save, serverNow, syncClock } from "../api";
import { OBSERVER_CODES } from "../states";

type Session = { id: number; participant_code: string; status: string; current_index: number };

// Momentary time sampling: every INTERVAL seconds the app names the next
// learner in rotation, the observer watches them briefly and taps one code.
const INTERVAL_S = 20;

export default function Observer() {
  const [key, setKey] = useState(() => load("sensai.key") ?? "");
  const [name, setName] = useState(() => load("sensai.observer") ?? "");
  const [ready, setReady] = useState(false);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selected, setSelected] = useState<number[]>([]);
  const [running, setRunning] = useState(false);
  const [turn, setTurn] = useState(0);
  const [secondsLeft, setSecondsLeft] = useState(INTERVAL_S);
  const [log, setLog] = useState<{ code: string; learner: string; at: string }[]>([]);
  const [error, setError] = useState<string | null>(null);
  const deadline = useRef(0);

  const refresh = useCallback(async () => {
    try {
      const list = await api<Session[]>("/api/observer/sessions", undefined, researcherHeaders(key));
      setSessions(list);
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Can't reach the server.");
      if (err instanceof ApiError && err.status === 401) setReady(false);
    }
  }, [key]);

  useEffect(() => {
    if (!ready) return;
    syncClock().catch(() => {});
    refresh();
    const t = window.setInterval(refresh, 10_000);
    return () => window.clearInterval(t);
  }, [ready, refresh]);

  const rotation = sessions.filter((s) => selected.includes(s.id));
  const target = rotation.length ? rotation[turn % rotation.length] : null;

  useEffect(() => {
    if (!running) return;
    deadline.current = Date.now() + INTERVAL_S * 1000;
    const t = window.setInterval(() => {
      const left = Math.max(0, Math.ceil((deadline.current - Date.now()) / 1000));
      setSecondsLeft(left);
      if (left === 0) {
        if (navigator.vibrate) navigator.vibrate(150);
      }
    }, 250);
    return () => window.clearInterval(t);
  }, [running, turn]);

  async function code(c: string) {
    if (!target) return;
    try {
      await api(
        "/api/observer/observations",
        { session_id: target.id, observer: name, code: c, client_ms: serverNow() },
        researcherHeaders(key),
      );
      setLog((l) => [{ code: c, learner: target.participant_code, at: new Date().toLocaleTimeString() }, ...l].slice(0, 8));
      setTurn((n) => n + 1);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Not saved. Tap again.");
    }
  }

  if (!ready) {
    return (
      <main className="observer">
        <form
          className="card"
          onSubmit={(e) => {
            e.preventDefault();
            save("sensai.key", key);
            save("sensai.observer", name);
            setReady(true);
          }}
        >
          <h1>Observer</h1>
          <label>
            Your role
            <select value={name} onChange={(e) => setName(e.target.value)} required>
              <option value="">Choose…</option>
              <option value="researcher">Researcher</option>
              <option value="teacher">Class teacher</option>
            </select>
          </label>
          <label>
            Researcher key
            <input type="password" value={key} onChange={(e) => setKey(e.target.value)} required />
          </label>
          <button className="big" disabled={!key || !name}>Continue</button>
        </form>
      </main>
    );
  }

  return (
    <main className="observer">
      <header className="bar">
        <strong>Observing as {name}</strong>
        <button className="ghost small" onClick={() => setReady(false)}>Change</button>
      </header>
      {error && <p className="error">{error}</p>}

      {!running && (
        <section className="card">
          <h2>Who are you watching?</h2>
          {sessions.length === 0 && <p className="muted">No open sessions yet. Create one on the researcher screen.</p>}
          <div className="chips">
            {sessions.map((s) => (
              <button
                key={s.id}
                className={selected.includes(s.id) ? "chip on" : "chip"}
                onClick={() => setSelected((sel) => (sel.includes(s.id) ? sel.filter((x) => x !== s.id) : [...sel, s.id]))}
              >
                {s.participant_code}
                <small>{s.status === "active" ? `Q${s.current_index + 1}` : "not started"}</small>
              </button>
            ))}
          </div>
          <p className="muted">
            Every {INTERVAL_S} seconds the app names one learner. Watch them for a moment, then tap what you see.
            Both observers should start together so you code the same moments.
          </p>
          <button className="big" disabled={rotation.length === 0} onClick={() => { setTurn(0); setRunning(true); }}>
            Start observing
          </button>
        </section>
      )}

      {running && target && (
        <section className="card">
          <div className="target">
            <span className="muted">Now watch</span>
            <span className="target-code">{target.participant_code}</span>
            <span className={secondsLeft === 0 ? "timer due" : "timer"}>
              {secondsLeft === 0 ? "Code now" : `${secondsLeft}s`}
            </span>
          </div>
          <div className="codes">
            {OBSERVER_CODES.map((c) => (
              <button key={c.code} className="code-btn" style={{ borderColor: c.color }} onClick={() => code(c.code)}>
                <b style={{ color: c.color }}>{c.label}</b>
                <small>{c.hint}</small>
              </button>
            ))}
          </div>
          <button className="ghost" onClick={() => setRunning(false)}>Stop</button>
          {log.length > 0 && (
            <ul className="obs-log">
              {log.map((l, i) => (
                <li key={i}>{l.at} · {l.learner} · {OBSERVER_CODES.find((c) => c.code === l.code)?.label}</li>
              ))}
            </ul>
          )}
        </section>
      )}
    </main>
  );
}

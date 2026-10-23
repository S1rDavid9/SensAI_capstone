import { useCallback, useEffect, useState } from "react";
import { api, ApiError, load, researcherHeaders, save } from "../api";
import { CHECKIN_CHOICES, STATE_LABEL } from "../states";

type Session = {
  id: number;
  participant_code: string;
  status: string;
  mode: string;
  current_index: number;
  answered: number;
  correct: number;
  observations: number;
  consent_recorded_by: string;
  notes: string;
  created_ms: number;
  latest_decision: { question_index: number; state: string; raw_action_name: string; action_name: string; guardrail: string | null } | null;
};

type Detail = {
  answers: { question_index: number; kind: string; difficulty: number; given: string; correct: number; response_time_ms: number; first_input_ms: number | null; hints_used: number }[];
  decisions: { question_index: number; state: string; raw_action_name: string; action_name: string; guardrail: string | null }[];
  checkins: { after_question: number; choice: string }[];
  observations: { observer: string; code: string; server_ms: number }[];
  events: { type: string }[];
};

const TABLES = ["sessions", "answers", "events", "checkins", "observations", "decisions"];

export default function Researcher() {
  const [key, setKey] = useState(() => load("sensai.key") ?? "");
  const [authed, setAuthed] = useState(false);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [openId, setOpenId] = useState<number | null>(null);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({ guardian_consent: false, learner_assent: false, recorded_by: "", notes: "", mode: "adaptive" });
  const [created, setCreated] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setSessions(await api<Session[]>("/api/researcher/sessions", undefined, researcherHeaders(key)));
      setAuthed(true);
      setError(null);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) setAuthed(false);
      setError(err instanceof ApiError ? err.message : "Can't reach the server.");
    }
  }, [key]);

  useEffect(() => {
    if (load("sensai.key")) refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!authed) return;
    const t = window.setInterval(refresh, 5000);
    return () => window.clearInterval(t);
  }, [authed, refresh]);

  useEffect(() => {
    if (openId === null) return;
    let stop = false;
    const get = () =>
      api<Detail>(`/api/researcher/sessions/${openId}`, undefined, researcherHeaders(key))
        .then((d) => !stop && setDetail(d))
        .catch(() => {});
    get();
    const t = window.setInterval(get, 5000);
    return () => {
      stop = true;
      window.clearInterval(t);
    };
  }, [openId, key]);

  async function createSession(e: React.FormEvent) {
    e.preventDefault();
    try {
      const s = await api<{ participant_code: string }>("/api/researcher/sessions", form, researcherHeaders(key));
      setCreated(s.participant_code);
      setForm({ guardian_consent: false, learner_assent: false, recorded_by: form.recorded_by, notes: "", mode: form.mode });
      refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't create the session.");
    }
  }

  async function closeSession(id: number) {
    if (!confirm("Close this session? The learner won't be able to continue it.")) return;
    await api(`/api/researcher/sessions/${id}/close`, {}, researcherHeaders(key));
    refresh();
  }

  async function download(table: string) {
    const res = await fetch(`/api/researcher/export/${table}.csv`, { headers: researcherHeaders(key) });
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement("a");
    a.href = url;
    a.download = `sensai_pilot_${table}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  if (!authed) {
    return (
      <main className="researcher">
        <form className="card narrow" onSubmit={(e) => { e.preventDefault(); save("sensai.key", key); refresh(); }}>
          <h1>Researcher</h1>
          <label>
            Researcher key
            <input type="password" value={key} onChange={(e) => setKey(e.target.value)} autoFocus />
          </label>
          {error && <p className="error">{error}</p>}
          <button className="big" disabled={!key}>Open</button>
        </form>
      </main>
    );
  }

  const decisionsByQ = new Map(detail?.decisions.map((d) => [d.question_index, d]) ?? []);
  const checkinFace = (c: string) => CHECKIN_CHOICES.find((x) => x.choice === c);

  return (
    <main className="researcher">
      <header className="bar">
        <h1 className="brand">Sens<span>AI</span> pilot · researcher</h1>
        <div className="row">
          {TABLES.map((t) => (
            <button key={t} className="ghost small" onClick={() => download(t)}>{t}.csv</button>
          ))}
        </div>
      </header>
      {error && <p className="error">{error}</p>}

      <div className="research-grid">
        <section className="card">
          <h2>New session</h2>
          <p className="muted">
            Only create a session once you hold the signed guardian consent form and the learner has agreed to take part.
            The app gives the learner a code. Write the code next to their name on your paper list; names are never entered here.
          </p>
          <form onSubmit={createSession} className="stack">
            <label className="check">
              <input type="checkbox" checked={form.guardian_consent} onChange={(e) => setForm({ ...form, guardian_consent: e.target.checked })} />
              Signed guardian consent form received
            </label>
            <label className="check">
              <input type="checkbox" checked={form.learner_assent} onChange={(e) => setForm({ ...form, learner_assent: e.target.checked })} />
              Learner has agreed to take part (assent)
            </label>
            <label>
              Recorded by
              <input value={form.recorded_by} onChange={(e) => setForm({ ...form, recorded_by: e.target.value })} placeholder="Your initials" required />
            </label>
            <label>
              Mode
              <select value={form.mode} onChange={(e) => setForm({ ...form, mode: e.target.value })}>
                <option value="adaptive">Adaptive: SensAI adapts the questions (final product)</option>
                <option value="shadow">Pilot: fixed 15 questions, model in shadow mode (research)</option>
              </select>
            </label>
            <label>
              Notes (no names)
              <input value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} placeholder="e.g. Session 1, laptop 3" />
            </label>
            <button className="big" disabled={!form.guardian_consent || !form.learner_assent || !form.recorded_by}>Create session</button>
          </form>
          {created && <p className="created">Learner code: <b>{created}</b></p>}
        </section>

        <section className="card wide">
          <h2>Sessions</h2>
          <table>
            <thead>
              <tr><th>Code</th><th>Mode</th><th>Status</th><th>Progress</th><th>Correct</th><th>Obs.</th><th>Latest shadow decision</th><th></th></tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.id} className={openId === s.id ? "open" : ""} onClick={() => setOpenId(s.id)}>
                  <td><b>{s.participant_code}</b></td>
                  <td>{s.mode === "adaptive" ? "Adaptive" : "Pilot"}</td>
                  <td><span className={`status ${s.status}`}>{s.status}</span></td>
                  <td>{s.answered}/15</td>
                  <td>{s.correct}</td>
                  <td>{s.observations}</td>
                  <td>
                    {s.latest_decision
                      ? `Q${s.latest_decision.question_index + 1}: ${STATE_LABEL[s.latest_decision.state]} → ${s.latest_decision.action_name}${s.latest_decision.guardrail ? " 🛡" : ""}`
                      : "—"}
                  </td>
                  <td>
                    {(s.status === "ready" || s.status === "active") && (
                      <button className="ghost small" onClick={(e) => { e.stopPropagation(); closeSession(s.id); }}>Close</button>
                    )}
                  </td>
                </tr>
              ))}
              {sessions.length === 0 && <tr><td colSpan={8} className="muted">No sessions yet.</td></tr>}
            </tbody>
          </table>
        </section>

        {detail && openId !== null && (
          <section className="card wide">
            <h2>{sessions.find((s) => s.id === openId)?.participant_code}: question by question</h2>
            <p className="muted">
              "State" comes from provisional rules, not a trained model. The DQN action is shadow mode: logged, never shown to the learner.
              Where a guardrail ruled out the DQN's first choice, both are shown.
            </p>
            <table>
              <thead>
                <tr><th>Q</th><th>Type</th><th>Level</th><th>Answer</th><th>Time</th><th>First key</th><th>Hints</th><th>State</th><th>DQN would</th></tr>
              </thead>
              <tbody>
                {detail.answers.map((a) => {
                  const d = decisionsByQ.get(a.question_index);
                  return (
                    <tr key={a.question_index}>
                      <td>{a.question_index + 1}</td>
                      <td>{a.kind}</td>
                      <td>{a.difficulty}</td>
                      <td className={a.correct ? "ok" : "bad"}>{a.given} {a.correct ? "✓" : "✗"}</td>
                      <td>{(a.response_time_ms / 1000).toFixed(1)}s</td>
                      <td>{a.first_input_ms === null ? "—" : `${(a.first_input_ms / 1000).toFixed(1)}s`}</td>
                      <td>{a.hints_used}</td>
                      <td>{d ? STATE_LABEL[d.state] : "—"}</td>
                      <td title={d?.guardrail ?? undefined}>
                        {!d ? "—" : d.guardrail ? <><s className="muted">{d.raw_action_name}</s> → {d.action_name} 🛡</> : d.action_name}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="row wrap">
              <div>
                <h3>Check-ins</h3>
                {detail.checkins.length === 0 ? <p className="muted">None yet</p> : (
                  <ul className="plain">
                    {detail.checkins.map((c, i) => (
                      <li key={i}>After Q{c.after_question + 1}: {checkinFace(c.choice)?.face} {checkinFace(c.choice)?.label}</li>
                    ))}
                  </ul>
                )}
              </div>
              <div>
                <h3>Observations</h3>
                {detail.observations.length === 0 ? <p className="muted">None yet</p> : (
                  <ul className="plain">
                    {detail.observations.slice(-8).map((o, i) => (
                      <li key={i}>{new Date(o.server_ms).toLocaleTimeString()} · {o.observer}: {STATE_LABEL[o.code]}</li>
                    ))}
                  </ul>
                )}
              </div>
              <div>
                <h3>Events logged</h3>
                <p>{detail.events.length}</p>
              </div>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}

import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, load, save, serverNow, syncClock } from "../api";
import Mascot from "../components/Mascot";
import { CHECKIN_CHOICES } from "../states";

type Support = { message: string; hint: string | null };

type Question = {
  id: string;
  index: number;
  kind: string;
  domain: "arithmetic" | "pattern";
  difficulty: number;
  prompt: string;
  instruction: string;
  choices: string[];
};

type Auth = { code: string; token: string; mode?: string };
type Summary = { correct: number; total: number; highest_level: number };
type Phase = "join" | "welcome" | "question" | "feedback" | "checkin" | "done";

const STORAGE_KEY = "sensai.learner";
const IDLE_MS = 20_000;
const ENCOURAGEMENT = ["Nice work!", "Great thinking!", "You got it!", "Well done!", "Brilliant!", "Super!"];

export default function Learner() {
  const [phase, setPhase] = useState<Phase>("join");
  const [auth, setAuth] = useState<Auth | null>(null);
  const [nQuestions, setNQuestions] = useState(15);
  const [question, setQuestion] = useState<Question | null>(null);
  const [answer, setAnswer] = useState("");
  const [hint, setHint] = useState<string | null>(null);
  const [support, setSupport] = useState<Support | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [feedback, setFeedback] = useState<{ correct: boolean; correctAnswer: string; checkinDue: boolean; finished: boolean } | null>(null);
  const [codeInput, setCodeInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Per-question telemetry, kept in refs so typing doesn't re-render timers.
  const shownAt = useRef(0);
  const shownClientMs = useRef(0);
  const firstInputMs = useRef<number | null>(null);
  const lastActivity = useRef(0);
  const keystrokes = useRef(0);
  const deletions = useRef(0);
  const idleCount = useRef(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const adaptive = auth?.mode === "adaptive";

  const logEvent = useCallback(
    (type: string, questionIndex: number | null, payload: Record<string, unknown> = {}) => {
      if (!auth) return;
      api("/api/learner/event", { ...auth, type, question_index: questionIndex, client_ms: serverNow(), payload }).catch(() => {
        /* telemetry is best effort; the answer itself is what must not be lost */
      });
    },
    [auth],
  );

  const loadQuestion = useCallback(async (a: Auth) => {
    const res = await api<{ done: boolean; question?: Question; support?: Support; summary?: Summary }>("/api/learner/question", a);
    if (res.done || !res.question) {
      setSummary(res.summary ?? null);
      setPhase("done");
      await api("/api/learner/finish", a).catch(() => {});
      save(STORAGE_KEY, null);
      return;
    }
    setQuestion(res.question);
    setAnswer("");
    // Adaptive mode: SensAI's response to how the last questions went.
    setSupport(res.support ?? null);
    setHint(res.support?.hint ?? null);
    setFeedback(null);
    setPhase("question");
  }, []);

  // Resume after a page refresh.
  useEffect(() => {
    syncClock().catch(() => {});
    const saved = load(STORAGE_KEY);
    if (!saved) return;
    const a = JSON.parse(saved) as Auth;
    setAuth(a);
    loadQuestion(a).catch(() => {
      save(STORAGE_KEY, null);
      setAuth(null);
    });
  }, [loadQuestion]);

  // Start the clock when a question appears.
  useEffect(() => {
    if (phase !== "question" || !question) return;
    shownAt.current = performance.now();
    shownClientMs.current = serverNow();
    lastActivity.current = shownAt.current;
    firstInputMs.current = null;
    keystrokes.current = 0;
    deletions.current = 0;
    idleCount.current = 0;
    logEvent("question_shown", question.index, { difficulty: question.difficulty, kind: question.kind });
    inputRef.current?.focus();

    const timer = window.setInterval(() => {
      if (performance.now() - lastActivity.current >= IDLE_MS) {
        idleCount.current += 1;
        lastActivity.current = performance.now();
        logEvent("idle", question.index, { seconds: IDLE_MS / 1000, count: idleCount.current });
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [phase, question, logEvent]);

  // Leaving the tab or window is a useful behavioural signal.
  useEffect(() => {
    if (phase !== "question" || !question) return;
    const onVisibility = () =>
      logEvent(document.hidden ? "focus_lost" : "focus_returned", question.index, { via: "visibility" });
    const onBlur = () => logEvent("focus_lost", question.index, { via: "window" });
    const onFocus = () => logEvent("focus_returned", question.index, { via: "window" });
    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("blur", onBlur);
    window.addEventListener("focus", onFocus);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("focus", onFocus);
    };
  }, [phase, question, logEvent]);

  async function join(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const res = await api<{ participant_code: string; token: string; mode: string; n_questions: number; resumed: boolean }>(
        "/api/learner/join",
        { code: codeInput },
      );
      const a = { code: res.participant_code, token: res.token, mode: res.mode };
      save(STORAGE_KEY, JSON.stringify(a));
      setAuth(a);
      setNQuestions(res.n_questions);
      // Record which device and browser this session runs on (compatibility evidence).
      api("/api/learner/event", {
        ...a, type: "device_info", client_ms: serverNow(),
        payload: {
          user_agent: navigator.userAgent,
          screen: `${screen.width}x${screen.height}`,
          viewport: `${window.innerWidth}x${window.innerHeight}`,
          pixel_ratio: window.devicePixelRatio,
          touch: navigator.maxTouchPoints > 0,
          cpu_cores: navigator.hardwareConcurrency ?? null,
        },
      }).catch(() => {});
      if (res.resumed) await loadQuestion(a);
      else setPhase("welcome");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't connect. Tell the researcher.");
    } finally {
      setBusy(false);
    }
  }

  function onInput(e: React.ChangeEvent<HTMLInputElement>) {
    const value = e.target.value.replace(/[^0-9\- ]/g, "");
    const now = performance.now();
    if (firstInputMs.current === null && value.length > 0) {
      firstInputMs.current = Math.round(now - shownAt.current);
      logEvent("first_input", question?.index ?? null, { ms: firstInputMs.current });
    }
    if (value.length < answer.length) deletions.current += 1;
    else keystrokes.current += 1;
    lastActivity.current = now;
    setAnswer(value);
  }

  function choose(choice: string) {
    const now = performance.now();
    if (firstInputMs.current === null) {
      firstInputMs.current = Math.round(now - shownAt.current);
      logEvent("first_input", question?.index ?? null, { ms: firstInputMs.current });
    } else if (choice !== answer) {
      deletions.current += 1; // changed their mind
    }
    keystrokes.current += 1;
    lastActivity.current = now;
    setAnswer(choice);
  }

  async function askHint() {
    if (!auth || !question || hint) return;
    lastActivity.current = performance.now();
    try {
      const res = await api<{ hint: string }>("/api/learner/hint", { ...auth, question_index: question.index, client_ms: serverNow() });
      setHint(res.hint);
    } catch {
      setHint("Take another look at the question. You can do this!");
    }
    inputRef.current?.focus();
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!auth || !question || !answer.trim() || busy) return;
    setBusy(true);
    const responseTime = Math.round(performance.now() - shownAt.current);
    logEvent("input_changed", question.index, { keystrokes: keystrokes.current, deletions: deletions.current });
    try {
      const res = await api<{ correct: boolean; correct_answer: string; checkin_due: boolean; finished: boolean }>(
        "/api/learner/answer",
        {
          ...auth,
          question_index: question.index,
          answer,
          response_time_ms: responseTime,
          first_input_ms: firstInputMs.current,
          shown_client_ms: shownClientMs.current,
          answered_client_ms: serverNow(),
        },
      );
      setFeedback({ correct: res.correct, correctAnswer: res.correct_answer, checkinDue: res.checkin_due, finished: res.finished });
      setPhase("feedback");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't save that answer. Try again.");
    } finally {
      setBusy(false);
    }
  }

  async function next() {
    if (!auth || !feedback) return;
    if (feedback.checkinDue) setPhase("checkin");
    else await loadQuestion(auth);
  }

  async function checkin(choice: string) {
    if (!auth || !question) return;
    await api("/api/learner/checkin", { ...auth, after_question: question.index, choice, client_ms: serverNow() }).catch(() => {});
    await loadQuestion(auth);
  }

  const progress = question ? question.index + (phase === "question" ? 0 : 1) : 0;
  const praise = ENCOURAGEMENT[(question?.index ?? 0) % ENCOURAGEMENT.length];

  return (
    <main className="learner">
      {phase !== "join" && phase !== "done" && (
        <ol className="star-path" aria-label={`Puzzle ${Math.min(progress + 1, nQuestions)} of ${nQuestions}`}>
          {Array.from({ length: nQuestions }, (_, i) => (
            <li key={i} className={i < progress ? "star done" : i === progress ? "star now" : "star"} aria-hidden>
              {i < progress ? "★" : i + 1}
            </li>
          ))}
        </ol>
      )}

      {phase === "join" && (
        <form className="card center pop" onSubmit={join}>
          <Mascot mood="hello" size={120} />
          <h1 className="brand">Sens<span>AI</span></h1>
          <p className="lead">Hi! Type your code to start.</p>
          <input
            className="code-input"
            value={codeInput}
            onChange={(e) => setCodeInput(e.target.value.toUpperCase())}
            placeholder="P01"
            maxLength={6}
            autoFocus
            aria-label="Your code"
          />
          {error && <p className="error">{error}</p>}
          <button className="big" disabled={busy || codeInput.trim().length < 2}>Start</button>
        </form>
      )}

      {phase === "welcome" && (
        <div className="card center pop">
          <div className="speaker">
            <Mascot mood="hello" size={110} />
            <p className="bubble">I'm Sensei Ryu! Let's do some puzzles together.</p>
          </div>
          <p className="lead">
            {adaptive
              ? <>You'll do {nQuestions} puzzles: sums, number patterns, shapes and letters. I'll pick each one to suit how you're doing.</>
              : <>You'll see {nQuestions} puzzles: sums, number patterns, shapes and letters. They start easy and get harder.</>}
          </p>
          <ul className="rules">
            <li><span aria-hidden>🌱</span> Take your time. This is not a test and there's no score.</li>
            <li><span aria-hidden>💡</span> If you're not sure, press <b>Need a hint?</b></li>
            <li><span aria-hidden>🙂</span> Now and then I'll ask how it's going.</li>
          </ul>
          <button className="big" onClick={() => auth && loadQuestion(auth)}>Let's go!</button>
        </div>
      )}

      {phase === "question" && question && (
        <form className="card question pop" key={question.id} onSubmit={submit}>
          {support && (
            <div className="speaker small">
              <Mascot mood={support.hint ? "thinking" : "cheering"} size={64} />
              <p className="bubble">{support.message}</p>
            </div>
          )}
          <p className="instruction">{question.instruction}</p>
          <p className={question.kind === "word_problem" ? "prompt story" : "prompt"}>{question.prompt}</p>
          {question.choices.length > 0 ? (
            <div className="choices" role="radiogroup" aria-label="Choose an answer">
              {question.choices.map((c) => (
                <button
                  key={c}
                  type="button"
                  role="radio"
                  aria-checked={answer === c}
                  className={answer === c ? "choice on" : "choice"}
                  onClick={() => choose(c)}
                >
                  {c}
                </button>
              ))}
            </div>
          ) : (
            <input
              ref={inputRef}
              className="answer-input"
              value={answer}
              onChange={onInput}
              inputMode="numeric"
              autoComplete="off"
              placeholder="?"
              aria-label="Your answer"
            />
          )}
          {hint && <p className="hint"><span aria-hidden>💡</span> {hint}</p>}
          {error && <p className="error">{error}</p>}
          <div className="row">
            <button type="button" className="ghost" onClick={askHint} disabled={!!hint}>
              Need a hint?
            </button>
            <button className="big" disabled={busy || !answer.trim()}>Check</button>
          </div>
        </form>
      )}

      {phase === "feedback" && feedback && (
        <div className={`card center feedback pop ${feedback.correct ? "yes" : "no"}`}>
          <Mascot mood={feedback.correct ? "cheering" : "gentle"} size={110} />
          {feedback.correct ? (
            <h2>{praise}</h2>
          ) : (
            <>
              <h2>Not quite</h2>
              <p className="lead">The answer was <b className="answer-reveal">{feedback.correctAnswer}</b>. You'll get the next one!</p>
            </>
          )}
          <button className="big" onClick={next} autoFocus>
            {feedback.finished && !feedback.checkinDue ? "Finish" : "Next"}
          </button>
        </div>
      )}

      {phase === "checkin" && (
        <div className="card center pop">
          <div className="speaker">
            <Mascot mood="thinking" size={90} />
            <p className="bubble">How is it going?</p>
          </div>
          <p className="muted">Pick the one that fits best. There's no wrong answer.</p>
          <div className="faces">
            {CHECKIN_CHOICES.map((c) => (
              <button key={c.choice} className="face" onClick={() => checkin(c.choice)}>
                <span aria-hidden>{c.face}</span>
                {c.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {phase === "done" && (
        <div className="card center pop done-card">
          <div className="confetti" aria-hidden>
            {Array.from({ length: 18 }, (_, i) => <i key={i} style={{ ["--i" as string]: i }} />)}
          </div>
          <Mascot mood="cheering" size={140} />
          {adaptive || summary ? (
            <>
              <h1>Great job!</h1>
              <p className="lead">You finished all {summary?.total ?? nQuestions} puzzles.</p>
              {summary && (
                <div className="summary">
                  <div><b>{summary.correct}</b><span>right answers</span></div>
                  <div><b>{"★".repeat(summary.highest_level)}</b><span>hardest level reached</span></div>
                </div>
              )}
              <p className="muted">Sensei Ryu will see you next time!</p>
            </>
          ) : (
            <>
              <h1>All done!</h1>
              <p className="lead">Thank you for helping. You can tell the researcher you've finished.</p>
            </>
          )}
        </div>
      )}
    </main>
  );
}

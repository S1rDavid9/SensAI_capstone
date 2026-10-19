import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import "./index.css";
import Learner from "./screens/Learner";
import Observer from "./screens/Observer";
import Researcher from "./screens/Researcher";

function Home() {
  return (
    <main className="home">
      <h1 className="brand">Sens<span>AI</span> pilot</h1>
      <p className="muted">Choose who is using this device.</p>
      <div className="home-grid">
        <Link className="home-card" to="/learn">
          <strong>Learner</strong>
          <span>The laptop a learner uses for the questions</span>
        </Link>
        <Link className="home-card" to="/observe">
          <strong>Observer</strong>
          <span>For the researcher or teacher, on a phone</span>
        </Link>
        <Link className="home-card" to="/researcher">
          <strong>Researcher</strong>
          <span>Sessions, consent, live progress and export</span>
        </Link>
      </div>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/learn" element={<Learner />} />
        <Route path="/observe" element={<Observer />} />
        <Route path="/researcher" element={<Researcher />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);

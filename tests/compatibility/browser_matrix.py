"""Compatibility test: a full learner session in real browser engines and
device sizes.

For each browser x device it starts a fresh adaptive session, opens the
learner screen, joins, answers all 15 questions (alternating right and
wrong, so the app adapts both ways), answers the check-ins and checks the
finish screen. It records load time, total time, JavaScript errors,
horizontal overflow, and the smallest tap target, and saves screenshots.
The observer screen is checked on the phone sizes.

    uv run python -m tests.compatibility.browser_matrix

Results: docs/testing/compatibility_results.md and .json, screenshots in
docs/testing/compatibility/.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from playwright.sync_api import Page, sync_playwright

from pilot.core.tasks import QuestionBank
from tests.performance.load_test import KEY, ServerUnderTest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "testing"
SHOTS = OUT / "compatibility"
BANK = QuestionBank()


@dataclass(frozen=True)
class Target:
    browser: str          # chromium (system Chrome) | webkit (Safari engine) | firefox
    device: str           # Playwright device name, or a custom size
    label: str
    width: int = 0
    height: int = 0


TARGETS = (
    Target("chromium", "Pixel 7", "Chrome · Android phone (Pixel 7)"),
    Target("chromium", "", "Chrome · Chromebook (1366×768)", 1366, 768),
    Target("chromium", "", "Chrome · desktop (1920×1080)", 1920, 1080),
    Target("webkit", "iPhone 13", "Safari engine · iPhone 13"),
    Target("webkit", "iPad (gen 7)", "Safari engine · iPad"),
    Target("webkit", "", "Safari engine · Mac laptop (1440×900)", 1440, 900),
    Target("firefox", "", "Firefox · laptop (1366×768)", 1366, 768),
)


def new_session(url: str) -> str:
    r = httpx.post(url + "/api/researcher/sessions", headers=KEY, json={
        "guardian_consent": True, "learner_assent": True, "recorded_by": "compat", "mode": "adaptive"})
    return r.json()["participant_code"]


def current_question_id(url: str, page: Page) -> str:
    auth = json.loads(page.evaluate("localStorage.getItem('sensai.learner')"))
    return httpx.post(url + "/api/learner/question", json=auth).json()["question"]["id"]


def layout_checks(page: Page) -> dict:
    return page.evaluate("""() => {
        const overflow = document.documentElement.scrollWidth - window.innerWidth;
        const sizes = [...document.querySelectorAll('button, input')]
            .filter(e => e.offsetParent !== null)
            .map(e => { const r = e.getBoundingClientRect(); return Math.min(r.width, r.height); });
        return { horizontal_overflow_px: Math.max(0, overflow), smallest_tap_target_px: Math.round(Math.min(...sizes)) };
    }""")


def run_target(p, url: str, t: Target) -> dict:
    launcher = getattr(p, t.browser)
    browser = launcher.launch(channel="chrome") if t.browser == "chromium" else launcher.launch()
    ctx_args = dict(p.devices[t.device]) if t.device else {"viewport": {"width": t.width, "height": t.height}}
    if t.browser == "firefox":
        ctx_args.pop("is_mobile", None)
    context = browser.new_context(**ctx_args)
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    slug = t.label.split("·")[0].strip().lower().replace(" ", "-") + "-" + (t.device or f"{t.width}x{t.height}").lower().replace(" ", "-").replace("(", "").replace(")", "")
    result = {"target": t.label, "browser_version": browser.version, "passed": False}
    try:
        code = new_session(url)
        start = time.perf_counter()
        page.goto(url + "/learn")
        page.wait_for_selector(".code-input")
        result["load_ms"] = round(page.evaluate(
            "performance.getEntriesByType('navigation')[0].loadEventEnd") or (time.perf_counter() - start) * 1000)
        page.fill(".code-input", code)
        page.click("button.big")
        page.wait_for_selector("text=Let's go!")
        page.screenshot(path=SHOTS / f"{slug}-welcome.jpg")
        page.click("text=Let's go!")
        layout = None
        for i in range(15):
            page.wait_for_selector("form.question")
            if i == 2:
                layout = layout_checks(page)
                page.wait_for_timeout(400)
                page.screenshot(path=SHOTS / f"{slug}-question.jpg")
            q = BANK.get(current_question_id(url, page))
            answer = q.answer if i % 2 == 0 else ("0" if not q.choices else next(c for c in q.choices if c != q.answer))
            if q.choices:
                page.click(f".choice >> text={answer}")
            else:
                page.fill(".answer-input", answer)
            page.click("button.big:has-text('Check')")
            page.wait_for_selector(".feedback")
            page.click(".feedback button.big")
            if i in (4, 9, 14):
                page.wait_for_selector(".faces")
                page.click(".face >> nth=1")
        page.wait_for_selector("text=All done!", timeout=10000)
        page.wait_for_timeout(500)
        page.screenshot(path=SHOTS / f"{slug}-done.jpg")
        result.update(layout or {})
        result["session_seconds"] = round(time.perf_counter() - start, 1)
        result["passed"] = True
    except Exception as e:   # record the failure and keep testing the other targets
        result["failure"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
        page.screenshot(path=SHOTS / f"{slug}-failure.jpg")
    result["js_errors"] = errors[:5]

    if t.device in ("Pixel 7", "iPhone 13"):
        obs = context.new_page()
        obs.goto(url + "/observe")
        obs.select_option("select", "teacher")
        obs.fill("input[type=password]", "sensai-observe-compat")
        obs.click("button.big")
        obs.wait_for_selector("text=Who are you watching?")
        result["observer_screen"] = "loads and accepts the observer key"
        obs.screenshot(path=SHOTS / f"{slug}-observer.jpg")
    browser.close()
    return result


def to_markdown(results: list[dict]) -> str:
    lines = ["# Compatibility results", "",
             "Generated by `tests/compatibility/browser_matrix.py`: one full adaptive learner session per row.",
             "", "| Browser · device | Engine version | Passed | Load ms | Full session s | JS errors | Horizontal overflow px | Smallest tap target px |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['target']} | {r['browser_version']} | {'✓' if r['passed'] else '✗ ' + r.get('failure', '')} | "
                     f"{r.get('load_ms', '-')} | {r.get('session_seconds', '-')} | {len(r['js_errors'])} | "
                     f"{r.get('horizontal_overflow_px', '-')} | {r.get('smallest_tap_target_px', '-')} |")
    return "\n".join(lines) + "\n"


def shrink_screenshots() -> None:
    """Screenshots are taken at device resolution (up to 3x); store them as
    900 px JPEGs so the repository stays small."""
    import subprocess
    for png in SHOTS.glob("*.png"):
        subprocess.run(["sips", "-Z", "900", "-s", "format", "jpeg", "-s", "formatOptions", "80", str(png),
                        "--out", str(png.with_suffix(".jpg"))], check=True, capture_output=True)
        png.unlink()


def main() -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        import os
        os.environ["PILOT_OBSERVER_KEY"] = "sensai-observe-compat"
        server = ServerUnderTest(Path(tmp) / "compat.sqlite3")
        try:
            with sync_playwright() as p:
                results = []
                for t in TARGETS:
                    r = run_target(p, server.url, t)
                    print(f"{t.label:42s} {'PASS' if r['passed'] else 'FAIL'}  load {r.get('load_ms', '-')} ms  "
                          f"session {r.get('session_seconds', '-')} s  overflow {r.get('horizontal_overflow_px', '-')}  "
                          f"min tap {r.get('smallest_tap_target_px', '-')}  errors {len(r['js_errors'])} {r.get('failure', '')}")
                    results.append(r)
        finally:
            server.shutdown()
    shrink_screenshots()
    (OUT / "compatibility_results.json").write_text(json.dumps(results, indent=2))
    (OUT / "compatibility_results.md").write_text(to_markdown(results))


if __name__ == "__main__":
    main()

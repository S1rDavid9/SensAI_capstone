# SensAI pilot

The tool for collecting real data in the pilot. It has three screens and a small server:

| Screen | Device | Used for |
|---|---|---|
| `/learn` | Learner's laptop | 15 fixed questions, a hint button, and a "How is it going?" check-in after questions 5, 10 and 15 |
| `/observe` | Researcher's and teacher's phones | Momentary time sampling: every 20 seconds the app names a learner and the observer taps what they see |
| `/researcher` | Researcher's laptop | Create sessions after consent, watch progress live, close sessions, export CSVs |

## The questions

Every learner gets the same 15 questions in the same order (`pilot/tasks.py`), three at each difficulty level from 1 to 5. They stay within the proposal's two domains but mix seven formats:

- **Arithmetic:** plain sums, missing numbers (☐ + 9 = 23), and short word problems set in Kigali (mangoes, a bus, saving francs, a class in groups).
- **Patterns:** number sequences, shape sequences (choose the next shape), letter sequences, and odd one out.

## What gets recorded

- **Answers:** what was typed, correct or not, time from question shown to answer, time to the first keystroke, hints used.
- **Events:** question shown, first keystroke, keystroke and deletion counts, every 20 seconds without input, leaving and returning to the window.
- **Check-ins:** the learner's own answer to "How is it going?" (Too easy / Going well / It's hard / I'm lost).
- **Observations:** each observer's code for a learner at a moment (On task, Rushing, Struggling, Stuck, Off task, Can't see).
- **Shadow decisions:** after each answer, the Stage 2 DQN is asked what it would do. The decision is stored and never shown to the learner.

All timestamps are on the server's clock (each device measures its offset when it opens), so observations from the phones can be matched to events on the laptops.

## Privacy

No names or other identifiers are entered anywhere. A session can only be created once the researcher confirms signed guardian consent and the learner's assent; the learner gets a code (P01, P02, …), and the code-to-name list stays on paper with the researcher. The database lives in `pilot/data/`, which git ignores.

## Shadow mode and provisional states

The Stage 1 classifier is trained on this pilot's data, so it can't exist yet. Until it does, each answer gets a **provisional** behavioural state from simple rules in `pilot/labeler.py` (on task, rushing, struggling, stuck). These rules are placeholders, not findings, and every decision is stored with `state_source = provisional_rules_v1`. The DQN's other five inputs come straight from the telemetry.

| Behavioural state | DQN state slot (simulator name) |
|---|---|
| On task | ENGAGED |
| Rushing | BORED |
| Struggling | FRUSTRATED |
| Stuck | CONFUSED |

The DQN used is the one trained on simulator v2 (`logs/v2/dqn/best_run.json`); set `SENSAI_POLICY=v1` for the Phase 1 model.

### Guardrails

A few actions are ruled out where no teacher would pick them (`pilot/policy.py`):

| Learner state | Ruled out |
|---|---|
| On task | Making it easier |
| Stuck or struggling | Moving on, making it harder, adding a challenge |
| At the easiest / hardest level | Making it easier / harder (it would do nothing) |

When the DQN's first choice is ruled out, its next-best allowed action is used. Both are stored (`raw_action_name` and `action_name`, with the reason in `guardrail`), so results can be reported for the DQN alone and with guardrails.

## Running it

```bash
cd pilot/web && npm install && npm run build && cd ../..
PILOT_RESEARCHER_KEY=choose-a-key uv run uvicorn pilot.server:app --host 0.0.0.0 --port 8001
```

Open http://localhost:8001. Other devices on the same Wi-Fi use the laptop's IP address instead of `localhost`. Without `PILOT_RESEARCHER_KEY` the key is `sensai-dev`, which is for local testing only.

For frontend work, run `npm run dev` in `pilot/web` (port 5180, forwards `/api` to the server).

Tests: `uv run pytest tests/test_pilot.py`.

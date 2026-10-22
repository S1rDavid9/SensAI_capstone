"""Turning a scaffolding action into what the learner sees next.

`CurriculumPlanner` maps each of the eight actions to a target difficulty
and a rule for the question type, then picks the closest unseen question
from the bank. It also returns the support shown to the learner (a message,
or a hint shown straight away). Learner-facing text never names the action.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from pilot.core.tasks import MAX_LEVEL, MIN_LEVEL, Question, QuestionBank

SAME, DIFFERENT, ANY = "same", "different", "any"


@dataclass(frozen=True)
class ActionEffect:
    level_change: int
    kind_rule: str           # SAME / DIFFERENT / ANY question type as the last one
    message: str
    show_hint: bool = False


EFFECTS: dict[str, ActionEffect] = {
    "increase_difficulty": ActionEffect(+1, ANY, "Nice work! This one is a little harder."),
    "decrease_difficulty": ActionEffect(-1, ANY, "Let's try one that's a bit easier."),
    "change_task_type": ActionEffect(0, DIFFERENT, "Let's try a different kind of puzzle."),
    "offer_hint": ActionEffect(0, ANY, "Here's a tip to help with this one.", show_hint=True),
    "add_encouragement": ActionEffect(0, ANY, "You're working hard. Keep going, you can do this!"),
    "introduce_challenge": ActionEffect(+1, DIFFERENT, "Challenge time! Can you crack this one?"),
    "repeat_simplified": ActionEffect(-1, SAME, "Let's try a simpler one like the last."),
    "advance_topic": ActionEffect(+1, DIFFERENT, "Great, let's move on to something new."),
}


@dataclass(frozen=True)
class NextStep:
    question: Question
    message: str
    hint: str | None          # shown straight away when the action was offer_hint


class CurriculumPlanner:
    def __init__(self, bank: QuestionBank) -> None:
        self.bank = bank

    def first_question(self) -> Question:
        return self.bank.pilot_form()[0]

    def next_step(self, action: str, last: Question, seen: set[str], seed: int) -> NextStep:
        effect = EFFECTS[action]
        target = min(MAX_LEVEL, max(MIN_LEVEL, last.difficulty + effect.level_change))
        question = self._pick(target, effect.kind_rule, last, seen, random.Random(seed))
        return NextStep(question, self._message(effect, last, question), question.hint if effect.show_hint else None)

    @staticmethod
    def _message(effect: ActionEffect, last: Question, question: Question) -> str:
        """The action's message, unless the bank couldn't move the difficulty
        the way the action intended (e.g. "easier" at level 1)."""
        moved = question.difficulty - last.difficulty
        if effect.level_change < 0 and moved >= 0:
            return "Let's try another one. Take your time."
        if effect.level_change > 0 and moved <= 0:
            return "Nice work! Here's another one."
        return effect.message

    def _pick(self, target: int, kind_rule: str, last: Question, seen: set[str], rng: random.Random) -> Question:
        """Closest level first (target, then ±1, ±2 …); within a level, prefer
        the requested type; never repeat a question while unseen ones remain."""
        for distance in range(MAX_LEVEL):
            levels = [target] if distance == 0 else [target - distance, target + distance]
            for level in levels:
                if not MIN_LEVEL <= level <= MAX_LEVEL:
                    continue
                pool = [q for q in self.bank.at_level(level) if q.id not in seen]
                if not pool:
                    continue
                preferred = [q for q in pool if self._kind_ok(q, kind_rule, last)]
                return rng.choice(preferred or pool)
        return rng.choice([q for q in self.bank.questions if q.id != last.id])

    @staticmethod
    def _kind_ok(q: Question, kind_rule: str, last: Question) -> bool:
        if kind_rule == SAME:
            return q.kind == last.kind
        if kind_rule == DIFFERENT:
            return q.kind != last.kind
        return True

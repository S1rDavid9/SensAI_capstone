"""The fixed task form every pilot learner sees.

Every learner gets the same 15 questions in the same order. The DQN runs in
shadow mode during the pilot, so nothing adapts, and a fixed form keeps
telemetry comparable across learners. The form stays inside the two task
domains in the proposal (mental arithmetic and pattern recognition) but
mixes formats: plain sums, missing numbers, short word problems, number,
shape and letter patterns, and odd-one-out. Difficulty rises from 1 to 5,
three questions per level, matching the simulator's difficulty scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

INSTRUCTIONS = {
    "arithmetic": "Work it out and type the answer.",
    "missing_number": "Which number goes in the box?",
    "word_problem": "Read the story and type the answer.",
    "number_pattern": "What number comes next?",
    "shape_pattern": "Which shape comes next?",
    "letter_pattern": "Which letter comes next?",
    "odd_one_out": "Which one does not belong?",
}

DOMAIN = {
    "arithmetic": "arithmetic",
    "missing_number": "arithmetic",
    "word_problem": "arithmetic",
    "number_pattern": "pattern",
    "shape_pattern": "pattern",
    "letter_pattern": "pattern",
    "odd_one_out": "pattern",
}


@dataclass(frozen=True)
class Question:
    index: int
    kind: str
    difficulty: int
    prompt: str
    answer: str
    hint: str
    choices: tuple[str, ...] = field(default=())

    @property
    def domain(self) -> str:
        return DOMAIN[self.kind]

    def is_correct(self, given: str) -> bool:
        if self.choices:
            return given.strip() == self.answer
        cleaned = given.strip().replace(" ", "").replace(",", "")
        return cleaned == self.answer.replace(",", "")

    def public(self) -> dict:
        """What the learner's browser receives: everything except the answer."""
        return {
            "index": self.index,
            "kind": self.kind,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "instruction": INSTRUCTIONS[self.kind],
            "choices": list(self.choices),
        }


_ITEMS: list[tuple[str, int, str, str, str, tuple[str, ...]]] = [
    # (kind, difficulty, prompt, answer, hint, choices)
    ("arithmetic", 1, "8 + 7", "15", "Make 10 first: 8 + 2 = 10, then add the 5 that are left.", ()),
    ("shape_pattern", 1, "🔴 🔵 🔴 🔵 🔴 ?", "🔵", "The colours take turns.", ("🔴", "🔵", "🟢", "🟡")),
    ("word_problem", 1, "Aline has 6 mangoes. She buys 5 more. How many mangoes does she have now?", "11",
     "Start at 6 and count on 5.", ()),

    ("number_pattern", 2, "3, 7, 11, 15, ?", "19", "Look at how much it goes up each time.", ()),
    ("missing_number", 2, "☐ + 9 = 23", "14", "What do you add to 9 to make 23? Try 23 − 9.", ()),
    ("letter_pattern", 2, "A, C, E, G, ?", "I", "Skip one letter each time.", ("H", "I", "J", "K")),

    ("word_problem", 3, "A bus has 4 rows with 6 seats in each row. How many seats are there?", "24",
     "That's 4 groups of 6.", ()),
    ("shape_pattern", 3, "▲ ■ ■ ▲ ■ ■ ▲ ?", "■", "Find the group of shapes that repeats.", ("▲", "■", "●", "◆")),
    ("arithmetic", 3, "47 + 38", "85", "7 + 8 is 15, so carry 1 ten over to the tens.", ()),

    ("odd_one_out", 4, "12   18   25   30", "25", "Are the numbers even or odd?", ("12", "18", "25", "30")),
    ("number_pattern", 4, "1, 3, 6, 10, 15, ?", "21", "The jumps get bigger: +2, +3, +4, +5 …", ()),
    ("word_problem", 4, "Keza saved 1,500 francs. She spent 650 francs on a book. How many francs does she have left?",
     "850", "Take away 600 first, then take away 50 more.", ()),

    ("missing_number", 5, "☐ × 7 + 4 = 60", "8", "Take away the 4 first. Then which number times 7 makes what's left?", ()),
    ("number_pattern", 5, "1, 1, 2, 3, 5, 8, ?", "13", "Add the two numbers before to get the next one.", ()),
    ("word_problem", 5, "A class of 28 learners sits in groups of 4. Each group gets 3 pencils. How many pencils are needed?",
     "21", "First find how many groups there are.", ()),
]

FORM: list[Question] = [Question(i, *item) for i, item in enumerate(_ITEMS)]
N_QUESTIONS = len(FORM)

# Self-report check-ins come after these question indices (0-based).
CHECKIN_AFTER = {4, 9, 14}

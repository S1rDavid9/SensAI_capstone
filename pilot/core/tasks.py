"""Questions: the bank, the fixed pilot form, and answer checking.

The bank covers difficulty levels 1-5 (the simulator's scale) across seven
formats in the proposal's two domains, mental arithmetic and pattern
recognition. The pilot used a fixed 15-question form drawn from the bank, so
every learner saw the same items; adaptive mode picks from the whole bank.
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
    "arithmetic": "arithmetic", "missing_number": "arithmetic", "word_problem": "arithmetic",
    "number_pattern": "pattern", "shape_pattern": "pattern", "letter_pattern": "pattern", "odd_one_out": "pattern",
}

MIN_LEVEL, MAX_LEVEL = 1, 5


@dataclass(frozen=True)
class Question:
    id: str
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

    def public(self, index: int) -> dict:
        """What the learner's device receives: never the answer."""
        return {
            "index": index,
            "id": self.id,
            "kind": self.kind,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "instruction": INSTRUCTIONS[self.kind],
            "choices": list(self.choices),
        }


def _q(id_, kind, level, prompt, answer, hint, choices=()) -> Question:
    return Question(id_, kind, level, prompt, answer, hint, tuple(choices))


BANK: tuple[Question, ...] = (
    # ---- Level 1 ----
    _q("1-sum-a", "arithmetic", 1, "8 + 7", "15", "Make 10 first: 8 + 2 = 10, then add the 5 that are left."),
    _q("1-shape-a", "shape_pattern", 1, "🔴 🔵 🔴 🔵 🔴 ?", "🔵", "The colours take turns.", ("🔴", "🔵", "🟢", "🟡")),
    _q("1-word-a", "word_problem", 1, "Aline has 6 mangoes. She buys 5 more. How many mangoes does she have now?", "11",
       "Start at 6 and count on 5."),
    _q("1-sum-b", "arithmetic", 1, "9 − 4", "5", "Count back 4 from 9."),
    _q("1-missing-a", "missing_number", 1, "☐ + 3 = 10", "7", "What do you add to 3 to make 10?"),
    _q("1-numpat-a", "number_pattern", 1, "1, 2, 3, 4, ?", "5", "Count up by one."),
    _q("1-letter-a", "letter_pattern", 1, "A, B, C, D, ?", "E", "It's the next letter of the alphabet.", ("D", "E", "F", "G")),
    _q("1-odd-a", "odd_one_out", 1, "2   4   6   7", "7", "Which one is not even?", ("2", "4", "6", "7")),
    # ---- Level 2 ----
    _q("2-numpat-a", "number_pattern", 2, "3, 7, 11, 15, ?", "19", "Look at how much it goes up each time."),
    _q("2-missing-a", "missing_number", 2, "☐ + 9 = 23", "14", "What do you add to 9 to make 23? Try 23 − 9."),
    _q("2-letter-a", "letter_pattern", 2, "A, C, E, G, ?", "I", "Skip one letter each time.", ("H", "I", "J", "K")),
    _q("2-sum-a", "arithmetic", 2, "23 + 14", "37", "Add the tens (20 + 10), then the ones (3 + 4)."),
    _q("2-sum-b", "arithmetic", 2, "58 − 25", "33", "Take away the tens first (58 − 20), then the ones."),
    _q("2-word-a", "word_problem", 2, "There are 12 eggs in a tray. 5 eggs break. How many eggs are left?", "7",
       "12 take away 5."),
    _q("2-shape-a", "shape_pattern", 2, "🟡 🟡 🔺 🟡 🟡 🔺 🟡 ?", "🟡", "Two circles, then a triangle.", ("🟡", "🔺", "🔵", "🟢")),
    _q("2-odd-a", "odd_one_out", 2, "10   20   25   30", "25", "Count in tens.", ("10", "20", "25", "30")),
    # ---- Level 3 ----
    _q("3-word-a", "word_problem", 3, "A bus has 4 rows with 6 seats in each row. How many seats are there?", "24",
       "That's 4 groups of 6."),
    _q("3-shape-a", "shape_pattern", 3, "▲ ■ ■ ▲ ■ ■ ▲ ?", "■", "Find the group of shapes that repeats.", ("▲", "■", "●", "◆")),
    _q("3-sum-a", "arithmetic", 3, "47 + 38", "85", "7 + 8 is 15, so carry 1 ten over to the tens."),
    _q("3-numpat-a", "number_pattern", 3, "3, 6, 12, 24, ?", "48", "Each number is double the one before."),
    _q("3-numpat-b", "number_pattern", 3, "50, 47, 44, 41, ?", "38", "The numbers go down by the same amount."),
    _q("3-missing-a", "missing_number", 3, "☐ × 4 = 28", "7", "Which number times 4 makes 28?"),
    _q("3-letter-a", "letter_pattern", 3, "Z, Y, X, W, ?", "V", "Go backwards through the alphabet.", ("U", "V", "W", "X")),
    _q("3-odd-a", "odd_one_out", 3, "15   25   35   42", "42", "Look at the last digit.", ("15", "25", "35", "42")),
    # ---- Level 4 ----
    _q("4-odd-a", "odd_one_out", 4, "12   18   25   30", "25", "Are the numbers even or odd?", ("12", "18", "25", "30")),
    _q("4-numpat-a", "number_pattern", 4, "1, 3, 6, 10, 15, ?", "21", "The jumps get bigger: +2, +3, +4, +5 …"),
    _q("4-word-a", "word_problem", 4,
       "Keza saved 1,500 francs. She spent 650 francs on a book. How many francs does she have left?", "850",
       "Take away 600 first, then take away 50 more."),
    _q("4-sum-a", "arithmetic", 4, "7 × 8", "56", "7 × 8 is the same as 7 × 4, doubled."),
    _q("4-sum-b", "arithmetic", 4, "92 − 47", "45", "Count up from 47 to 50, then from 50 to 92."),
    _q("4-missing-a", "missing_number", 4, "☐ − 18 = 27", "45", "Add 18 to 27."),
    _q("4-letter-a", "letter_pattern", 4, "B, D, G, K, ?", "P", "The gaps grow: +2, +3, +4 …", ("N", "O", "P", "Q")),
    _q("4-shape-a", "shape_pattern", 4, "▲ ● ▲ ● ● ▲ ● ● ● ?", "▲", "The number of circles grows each time.",
       ("▲", "●", "■", "◆")),
    # ---- Level 5 ----
    _q("5-missing-a", "missing_number", 5, "☐ × 7 + 4 = 60", "8",
       "Take away the 4 first. Then which number times 7 makes what's left?"),
    _q("5-numpat-a", "number_pattern", 5, "1, 1, 2, 3, 5, 8, ?", "13", "Add the two numbers before to get the next one."),
    _q("5-word-a", "word_problem", 5,
       "A class of 28 learners sits in groups of 4. Each group gets 3 pencils. How many pencils are needed?", "21",
       "First find how many groups there are."),
    _q("5-sum-a", "arithmetic", 5, "144 ÷ 12", "12", "Which number times 12 makes 144?"),
    _q("5-sum-b", "arithmetic", 5, "6 × 7 + 9", "51", "Do the multiplication first, then add 9."),
    _q("5-numpat-b", "number_pattern", 5, "1, 4, 9, 16, 25, ?", "36", "These are 1×1, 2×2, 3×3 …"),
    _q("5-word-b", "word_problem", 5,
       "Pens cost 250 francs each. Musa buys 4 pens and pays with 2,000 francs. How much change does he get?", "1000",
       "First find the cost of 4 pens."),
    _q("5-letter-a", "letter_pattern", 5, "A, C, F, J, ?", "O", "The gaps grow: +2, +3, +4 …", ("M", "N", "O", "P")),
)

# The fixed form used in the pilot (21 October 2026). Order must not change:
# the collected data refers to these positions.
PILOT_FORM_IDS = (
    "1-sum-a", "1-shape-a", "1-word-a",
    "2-numpat-a", "2-missing-a", "2-letter-a",
    "3-word-a", "3-shape-a", "3-sum-a",
    "4-odd-a", "4-numpat-a", "4-word-a",
    "5-missing-a", "5-numpat-a", "5-word-a",
)
SESSION_LENGTH = 15
CHECKIN_AFTER = frozenset({4, 9, 14})   # 0-based positions


class QuestionBank:
    """Looks questions up by id, level and kind."""

    def __init__(self, questions: tuple[Question, ...] = BANK) -> None:
        self._by_id = {q.id: q for q in questions}
        if len(self._by_id) != len(questions):
            raise ValueError("Question ids must be unique.")
        self.questions = questions

    def get(self, question_id: str) -> Question:
        return self._by_id[question_id]

    def at_level(self, level: int) -> list[Question]:
        return [q for q in self.questions if q.difficulty == level]

    def pilot_form(self) -> list[Question]:
        return [self.get(i) for i in PILOT_FORM_IDS]

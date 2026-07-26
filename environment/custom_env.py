"""AdaptLearn-v1: a custom Gymnasium environment for adaptive-tutoring policy learning.

WHAT THIS ENVIRONMENT IS
-------------------------
An RL agent plays the role of an intelligent tutoring system. At every step it
chooses one pedagogical action (e.g. raise difficulty, offer a hint, give
encouragement). A *simulated* child learner reacts to that action and to how
well the current task difficulty matches the learner's (hidden) ability, and
transitions between four affective/engagement states:

    FRUSTRATED, BORED, CONFUSED, ENGAGED ("flow")

The agent is never told the correct action. It must discover, purely from
reward feedback across many episodes, which actions reliably nudge the
learner toward ENGAGED and keep them there. Nothing about the policy is
hand-coded.

WHY THE TRANSITION MODEL LOOKS THE WAY IT DOES
-------------------------------------------------
The state-transition model implemented below is a *transparent, parametric*
model (softmax over hand-specified logits), not a black box, so that every
transition probability can be traced back to a named, commented constant and
justified in a written report. The qualitative shape of the model is grounded
in three well-established bodies of theory, cited inline wherever they drive
a specific design choice:

  * Flow Theory (Csikszentmihalyi, 1990, *Flow: The Psychology of Optimal
    Experience*): flow ("ENGAGED") emerges specifically in the channel where
    perceived challenge and perceived skill are closely matched. Challenge
    that outstrips skill produces anxiety/frustration; skill that outstrips
    challenge produces boredom. This directly motivates using the signed gap
    `difficulty - ability` as the primary driver of FRUSTRATED vs. BORED vs.
    ENGAGED logits below.
  * Self-Determination Theory (Deci & Ryan, 2000): intrinsic motivation is
    supported by autonomy, competence, and relatedness. This motivates why
    "encouragement" (relatedness/competence support) reduces frustration risk
    and why "challenge/game element" (autonomy- and competence-affirming
    framing) boosts engagement independent of raw difficulty.
  * Growth Mindset research (Dweck, 2006): effort-oriented encouragement
    lowers the threat response to difficulty rather than difficulty itself
    changing. This is why the "encouragement" action perturbs the emotional
    logits directly rather than silently changing the (agent-invisible)
    ability or (agent-visible) difficulty terms.

The exact numeric constants (weights, thresholds, base rates) are *not*
empirical fits to real learner-interaction data -- no such dataset is used
here. They are illustrative, hand-calibrated placeholders chosen so the model
behaves in the theory-predicted *direction* (e.g. large positive gap reliably
raises frustration risk). This is stated plainly so it can be defended
honestly in an academic report: the contribution is the structure of the
model and the environment, not a claim of empirically validated constants.
Future work (e.g. a capstone extension) could fit these constants to logged
real tutoring-system data.

KEY DESIGN DECISIONS NOT FULLY SPECIFIED IN THE ORIGINAL BRIEF
------------------------------------------------------------------
Several details were left open in the assignment brief and required a
judgment call. Each is documented at its point of use below, and summarized
here for convenience:

1. Hidden "ability" variable. The brief's observation space has no ability
   feature, only behavioural proxies (response time, error rate, hint count).
   We treat ability as a latent variable of the simulated learner, sampled
   once per episode and held fixed, which the agent can never directly
   observe. This is a deliberate partial-observability design: it forces the
   agent to infer "is this learner struggling because of *me* or because of
   *them*" indirectly, from behaviour -- mirroring a real tutoring system,
   which never has ground-truth access to a student's ability either.
2. One-hot encoding of engagement state. The four states are nominal
   categories, not an ordinal scale, so a single normalized index (0, 1/3,
   2/3, 1) would imply false distances between e.g. BORED and CONFUSED. We
   use a 4-dimensional one-hot block instead.
3. Reward for CONFUSED is not specified in the brief (only ENGAGED,
   FRUSTRATED, BORED, and terminal dropout are). We assign a small negative
   constant (`CONFUSED_STEP_REWARD`), smaller in magnitude than FRUSTRATED
   or BORED, reasoning that confusion is a solvable task-clarity problem
   rather than a motivational collapse.
4. "Transitioning to" a negative state is interpreted as *being in* that
   state at the end of a step (i.e. it also applies on self-loops, not only
   on entry), symmetrically with how ENGAGED is explicitly split into
   "reaching" (+1) vs. "maintaining" (+0.3). Without this, an agent could
   pay the -1/-0.5 penalty once and then coast for free while the learner
   stays miserable, which is not a defensible tutoring policy.
5. "Fully disengaged" (for the 3-consecutive-steps dropout condition) is
   defined as FRUSTRATED or BORED, but *not* CONFUSED. Flow Theory frames
   boredom and anxiety/frustration as the two failure modes flanking the
   flow channel (a challenge-skill mismatch); confusion is a different axis
   (task clarity), so it is treated as an undesirable-but-not-dropout-risk
   state.
6. "All tasks in the session completed while engagement stayed non-negative"
   is operationalised as: the agent has advanced through a fixed number of
   topics (`n_topics`, default 8, configurable) via action 7, AND the
   learner was never in a FRUSTRATED or BORED state at any point in the
   episode. This is a deliberately strict bar, matching the pedagogical
   intent that an ideal tutor keeps the learner in flow *consistently*, not
   just on average.
7. A symmetric one-off `SESSION_COMPLETE_BONUS` (+2) is added on session
   success, mirroring the -2 dropout penalty, since the brief specifies the
   dropout penalty but not a completion bonus. This gives the terminal
   signal comparable magnitude in both directions.
8. Actions 3 (hint) and 6 (repeat simplified) are modelled as *scaffolding*:
   they change how hard the task *feels* for this transition only (an
   "effective gap" used solely inside the transition-probability
   computation), not the stored, agent-visible `difficulty` value. This
   matches how hints and simplified repeats function pedagogically -- the
   nominal difficulty of the curriculum item hasn't changed, but the
   learner's felt experience of it has.
9. "Hint request count" (an observation feature) is reconciled with the fact
   that hints are tutor-initiated (action 3), not literally learner-
   requested, by treating it as a cumulative count of hints *delivered* so
   far on the current task -- a proxy for how much scaffolding this task has
   needed, which is the quantity that actually matters pedagogically.
10. REVISION NOTE: the reward constants below (`RewardConfig`) were revised
    after policy inspection of a trained DQN agent surfaced a reward-hacking
    behavior in an earlier iteration -- the agent learned to reach ENGAGED
    once and then "camp" there indefinitely using low-risk filler actions,
    since accumulated per-step maintenance reward outweighed the one-time
    reward for genuinely advancing through the curriculum. `RewardConfig`'s
    docstring explains the corrected balance; this is a deliberate,
    diagnosed correction, not an arbitrary tuning pass.
11. SECOND REVISION NOTE (TRIED AND REVERTED): after the above fix, policy
    inspection across DQN/PPO/A2C showed all three converging on
    `decrease_difficulty` as their CONFUSED response instead of
    `offer_hint`/`repeat_simplified`. A targeted `confused_correct_tool_bonus`
    was added to `RewardConfig` to fix this, then reverted after a full
    sweep rerun showed it destabilized PPO/A2C/REINFORCE's broader policies
    while only partially fixing DQN -- see `RewardConfig`'s docstring; full
    analysis lives in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EngagementState(IntEnum):
    """The four affective/engagement states the simulated learner can occupy.

    Order matches the brief exactly (FRUSTRATED, BORED, CONFUSED, ENGAGED).
    These are nominal categories -- the integer values are only used as
    array indices, never treated as an ordinal quantity anywhere in the
    environment (see the one-hot observation encoding).
    """

    FRUSTRATED = 0
    BORED = 1
    CONFUSED = 2
    ENGAGED = 3


N_STATES = len(EngagementState)

# States that count toward the "3 consecutive fully disengaged steps" dropout
# condition. See design decision (5) above for why CONFUSED is excluded.
DISENGAGED_STATES = (EngagementState.FRUSTRATED, EngagementState.BORED)

ACTION_NAMES = (
    "increase_difficulty",
    "decrease_difficulty",
    "change_task_type",
    "offer_hint",
    "add_encouragement",
    "introduce_challenge",
    "repeat_simplified",
    "advance_topic",
)
N_ACTIONS = len(ACTION_NAMES)

# Index of each named action, for readability at call sites below.
ACTION_INCREASE_DIFFICULTY = 0
ACTION_DECREASE_DIFFICULTY = 1
ACTION_CHANGE_TASK_TYPE = 2
ACTION_OFFER_HINT = 3
ACTION_ADD_ENCOURAGEMENT = 4
ACTION_INTRODUCE_CHALLENGE = 5
ACTION_REPEAT_SIMPLIFIED = 6
ACTION_ADVANCE_TOPIC = 7

MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 5


@dataclass(frozen=True)
class TransitionConfig:
    """Tunable constants governing the engagement-state transition model.

    All transition probabilities are produced by a softmax over per-state
    "logits" (unnormalized log-scores). Each constant below contributes an
    additive term to one or more of those logits. Keeping every contribution
    as a single named, commented constant -- rather than folding magic
    numbers directly into the transition code -- is what makes this model
    "explicit" rather than "black-box": every probability the environment
    ever produces can be traced back to a term in this table, which is what
    a report/paper needs to cite and justify.

    gap := difficulty_level - ability, both on a 1..5 scale, so
    gap in [-4, +4]. gap > 0 means the task is harder than the learner's
    ability (frustration risk); gap < 0 means it is easier (boredom risk).
    """

    # --- Challenge-skill match -> ENGAGED (Flow Theory) ---
    # Gaussian bump centered at gap = 0; match_sigma controls how forgiving
    # the "well matched" zone is (larger = wider flow channel).
    match_engage_weight: float = 3.0
    match_sigma: float = 1.15

    # --- Excess challenge -> FRUSTRATED / excess ease -> BORED (Flow Theory) ---
    gap_frustrate_weight: float = 1.6
    gap_bored_weight: float = 1.4
    frustrate_gap_threshold: float = 1.0
    bored_gap_threshold: float = 1.0

    # --- Confusion: driven by error rate and abrupt format changes, not by
    # the difficulty gap (Flow Theory treats clarity of goals/feedback as an
    # independent channel condition, distinct from challenge-skill balance) ---
    base_confusion_logit: float = -1.0
    confusion_error_weight: float = 1.5
    confusion_format_change_bonus: float = 1.2
    confusion_clarity_relief: float = 1.5

    # --- Affective inertia: states have some stickiness step-to-step ---
    persistence_bonus: float = 0.6

    # --- Scaffolding actions (hint / simplify) reduce the *felt* gap only,
    # not the stored difficulty (design decision 8 above) ---
    hint_gap_relief: float = 1.5
    simplify_gap_relief: float = 2.2

    # --- Encouragement (SDT relatedness/competence support + Growth Mindset:
    # softens the threat response to difficulty without changing difficulty) ---
    encouragement_frustration_relief: float = 1.2
    encouragement_engage_bonus: float = 0.6

    # --- Gamification (SDT autonomy/competence framing) ---
    challenge_engage_bonus: float = 1.0
    challenge_bored_relief: float = 1.0

    # --- Novelty from changing task format can itself relieve boredom ---
    novelty_engage_bonus: float = 0.4

    # --- Behavioural-signal simulation (response time / error rate) ---
    base_error_rate: float = 0.15
    error_rate_gap_coef: float = 0.5
    base_response_time: float = 0.4
    response_time_gap_coef: float = 0.3
    gap_range: float = 4.0  # normalizes gap into roughly [0, 1] for the terms above
    error_rate_noise_std: float = 0.05
    response_time_noise_std: float = 0.05

    # Per-state additive offsets to error rate / response time, reflecting
    # how behaviour differs *within* a given engagement state independent of
    # the difficulty gap (e.g. a bored learner rushes and guesses; a
    # confused learner is both slow and error-prone).
    state_error_offset: tuple[float, float, float, float] = (0.15, 0.05, 0.20, -0.10)
    state_rt_offset: tuple[float, float, float, float] = (0.10, -0.15, 0.20, 0.00)

    softmax_temperature: float = 1.0


@dataclass(frozen=True)
class RewardConfig:
    """Reward constants. See module docstring, decisions (3), (4), (7), (10).

    `engaged_maintain_reward`, `session_complete_bonus`, `topic_advance_bonus`,
    and `step_cost` were rebalanced together after policy inspection of a
    trained DQN agent (see decision 10) showed the *original* constants
    (engaged_maintain_reward=0.3, session_complete_bonus=2.0, no per-topic
    bonus, no step cost) made passively camping in ENGAGED for a full
    50-step episode (~15.7 reward, best case) earn nearly 4x more than
    cleanly advancing through all `n_topics` (~4.4 reward, best case) --
    doing nothing productive was structurally the dominant strategy. The
    three changes below are coordinated, not independent tweaks:

      * `engaged_maintain_reward` is *halved*, not zeroed -- Flow Theory
        treats sustained flow as the desired end state, so it must stay
        rewarded; it just must stop being *competitive with* progression.
      * `topic_advance_bonus` makes genuine curriculum progression pay
        immediately (each successful advance), not only in one lump sum
        many steps later -- a dense signal is what TD-learning-based
        algorithms (DQN, A2C) can actually propagate credit from; a sparse
        8-steps-away bonus is a hard bootstrapping target. It is explicitly
        capped at the first `n_topics` awards per episode (see `step()`) so
        it cannot be farmed by spamming `advance_topic` past the curriculum
        length -- an earlier, uncapped version of this idea would have
        simply relocated the exploit rather than closing it.
      * `step_cost` is a small constant, state-and-action-independent
        "living penalty" (standard in RL environments, e.g. Gridworld/
        MountainCar) that makes idle time mildly costly without hard-coding
        any preference over *which* actions count as "idle" -- deliberately
        rejected: penalizing specific filler actions (e.g. change_task_type
        while already ENGAGED) directly, since that would hand-code the
        behavior the agent is supposed to discover, and would also wrongly
        penalize those same actions' legitimate preventive use.

    See `tests/test_reward_and_terminal.py::test_progression_beats_camping_in_expected_value`
    for a regression test that would have caught the original imbalance.

    SECOND REVISION, TRIED AND REVERTED: a targeted `confused_correct_tool_bonus`
    (rewarding offer_hint/repeat_simplified specifically while CONFUSED, plus a
    raised `confused_step_reward`) was tried after policy inspection showed
    DQN/PPO/A2C all converging on `decrease_difficulty` for CONFUSED. Rerunning
    the full sweep against it showed it destabilized the broader policy for
    PPO, A2C, and REINFORCE (action-repertoire collapse, higher dropout rates)
    while only partially improving DQN's CONFUSED handling. Reverted; full
    analysis lives in the report, not here. `confused_step_reward` is back at
    its original value below.
    """

    engaged_reach_reward: float = 1.0
    engaged_maintain_reward: float = 0.15  # halved from 0.30 -- see class docstring
    frustrated_step_reward: float = -1.0
    bored_step_reward: float = -0.5
    confused_step_reward: float = -0.2
    dropout_terminal_penalty: float = -2.0
    session_complete_bonus: float = 5.0  # raised from 2.0 -- see class docstring
    topic_advance_bonus: float = 0.5  # new; capped at n_topics awards/episode -- see step()
    step_cost: float = -0.02  # new; flat per-step "living penalty" -- see class docstring


class AdaptLearnEnv(gym.Env):
    """AdaptLearn-v1: adaptive-tutoring policy environment.

    Action space
    ------------
    Discrete(8) -- see `ACTION_NAMES` for the meaning of each index.

    Observation space
    ------------------
    Box(0, 1, shape=(9,), float32):
        [0:4]  one-hot current engagement state (see `EngagementState`)
        [4]    response time, normalized
        [5]    error rate, normalized
        [6]    hint count (this task), normalized
        [7]    current difficulty level, normalized to [0, 1] from [1, 5]
        [8]    time on current task, normalized

    Note the learner's true `ability` is *not* observable -- see design
    decision (1) in the module docstring.

    Reward
    ------
    See `RewardConfig` and `_compute_reward`.

    Episode termination
    --------------------
    * terminated=True on 3 consecutive fully-disengaged steps ("dropout"),
      or on completing all topics without ever being fully disengaged
      ("success"). Both are genuine terminal states of the MDP.
    * truncated=True at `max_episode_steps` (default 50), a time-limit cutoff
      unrelated to the MDP's own dynamics (standard Gymnasium convention).
    """

    # "human_manual" is a pacing-only variant of "human": it plays the exact
    # same deterministic action/state sequence but holds after each step's
    # action-cue+mood animation until a spacebar press, instead of a fixed
    # duration -- intended for recording, see rendering.py's render().
    metadata = {"render_modes": ["human", "human_manual", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        max_episode_steps: int = 50,
        n_topics: int = 8,
        render_mode: str | None = None,
        transition_config: TransitionConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")

        self.max_episode_steps = max_episode_steps
        self.n_topics = n_topics
        self.render_mode = render_mode
        self.tc = transition_config or TransitionConfig()
        self.rc = reward_config or RewardConfig()

        # Normalization caps for observation features that are otherwise
        # unbounded counters. Chosen to be "typical session" magnitudes
        # rather than theoretical maxima, so the normalized value actually
        # uses the [0, 1] range usefully instead of saturating at ~0 for
        # almost the entire episode. Flagged here because the brief does not
        # specify these caps.
        self._hint_count_norm_cap = 5.0
        self._time_on_task_norm_cap = 10.0

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)

        # Renderer is created lazily on the first human-mode render() call so
        # that headless use (training, unit tests) never imports PyOpenGL.
        self._renderer = None

        # Episode-local state, populated by reset().
        self.ability: float = 0.0
        self.difficulty: int = 1
        self.engagement_state: EngagementState = EngagementState.BORED
        self.error_rate: float = 0.0
        self.response_time: float = 0.0
        self.hint_count: int = 0
        self.time_on_task: int = 0
        self.consecutive_disengaged: int = 0
        self.topics_completed: int = 0
        self.session_had_negative: bool = False
        self.steps: int = 0
        self.last_action: int | None = None
        self.last_reward: float = 0.0

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode with a randomized learner and starting task.

        Both the learner's hidden ability and the initial engagement state
        and difficulty are randomized every episode (per the brief). This is
        essential, not cosmetic: if the environment always started from the
        same state, the agent could memorize a single fixed recovery script
        rather than learning a policy that generalizes across the range of
        learners and starting conditions a real tutoring system would face.
        """
        super().reset(seed=seed)  # sets self.np_random deterministically from `seed`

        self.ability = float(self.np_random.uniform(1.0, MAX_DIFFICULTY))
        self.difficulty = int(self.np_random.integers(MIN_DIFFICULTY, MAX_DIFFICULTY + 1))
        self.engagement_state = EngagementState(int(self.np_random.integers(0, N_STATES)))

        gap = self.difficulty - self.ability
        self.response_time, self.error_rate = self._simulate_behavioral_signals(
            gap, self.engagement_state
        )

        self.hint_count = 0
        self.time_on_task = 0
        self.consecutive_disengaged = 0
        self.topics_completed = 0
        self.session_had_negative = self.engagement_state in DISENGAGED_STATES
        self.steps = 0
        self.last_action = None
        self.last_reward = 0.0

        obs = self._build_observation()
        info = self._build_info()

        if self.render_mode in ("human", "human_manual"):
            self.render()

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the simulation by one pedagogical action.

        The step is organised in a fixed causal order so every quantity is
        computed from values that are already finalized for this step:

          1. Apply the action's effect on the *stored* difficulty (only
             actions 0, 1, 7 change it; see class docstring decision 8).
          2. Compute the resulting difficulty-ability gap.
          3. Build the engagement-state transition logits from the gap plus
             any action-specific perturbations (hints/simplify only affect
             an *effective* gap used here, not the stored difficulty).
          4. Sample the next engagement state from those logits.
          5. Simulate this step's behavioural signals (response time, error
             rate) from the gap and the *resulting* state, since these are
             observed as a consequence of how the learner just behaved.
          6. Compute reward from the (previous state -> next state)
             transition.
          7. Update bookkeeping counters and check terminal conditions.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected one of 0..{N_ACTIONS - 1}")

        prev_state = self.engagement_state

        # --- 1. Apply direct difficulty changes -----------------------------
        if action == ACTION_INCREASE_DIFFICULTY:
            self.difficulty = min(MAX_DIFFICULTY, self.difficulty + 1)
        elif action == ACTION_DECREASE_DIFFICULTY:
            self.difficulty = max(MIN_DIFFICULTY, self.difficulty - 1)
        elif action == ACTION_ADVANCE_TOPIC:
            # A new topic is nominally a bit harder than the one just left,
            # representing normal curriculum progression.
            self.difficulty = min(MAX_DIFFICULTY, self.difficulty + 1)

        # --- 2. Difficulty-ability gap for this step ------------------------
        gap = self.difficulty - self.ability

        # --- 3. Build transition logits --------------------------------------
        logits = self._transition_logits(gap=gap, prev_state=prev_state, action=action)

        # --- 4. Sample next engagement state ----------------------------------
        probs = _softmax(logits, temperature=self.tc.softmax_temperature)
        next_state = EngagementState(int(self.np_random.choice(N_STATES, p=probs)))

        # --- 5. Simulate behavioural signals for the resulting state -----------
        self.response_time, self.error_rate = self._simulate_behavioral_signals(gap, next_state)

        # --- 6. Reward ------------------------------------------------------
        reward = self._compute_reward(prev_state, next_state)
        # Flat "living penalty" applied every step regardless of state/action
        # -- see RewardConfig's docstring for why this exists and why it is
        # deliberately state/action-independent rather than targeting
        # specific "filler" actions.
        reward += self.rc.step_cost

        # --- 7. Bookkeeping and terminal conditions -----------------------------
        if action == ACTION_OFFER_HINT:
            self.hint_count += 1

        if action == ACTION_ADVANCE_TOPIC:
            self.topics_completed += 1
            self.time_on_task = 0
            self.hint_count = 0
            # Dense, per-topic progression reward -- deliberately capped at
            # the first `n_topics` awards per episode (rather than awarded
            # for every advance_topic call, uncapped) so it cannot be farmed
            # by spamming advance_topic past the curriculum length; see
            # RewardConfig's docstring for the exploit this closes. Also
            # withheld if the advance itself immediately harmed the learner
            # (landed in FRUSTRATED/BORED) -- progression credit is for
            # advances that didn't hurt the learner, not any advance at all.
            if self.topics_completed <= self.n_topics and next_state not in DISENGAGED_STATES:
                reward += self.rc.topic_advance_bonus
        else:
            self.time_on_task += 1

        if next_state in DISENGAGED_STATES:
            self.consecutive_disengaged += 1
            self.session_had_negative = True
        else:
            self.consecutive_disengaged = 0

        self.engagement_state = next_state
        self.steps += 1
        self.last_action = action

        dropout = self.consecutive_disengaged >= 3
        success = self.topics_completed >= self.n_topics and not self.session_had_negative

        if dropout:
            reward += self.rc.dropout_terminal_penalty
        if success:
            reward += self.rc.session_complete_bonus

        terminated = bool(dropout or success)
        truncated = bool(self.steps >= self.max_episode_steps)

        self.last_reward = reward

        obs = self._build_observation()
        info = self._build_info()
        info["dropout"] = dropout
        info["success"] = success

        if self.render_mode in ("human", "human_manual"):
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Delegate to the PyOpenGL renderer (human mode) or return an RGB array.

        Imported lazily so that headless code paths (training, unit tests)
        never require PyOpenGL/GLFW to be importable or a display to exist.
        """
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from environment.rendering import AdaptLearnRenderer

            self._renderer = AdaptLearnRenderer()

        return self._renderer.render(self, mode=self.render_mode)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _transition_logits(
        self, *, gap: float, prev_state: EngagementState, action: int
    ) -> np.ndarray:
        """Compute unnormalized log-scores for each candidate next state.

        This is the heart of the "explicit, documented model" requirement:
        every term added below corresponds to exactly one named constant in
        `TransitionConfig`, and one sentence of theory-grounded rationale in
        this function's inline comments.
        """
        tc = self.tc

        # Scaffolding actions (hint / simplify) make the task *feel* easier
        # for the purposes of the engagement transition, without touching
        # the stored, agent-observed difficulty (decision 8).
        effective_gap = gap
        if action == ACTION_OFFER_HINT:
            effective_gap = gap - tc.hint_gap_relief
        elif action == ACTION_REPEAT_SIMPLIFIED:
            effective_gap = gap - tc.simplify_gap_relief

        logits = np.zeros(N_STATES, dtype=np.float64)

        # ENGAGED: Gaussian bump centered on effective_gap == 0 (Flow Theory
        # challenge-skill match).
        match_score = np.exp(-(effective_gap**2) / (2.0 * tc.match_sigma**2))
        logits[EngagementState.ENGAGED] += tc.match_engage_weight * match_score

        # FRUSTRATED: rises once effective_gap exceeds a "too hard" threshold.
        over = max(0.0, effective_gap - tc.frustrate_gap_threshold)
        logits[EngagementState.FRUSTRATED] += tc.gap_frustrate_weight * over

        # BORED: rises once -effective_gap exceeds a "too easy" threshold.
        under = max(0.0, -effective_gap - tc.bored_gap_threshold)
        logits[EngagementState.BORED] += tc.gap_bored_weight * under

        # CONFUSED: baseline rate plus a contribution from how error-prone
        # the current (nominal, not effective) gap makes the task -- an
        # erratic, high-error experience is itself a source of confusion,
        # independent of the challenge-skill axis.
        error_proxy = max(0.0, gap) / tc.gap_range
        logits[EngagementState.CONFUSED] += (
            tc.base_confusion_logit + tc.confusion_error_weight * error_proxy
        )

        # Affective inertia: current state gets a bonus toward persisting.
        logits[prev_state] += tc.persistence_bonus

        # --- Action-specific perturbations (beyond the gap effects above) ---
        if action == ACTION_CHANGE_TASK_TYPE:
            # A format change is momentarily disorienting (erratic/unclear
            # presentation -> higher confusion risk) but can also relieve
            # staleness/boredom via novelty.
            logits[EngagementState.CONFUSED] += tc.confusion_format_change_bonus
            logits[EngagementState.ENGAGED] += tc.novelty_engage_bonus
        elif action == ACTION_OFFER_HINT:
            logits[EngagementState.CONFUSED] -= tc.confusion_clarity_relief
        elif action == ACTION_ADD_ENCOURAGEMENT:
            logits[EngagementState.FRUSTRATED] -= tc.encouragement_frustration_relief
            logits[EngagementState.ENGAGED] += tc.encouragement_engage_bonus
        elif action == ACTION_INTRODUCE_CHALLENGE:
            logits[EngagementState.ENGAGED] += tc.challenge_engage_bonus
            logits[EngagementState.BORED] -= tc.challenge_bored_relief
        elif action == ACTION_REPEAT_SIMPLIFIED:
            logits[EngagementState.CONFUSED] -= tc.confusion_clarity_relief

        return logits

    def _simulate_behavioral_signals(
        self, gap: float, state: EngagementState
    ) -> tuple[float, float]:
        """Generate this step's (response_time, error_rate) observation features.

        Both are modelled as a base rate, plus a term driven by how far
        outside the learner's ability the task difficulty sits, plus a
        state-specific offset capturing behaviour that isn't explained by
        the gap alone (e.g. a bored learner answers fast but carelessly), plus
        i.i.d. Gaussian noise. Everything is clipped to the valid [0, 1]
        observation range.
        """
        tc = self.tc
        rng = self.np_random

        error_rate = (
            tc.base_error_rate
            + tc.error_rate_gap_coef * max(gap, 0.0) / tc.gap_range
            + tc.state_error_offset[int(state)]
            + rng.normal(0.0, tc.error_rate_noise_std)
        )
        response_time = (
            tc.base_response_time
            + tc.response_time_gap_coef * abs(gap) / tc.gap_range
            + tc.state_rt_offset[int(state)]
            + rng.normal(0.0, tc.response_time_noise_std)
        )

        return float(np.clip(response_time, 0.0, 1.0)), float(np.clip(error_rate, 0.0, 1.0))

    def _compute_reward(self, prev_state: EngagementState, next_state: EngagementState) -> float:
        """Reward for this step's (prev_state -> next_state) transition.

        See module docstring decisions (3) and (4) for why CONFUSED has a
        reward at all, and why FRUSTRATED/BORED penalties apply on every
        step spent in that state rather than only on entry.
        """
        rc = self.rc

        if next_state == EngagementState.ENGAGED:
            if prev_state == EngagementState.ENGAGED:
                return rc.engaged_maintain_reward
            return rc.engaged_reach_reward
        if next_state == EngagementState.FRUSTRATED:
            return rc.frustrated_step_reward
        if next_state == EngagementState.BORED:
            return rc.bored_step_reward
        return rc.confused_step_reward  # EngagementState.CONFUSED

    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(9, dtype=np.float32)
        obs[int(self.engagement_state)] = 1.0  # one-hot block, indices [0:4]
        obs[4] = self.response_time
        obs[5] = self.error_rate
        obs[6] = min(1.0, self.hint_count / self._hint_count_norm_cap)
        obs[7] = (self.difficulty - MIN_DIFFICULTY) / (MAX_DIFFICULTY - MIN_DIFFICULTY)
        obs[8] = min(1.0, self.time_on_task / self._time_on_task_norm_cap)
        return obs

    def _build_info(self) -> dict[str, Any]:
        # NOTE: `ability` and `gap` are included here for logging/rendering/
        # analysis only. `info` is never passed to the policy network by
        # Stable-Baselines3 (or any standard Gymnasium-compatible learner),
        # so exposing them here does not violate the partial-observability
        # design (decision 1) -- the agent still never sees `ability`.
        return {
            "ability": self.ability,
            "gap": self.difficulty - self.ability,
            "difficulty": self.difficulty,
            "engagement_state": self.engagement_state.name,
            "hint_count": self.hint_count,
            "time_on_task": self.time_on_task,
            "consecutive_disengaged": self.consecutive_disengaged,
            "topics_completed": self.topics_completed,
            "session_had_negative": self.session_had_negative,
            "steps": self.steps,
            "last_action": None if self.last_action is None else ACTION_NAMES[self.last_action],
            "last_reward": self.last_reward,
        }


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically-stable softmax used to turn transition logits into probabilities."""
    scaled = logits / max(temperature, 1e-8)
    shifted = scaled - np.max(scaled)
    exp = np.exp(shifted)
    return exp / exp.sum()

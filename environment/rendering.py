"""Real-time PyOpenGL renderer for AdaptLearn-v1: a fixed-camera classroom scene.

WHY PyOpenGL (AND NOT MuJoCo OR THREE.JS)
-------------------------------------------
This is a short, explicit justification, as required by the assignment brief.

  * Not MuJoCo. MuJoCo is a physics engine: its entire value proposition is
    accurate rigid-body/joint dynamics, contact forces, and actuator
    simulation. AdaptLearn-v1 has no physical dynamics at all -- the
    simulated child is a cognitive/affective *state machine* (engagement
    transitions driven by a difficulty-ability gap and pedagogical actions),
    not a mechanical body. Adopting MuJoCo would mean pulling in an entire
    physics engine and a joint-rigged articulated body to solve a problem
    this environment does not have, for a payoff (physical realism) this
    environment does not need.
  * Not Three.js. Three.js would require running a JS/WebGL renderer in a
    browser, driven from Python either by a local web server + browser
    automation, or a websocket bridge -- an extra process, an extra
    language, and a network hop between "the RL loop decided an action" and
    "a pixel changed". That is a fragile stack for the one thing this
    project actually needs from rendering: a single reliable process that
    can be screen-recorded end-to-end while `main.py` runs a trained agent.
  * PyOpenGL fits what's actually required: simple low-poly geometry (a
    stylized avatar, a small room's worth of furniture, no skeletal
    rigging), no physics, and a real-time, single-process, recordable
    window. Fixed-function OpenGL (immediate-mode `glBegin`/`glEnd`, no
    shaders) keeps the implementation approachable and fully sufficient for
    this visual complexity level.

WHY A CLASSROOM SCENE, NOT A "PATH" (DESIGN HISTORY)
---------------------------------------------------------
An earlier version of this renderer depicted the episode as a winding path
the avatar walked along, with terrain responding to actions. That metaphor
was reconsidered and discarded: AdaptLearn-v1's actual RL mechanics involve
no spatial decisions whatsoever -- the agent chooses one of 8 pedagogical
actions per step, which shift a simulated learner's *engagement state*.
Representing that as literal movement through space implied a kind of
choice the environment doesn't have, and actively worked against
legibility (a viewer has to first discount "why is she walking/turning"
before they can read what actually happened). This version instead depicts
a single, static classroom scene -- the learner seated at a desk the whole
episode -- with actions and state changes visualized as effects that
appear *in place*, which is a more honest match to what the environment
actually models.

Coordinate convention: +x = right, +y = up, +z = toward the camera (out of
the back-left corner of the room), matching common OpenGL convention. The
camera is fixed for the entire episode -- see CAMERA_EYE/CAMERA_TARGET
below -- there is no per-step camera movement in this design.

STAGE 4 OF 4 (current): the above, plus a HUD (bottom bar: last action
taken, engagement state, difficulty, reward, engagement-over-time graph --
all rendered as real readable text via a small self-authored bitmap font,
see _FONT_5X7), and a staged per-step choreography: action cue appears and
holds, THEN the mood/posture tween begins (not simultaneously), THEN a
brief pause, before the next step's presentation starts. `render()` now
paces this internally in real time for render_mode="human", so a plain
`env.step()` loop (see tests/manual_watch_episode.py) is correctly paced
with no special-case timing code needed by the caller.

render_mode="human_manual" (recording aid): identical action-cue/mood
animation, but the trailing pause becomes an indefinite hold that only
ends on a spacebar press, instead of a fixed duration -- see
`AdaptLearnRenderer._wait_for_spacebar_press`. This only changes when
render() returns control to the caller; it has no effect on the
environment, the agent's policy, or the already-fully-determined episode
sequence being played back.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import glfw
from OpenGL.GL import *  # noqa: F403 -- standard practice for immediate-mode PyOpenGL code

from environment.custom_env import (
    ACTION_ADD_ENCOURAGEMENT,
    ACTION_ADVANCE_TOPIC,
    ACTION_CHANGE_TASK_TYPE,
    ACTION_INTRODUCE_CHALLENGE,
    ACTION_NAMES,
    ACTION_OFFER_HINT,
    ACTION_REPEAT_SIMPLIFIED,
    MAX_DIFFICULTY,
    MIN_DIFFICULTY,
    EngagementState,
)

# --------------------------------------------------------------------------
# Window / rendering constants
# --------------------------------------------------------------------------

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800

# --------------------------------------------------------------------------
# Room layout (world units; floor at y=0)
# --------------------------------------------------------------------------

ROOM_HALF_X = 3.0  # room spans x in [-ROOM_HALF_X, ROOM_HALF_X]
ROOM_HALF_Z = 3.0  # room spans z in [-ROOM_HALF_Z, ROOM_HALF_Z]
WALL_HEIGHT = 2.6

FLOOR_COLOR = (0.50, 0.38, 0.27)  # warm wood tone
WALL_COLOR = (0.90, 0.88, 0.83)  # soft off-white
BASEBOARD_COLOR = (0.98, 0.97, 0.94)

RUG_COLOR = (0.55, 0.32, 0.34)
RUG_CENTER = (-0.55, -0.85)  # (x, z)
RUG_SIZE = (2.0, 1.8)  # (width along x, depth along z)

# Desk + chair + tablet
DESK_POS = (-0.55, -0.9)  # (x, z) floor footprint center
DESK_SIZE = (1.05, 0.72, 0.55)  # (width x, height y, depth z)
DESK_COLOR = (0.62, 0.44, 0.30)

CHAIR_POS = (-0.55, -0.35)  # (x, z)
CHAIR_SEAT_Y = 0.45
CHAIR_COLOR = (0.35, 0.36, 0.42)

TABLET_POS = (-0.30, DESK_SIZE[1] + 0.015, -0.78)  # sits on the desk surface, near edge, offset from her arm
TABLET_SIZE = (0.34, 0.02, 0.24)
TABLET_BODY_COLOR = (0.15, 0.15, 0.17)
TABLET_SCREEN_COLOR = (0.35, 0.55, 0.85)

# Blackboard, mounted on the left wall (x = -ROOM_HALF_X)
BLACKBOARD_CENTER = (-ROOM_HALF_X + 0.02, 1.55, -1.0)  # (x, y, z)
BLACKBOARD_SIZE = (1.0, 1.6)  # (height, width-along-z)
BLACKBOARD_FRAME_MARGIN = 0.09
BLACKBOARD_COLOR = (0.12, 0.28, 0.20)
BLACKBOARD_FRAME_COLOR = (0.42, 0.28, 0.18)

# Blackboard content: small hardcoded "curriculum" problems, real text
# rather than abstract shapes, so a viewer reads a genuine (if simple)
# task rather than an icon (see _draw_text_on_board). Purely a legibility
# aid -- these strings have no bearing on the RL logic whatsoever, they
# just visualize advance_topic (cycle within the current category) and
# change_task_type (swap category) as two visually distinct effects.
# "X" stands in for a multiplication sign (no dedicated glyph in
# _FONT_5X7, and it reads unambiguously in a simple arithmetic string).
BLACKBOARD_PROBLEMS: dict[str, tuple[str, ...]] = {
    "arithmetic": ("2 + 2 = ?", "3 X 5 = ?", "4 + 4 = ?"),
    "sequence": ("3, 6, 9, ___?", "A, B, C, ___?"),
}
BLACKBOARD_CATEGORIES: tuple[str, ...] = tuple(BLACKBOARD_PROBLEMS.keys())
BLACKBOARD_TEXT_CHAR_SIZE = 0.095
BLACKBOARD_TEXT_COLOR = (0.90, 0.90, 0.85)

# Window, on the back wall (z = -ROOM_HALF_Z)
WINDOW_CENTER = (1.3, 1.65, -ROOM_HALF_Z + 0.02)
WINDOW_SIZE = (1.1, 0.9)  # (width along x, height)
WINDOW_FRAME_COLOR = (0.94, 0.93, 0.90)
WINDOW_SKY_COLOR = (0.62, 0.78, 0.90)

# Plant, in the back-left corner
PLANT_POS = (-ROOM_HALF_X + 0.5, ROOM_HALF_Z - 2.6)  # (x, z)
POT_COLOR = (0.55, 0.30, 0.22)
LEAF_COLOR = (0.24, 0.52, 0.28)

# Seated girl
GIRL_POS = CHAIR_POS  # she sits exactly at the chair -- floor footprint, seated at CHAIR_SEAT_Y
GIRL_FACING_DEG = 0.0  # local forward = -z; desk is in the -z direction from the chair, so she faces it

HAIR_COLOR = (0.32, 0.18, 0.10)
SKIN_COLOR = (0.94, 0.80, 0.68)
CLOTHES_COLOR = (0.70, 0.55, 0.62)

# --------------------------------------------------------------------------
# Engagement-state indicator + posture cues (Stage 2)
# --------------------------------------------------------------------------

# Indicator-orb color per state. Matches the brief exactly: green=ENGAGED,
# red=FRUSTRATED, grey/blue=BORED, amber=CONFUSED.
STATE_COLOR = {
    EngagementState.ENGAGED: (0.25, 0.85, 0.35),
    EngagementState.FRUSTRATED: (0.85, 0.20, 0.20),
    EngagementState.BORED: (0.55, 0.62, 0.78),
    EngagementState.CONFUSED: (0.95, 0.72, 0.15),
}

# Posture targets per state, all in degrees:
#   torso_pitch -- lean forward(+)/back(-) at the hip
#   head_pitch  -- additional nod down(+)/up(-) at the neck, on top of torso
#   head_roll   -- side tilt at the neck (a "quizzical"/drooping tilt)
#   idle_amp/idle_freq/idle_axis -- a continuous idle oscillation layered on
#     top of the static targets above (pure sway/nod, not a pose change),
#     added on the named axis ("pitch" or "roll")
STATE_POSTURE = {
    # Upright with a subtle attentive nod.
    EngagementState.ENGAGED: dict(
        torso_pitch=-4.0, head_pitch=-2.0, head_roll=0.0, idle_amp=4.0, idle_freq=1.8, idle_axis="pitch"
    ),
    # Slumped/head-down, and mostly still -- frustration reads as low energy here.
    EngagementState.FRUSTRATED: dict(
        torso_pitch=20.0, head_pitch=16.0, head_roll=0.0, idle_amp=1.0, idle_freq=0.4, idle_axis="pitch"
    ),
    # Slow head droop/sway to one side, like resting on a hand.
    EngagementState.BORED: dict(
        torso_pitch=8.0, head_pitch=6.0, head_roll=12.0, idle_amp=6.0, idle_freq=0.5, idle_axis="roll"
    ),
    # A tilted, "questioning" head with a quicker wobble than BORED's droop.
    EngagementState.CONFUSED: dict(
        torso_pitch=2.0, head_pitch=-2.0, head_roll=8.0, idle_amp=10.0, idle_freq=2.2, idle_axis="roll"
    ),
}

TRANSITION_SECONDS = 1.0  # mood/posture tween duration -- see the Stage 4 pacing note below; slowed twice
                          # over from the brief's literal "10-15 frames" (~0.4s) based on direct user
                          # feedback that even ~2.85s/step overall was still too fast to follow.
NECK_Y = 0.78  # local y of the neck pivot, relative to the seat root (head/ponytail/orb hang off this)

# --------------------------------------------------------------------------
# Fixed camera (never moves during an episode -- see module docstring)
# --------------------------------------------------------------------------

CAMERA_EYE = (2.95, 2.85, 2.4)
CAMERA_TARGET = (-0.75, 1.05, -0.95)
CAMERA_UP = (0.0, 1.0, 0.0)
CAMERA_FOV_DEG = 40.0

# --------------------------------------------------------------------------
# Action-cue visuals (Stage 3)
# --------------------------------------------------------------------------

# offer_hint / introduce_challenge / repeat_simplified / add_encouragement
# all use this appear -> hold -> fade envelope for their popup icon.
# increase/decrease_difficulty are visualized entirely by the persistent
# gauge below (no popup); change_task_type / advance_topic are visualized
# by the tablet-screen / blackboard content swap (plus a glow pulse using
# this same envelope, so the swap itself doesn't go unnoticed).
#
# These (and the Stage 4 pacing constants below) were slowed down twice
# over from their original values, based on direct user feedback after
# watching real episodes: first pass ~1.55s/step (technically within the
# brief's 1.5-2.5s target, but too fast to follow), second pass ~2.85s/step
# (still too fast) -- current pass targets a flat 5.0s/step, well past the
# brief's suggested range, per "tune for legibility over speed" and the
# user's explicit request.
ACTION_CUE_APPEAR_SECONDS = 0.3
ACTION_CUE_HOLD_SECONDS = 1.7
ACTION_CUE_FADE_SECONDS = 0.8
ACTION_CUE_TOTAL_SECONDS = ACTION_CUE_APPEAR_SECONDS + ACTION_CUE_HOLD_SECONDS + ACTION_CUE_FADE_SECONDS

# Persistent difficulty gauge: a small vertical fill-bar beside the desk
# that animates toward env.difficulty whenever it changes (increase/
# decrease_difficulty, and also advance_topic, which raises difficulty as
# part of normal curriculum progression -- see custom_env.py's step()).
GAUGE_TRANSITION_SECONDS = 1.0
GAUGE_ANCHOR = (DESK_POS[0] + DESK_SIZE[0] / 2 + 0.22, DESK_SIZE[1], -0.85)  # (x, y0, z)
GAUGE_WIDTH = 0.10
GAUGE_DEPTH = 0.10
GAUGE_MAX_HEIGHT = 0.55
GAUGE_FRAME_COLOR = (0.30, 0.30, 0.33)
GAUGE_FILL_COLOR = (0.95, 0.65, 0.20)

# Anchor points for the billboarded popup icons and the tablet/blackboard
# content swaps.
TABLET_CUE_ANCHOR = (TABLET_POS[0] + 0.30, TABLET_POS[1] + 0.55, TABLET_POS[2])  # clear of her reaching arm
ENCOURAGEMENT_ANCHOR = (GIRL_POS[0] + 0.45, 1.55, GIRL_POS[1] + 0.05)

TABLET_SCREEN_PALETTE = [
    (0.35, 0.55, 0.85),
    (0.85, 0.55, 0.30),
    (0.40, 0.75, 0.45),
    (0.70, 0.45, 0.80),
]

# --------------------------------------------------------------------------
# Staged per-step choreography + HUD (Stage 4)
# --------------------------------------------------------------------------

# The brief's intended "story" of one step is sequential, not simultaneous:
# action icon appears and holds, THEN the mood/posture tween begins, THEN a
# brief pause before the next step. STEP_ACTION_PHASE_SECONDS is timed to
# end right as the action-cue icon starts fading (see ACTION_CUE_* above),
# so the mood tween's start and the icon's fade read as one handoff rather
# than two competing changes.
STEP_ACTION_PHASE_SECONDS = ACTION_CUE_APPEAR_SECONDS + ACTION_CUE_HOLD_SECONDS
STEP_MOOD_PHASE_SECONDS = TRANSITION_SECONDS
STEP_HOLD_PAUSE_SECONDS = 2.0
STEP_TOTAL_SECONDS = STEP_ACTION_PHASE_SECONDS + max(ACTION_CUE_FADE_SECONDS, STEP_MOOD_PHASE_SECONDS) + STEP_HOLD_PAUSE_SECONDS

HUD_HEIGHT_FRACTION = 0.20  # bottom bar height, as a fraction of the framebuffer height
HUD_BG_COLOR = (0.06, 0.07, 0.09)
HUD_BG_ALPHA = 0.82
HUD_TEXT_COLOR = (0.92, 0.93, 0.95)
HUD_LABEL_COLOR = (0.62, 0.65, 0.70)  # dimmer, for field labels vs. values
GRAPH_BAND_ALPHA = 0.20


def _normalize(v):
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n < 1e-8:
        return v
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _smoothstep(t: float) -> float:
    """Eased 0..1 progress curve (zero velocity at both ends), so a tween
    reads as a deliberate transition rather than a linear slide."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp3(a, b, t: float):
    return (_lerp(a[0], b[0], t), _lerp(a[1], b[1], t), _lerp(a[2], b[2], t))


def _perspective_matrix(fovy_deg: float, aspect: float, znear: float, zfar: float):
    """Standard OpenGL perspective projection matrix, column-major flattened,
    equivalent to the classic gluPerspective formula (reimplemented by hand
    to avoid a GLU dependency -- see module docstring)."""
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    return [
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (zfar + znear) / (znear - zfar), -1,
        0, 0, (2 * zfar * znear) / (znear - zfar), 0,
    ]


def _look_at_matrix(eye, center, up):
    """Standard OpenGL view matrix, column-major flattened, equivalent to the
    classic gluLookAt formula (reimplemented by hand -- see module docstring)."""
    f = _normalize(_sub(center, eye))
    s = _normalize(_cross(f, up))
    u = _cross(s, f)
    return [
        s[0], u[0], -f[0], 0,
        s[1], u[1], -f[1], 0,
        s[2], u[2], -f[2], 0,
        -_dot(s, eye), -_dot(u, eye), _dot(f, eye), 1,
    ]


def _billboard_matrix():
    """A rotation matrix (no translation) that maps local +x/+y/+z to
    world right/up/toward-camera, for drawing 2D icon cues that always face
    the camera. Computed once from CAMERA_EYE/TARGET/UP and reused for the
    renderer's lifetime, since the camera is fixed (see module docstring).

    NOTE: unlike `_look_at_matrix`, this is a plain model-space rotation,
    not a view matrix -- it must NOT be transposed. (A view matrix's
    rotation part is intentionally the transpose of the camera's own
    orientation, since inverting a pure rotation is the same as
    transposing it; a billboard's model matrix has no such inversion and
    needs the untransposed columns, or the quad ends up rotated 90 degrees
    -- i.e. seen edge-on -- rather than facing the camera.)"""
    forward = _normalize(_sub(CAMERA_EYE, CAMERA_TARGET))  # points toward the camera
    right = _normalize(_cross(CAMERA_UP, forward))
    true_up = _cross(forward, right)
    return [
        right[0], right[1], right[2], 0,
        true_up[0], true_up[1], true_up[2], 0,
        forward[0], forward[1], forward[2], 0,
        0, 0, 0, 1,
    ]


@dataclass
class _StepRecord:
    """One env step's worth of bookkeeping: what happened, for later HUD/
    history use (engagement-over-time graph, "last action taken", etc).
    Deliberately holds no spatial information -- nothing in this scene
    moves through space (see module docstring)."""

    index: int
    action: int | None
    state: EngagementState
    reward: float
    difficulty: int


class AdaptLearnRenderer:
    """Owns the GLFW window/GL context and draws one AdaptLearnEnv frame."""

    def __init__(self, width: int = WINDOW_WIDTH, height: int = WINDOW_HEIGHT) -> None:
        if not glfw.init():
            raise RuntimeError(
                "GLFW failed to initialize -- a display/window server is required to "
                "render AdaptLearn-v1. Run headless (render_mode=None) for training."
            )

        # Request a legacy (2.1) context: macOS only supports either this or
        # a 3.2+ core profile, and core profiles remove immediate-mode
        # drawing (glBegin/glEnd), which this simple low-poly renderer relies
        # on for readability of the code.
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self.window = glfw.create_window(width, height, "AdaptLearn-v1", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW failed to create a window.")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # vsync

        self._init_gl_state()

        # The camera is fixed for the whole episode (see module docstring),
        # so its matrices are computed once here rather than per frame.
        self._projection_matrix_cache: dict[float, list] = {}
        self._view_matrix = _look_at_matrix(CAMERA_EYE, CAMERA_TARGET, CAMERA_UP)

        self.history: list[_StepRecord] = []
        self._last_steps_seen: int | None = None

        # Engagement-state tween tracking (Stage 2): `_pose_to` is always
        # the most recently observed env.engagement_state; when it changes,
        # a new transition starts from `_pose_from` (the previous target)
        # rather than from wherever the tween had actually interpolated to
        # -- a small simplification that only matters if the state changes
        # again faster than TRANSITION_SECONDS, which the paced integration
        # (Stage 4) won't do.
        self._pose_from: EngagementState = EngagementState.ENGAGED
        self._pose_to: EngagementState = EngagementState.ENGAGED
        self._pose_transition_start: float = 0.0
        self._t0 = time.time()

        # Action-cue tracking (Stage 3): which action last fired and when,
        # driving the appear/hold/fade envelope. None means nothing to show
        # (e.g. right after reset(), before any step has happened).
        self._action_cue_action: int | None = None
        self._action_cue_start: float = 0.0

        # Difficulty-gauge tween (Stage 3): same from/to/start pattern as
        # the mood tween. `None` sentinel means "no step registered yet" --
        # the first registration snaps to the initial difficulty with no
        # animation, rather than animating in from a hardcoded default.
        self._gauge_from: int | None = None
        self._gauge_to: int | None = None
        self._gauge_transition_start: float = 0.0

        # Tablet-screen content swap counter (Stage 3): incremented on
        # change_task_type, so repeated triggers still visibly cycle rather
        # than staying static.
        self._tablet_content_index: int = 0

        # Blackboard content state: which category is showing, plus a
        # per-category problem index so switching category and back doesn't
        # lose your place. advance_topic cycles the problem within the
        # current category; change_task_type swaps the category -- see
        # `_register_step` and BLACKBOARD_PROBLEMS above.
        self._blackboard_category_index: int = 0
        self._blackboard_problem_indices: dict[str, int] = {cat: 0 for cat in BLACKBOARD_CATEGORIES}

        # Billboard matrix for camera-facing icon cues (Stage 3): constant
        # for the renderer's lifetime since the camera never moves.
        self._billboard_matrix = _billboard_matrix()

        # Staged per-step choreography (Stage 4): `_register_step` schedules
        # the mood tween to start once the action-cue phase ends, via
        # `_scheduled_mood_start`; `_current_mood` consumes it. If a caller
        # changes env.engagement_state directly without going through a
        # real step (see tests/manual_cycle_states.py), no schedule is
        # pending and the tween starts immediately instead -- see
        # `_current_mood`'s docstring.
        self._step_phase_start: float = time.time()
        self._scheduled_mood_start: float | None = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _init_gl_state(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (3.0, 8.0, 4.0, 0.0))  # w=0 -> directional light
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 0.98, 1.0))
        # Interior scene: higher ambient than an outdoor scene would need, so
        # walls/floor away from the key light don't read as pure black.
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.48, 0.47, 0.50, 1.0))
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)  # keep normals unit-length even after glScalef
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.05, 0.05, 0.07, 1.0)  # outside the room reads as neutral dark, not sky

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def render(self, env, mode: str = "human"):
        if glfw.window_should_close(self.window):
            return None if mode != "rgb_array" else self._blank_frame()

        advanced = self._sync_history_with_env(env)

        if mode == "rgb_array":
            self._draw_frame(env)
            return self._read_pixels()

        # A genuine new step (with an action -- reset's synthetic first
        # record has action=None) runs its full staged presentation (action
        # cue -> mood tween, then a hold) in real time before returning, so
        # a plain `env.step()` loop is correctly paced with no special-case
        # timing code needed by the caller (see tests/manual_watch_episode.py).
        # A stray extra render() call, or reset()'s call, just draws the
        # current frame once.
        #
        # "human" and "human_manual" share the identical action-cue/mood
        # animation; they differ only in how the trailing hold ends: "human"
        # holds for a fixed STEP_HOLD_PAUSE_SECONDS, "human_manual" instead
        # holds indefinitely until a spacebar press (see
        # `_wait_for_spacebar_press`) -- this does not touch env/agent
        # behavior at all, it only paces when render() returns control to
        # the caller, purely for recording purposes (see module docstring).
        if advanced and self.history[-1].action is not None:
            animate_seconds = STEP_ACTION_PHASE_SECONDS + max(ACTION_CUE_FADE_SECONDS, STEP_MOOD_PHASE_SECONDS)
            start = time.time()
            while time.time() - start < animate_seconds:
                if glfw.window_should_close(self.window):
                    break
                self._draw_frame(env)
                glfw.swap_buffers(self.window)
                glfw.poll_events()

            if glfw.window_should_close(self.window):
                return None

            if mode == "human_manual":
                self._wait_for_spacebar_press(env)
            else:
                hold_start = time.time()
                while time.time() - hold_start < STEP_HOLD_PAUSE_SECONDS:
                    if glfw.window_should_close(self.window):
                        break
                    self._draw_frame(env)
                    glfw.swap_buffers(self.window)
                    glfw.poll_events()
        else:
            self._draw_frame(env)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        return None

    def _wait_for_spacebar_press(self, env) -> None:
        """Blocks -- while keeping the window responsive by redrawing the
        current, already-fully-advanced frame each iteration -- until the
        user presses spacebar. Debounced: if spacebar is already held down
        when this is called (e.g. still held from the press that ended the
        PREVIOUS wait), first waits for release, so one physical press can't
        be consumed as two advances."""
        while glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS:
            if glfw.window_should_close(self.window):
                return
            self._draw_frame(env)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        while glfw.get_key(self.window, glfw.KEY_SPACE) != glfw.PRESS:
            if glfw.window_should_close(self.window):
                return
            self._draw_frame(env)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

    def close(self) -> None:
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()

    # ------------------------------------------------------------------ #
    # History bookkeeping (no spatial data -- see _StepRecord)
    # ------------------------------------------------------------------ #

    def _sync_history_with_env(self, env) -> bool:
        """Grow `self.history` to match env's current step count. Returns
        True if a new record was appended (an env.step()/reset() genuinely
        happened since the last render() call). Also registers each new
        record with `_register_step` to drive the Stage 3 action-cue/gauge
        tweens (see that method)."""
        if env.steps == 0:
            if self._last_steps_seen != 0 or not self.history:
                self.history = [
                    _StepRecord(index=0, action=None, state=env.engagement_state, reward=0.0, difficulty=env.difficulty)
                ]
                self._last_steps_seen = 0
                self._register_step(None, env.difficulty)
                return True
            return False

        if not self.history:
            # First render() call happened mid-episode (caller skipped the
            # reset frame) -- start history at the current step index
            # rather than mislabeling it as index 0.
            self.history = [
                _StepRecord(
                    index=env.steps,
                    action=env.last_action,
                    state=env.engagement_state,
                    reward=env.last_reward,
                    difficulty=env.difficulty,
                )
            ]
            self._last_steps_seen = env.steps
            self._register_step(env.last_action, env.difficulty)
            return True

        if env.steps != self._last_steps_seen:
            self.history.append(
                _StepRecord(
                    index=self.history[-1].index + 1,
                    action=env.last_action,
                    state=env.engagement_state,
                    reward=env.last_reward,
                    difficulty=env.difficulty,
                )
            )
            self._last_steps_seen = env.steps
            self._register_step(env.last_action, env.difficulty)
            return True

        return False

    def _register_step(self, action: int | None, difficulty: int) -> None:
        """Called once per genuinely-new step (see `_sync_history_with_env`)
        to start the difficulty-gauge tween (if difficulty changed) and the
        action-cue envelope (if an action was taken), and to schedule the
        mood tween's start (Stage 4: the action cue plays out *first*, then
        the mood/posture transitions -- see `_current_mood`). Test scripts
        that mutate env.steps/env.last_action/env.difficulty directly
        (rather than running a real episode) exercise this exact path --
        see tests/manual_trigger_actions.py."""
        now = time.time()
        self._step_phase_start = now
        self._scheduled_mood_start = now + STEP_ACTION_PHASE_SECONDS

        if self._gauge_to is None:
            self._gauge_from = self._gauge_to = difficulty
        elif difficulty != self._gauge_to:
            self._gauge_from = self._gauge_to
            self._gauge_to = difficulty
            self._gauge_transition_start = now

        if action is not None:
            self._action_cue_action = action
            self._action_cue_start = now
            if action == ACTION_CHANGE_TASK_TYPE:
                self._tablet_content_index += 1
                self._blackboard_category_index = (self._blackboard_category_index + 1) % len(BLACKBOARD_CATEGORIES)
            elif action == ACTION_ADVANCE_TOPIC:
                cat = BLACKBOARD_CATEGORIES[self._blackboard_category_index]
                n = len(BLACKBOARD_PROBLEMS[cat])
                self._blackboard_problem_indices[cat] = (self._blackboard_problem_indices[cat] + 1) % n

    # ------------------------------------------------------------------ #
    # Frame drawing
    # ------------------------------------------------------------------ #

    def _draw_frame(self, env) -> None:
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        fb_width, fb_height = max(fb_width, 1), max(fb_height, 1)

        glViewport(0, 0, fb_width, fb_height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        aspect = fb_width / fb_height
        if aspect not in self._projection_matrix_cache:
            self._projection_matrix_cache[aspect] = _perspective_matrix(CAMERA_FOV_DEG, aspect, 0.1, 100.0)
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(self._projection_matrix_cache[aspect])
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(self._view_matrix)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        cue_intensity = self._action_cue_intensity()
        tablet_glow = cue_intensity if self._action_cue_action == ACTION_CHANGE_TASK_TYPE else 0.0
        # The board reacts to both actions that touch it -- advance_topic
        # (new problem, same category) and change_task_type (new category) --
        # so both read as "the board just changed", while the text content
        # itself is what actually distinguishes the two (see _register_step).
        board_glow = cue_intensity if self._action_cue_action in (ACTION_ADVANCE_TOPIC, ACTION_CHANGE_TASK_TYPE) else 0.0

        self._draw_room()
        self._draw_rug()
        self._draw_desk()
        self._draw_tablet(self._tablet_content_index, tablet_glow)
        self._draw_chair()
        self._draw_blackboard(self._blackboard_category_index, self._blackboard_problem_indices, board_glow)
        self._draw_window()
        self._draw_plant()
        self._draw_difficulty_gauge()
        self._draw_action_cue(cue_intensity)

        color, posture = self._current_mood(env)
        self._draw_girl_seated(color, posture)

        self._draw_hud(env, fb_width, fb_height, color)

    # -- engagement-state tween (Stage 2) -------------------------------------

    def _current_mood(self, env):
        """Returns (indicator_color, posture_params) for the CURRENT frame,
        smoothly tweened toward env.engagement_state. Reads env.engagement_state
        directly every frame (not from `self.history`), so this responds
        identically whether the state changed via a real env.step() or a
        test script setting env.engagement_state directly -- see
        tests/manual_cycle_states.py, which relies on exactly that.

        Stage 4 staging: if `_register_step` scheduled a delayed start (the
        normal case for a real step -- the mood should only start tweening
        once the action-cue phase finishes, see that method), use it and
        consume it. Otherwise (a test script changed env.engagement_state
        directly, with no step registered) start tweening immediately, so
        tests/manual_cycle_states.py keeps working exactly as it did in
        Stage 2."""
        target = env.engagement_state
        if target != self._pose_to:
            self._pose_from = self._pose_to
            self._pose_to = target
            if self._scheduled_mood_start is not None:
                self._pose_transition_start = self._scheduled_mood_start
                self._scheduled_mood_start = None
            else:
                self._pose_transition_start = time.time()

        elapsed = time.time() - self._pose_transition_start
        t = _smoothstep(min(1.0, elapsed / TRANSITION_SECONDS))

        color = _lerp3(STATE_COLOR[self._pose_from], STATE_COLOR[self._pose_to], t)
        from_posture = STATE_POSTURE[self._pose_from]
        to_posture = STATE_POSTURE[self._pose_to]
        posture = {
            "torso_pitch": _lerp(from_posture["torso_pitch"], to_posture["torso_pitch"], t),
            "head_pitch": _lerp(from_posture["head_pitch"], to_posture["head_pitch"], t),
            "head_roll": _lerp(from_posture["head_roll"], to_posture["head_roll"], t),
            "idle_amp": _lerp(from_posture["idle_amp"], to_posture["idle_amp"], t),
            "idle_freq": _lerp(from_posture["idle_freq"], to_posture["idle_freq"], t),
            # The idle axis itself doesn't tween (it's categorical); switch
            # it at the transition's midpoint so the outgoing state's idle
            # motion isn't visibly applied on the wrong axis for too long.
            "idle_axis": to_posture["idle_axis"] if t >= 0.5 else from_posture["idle_axis"],
        }
        return color, posture

    # -- action cues (Stage 3) ---------------------------------------------

    def _action_cue_intensity(self) -> float:
        """0..1..0 envelope for the currently active action-cue icon:
        appear (ease in), hold (flat), fade (ease out), then nothing."""
        if self._action_cue_action is None:
            return 0.0
        elapsed = time.time() - self._action_cue_start
        if elapsed < ACTION_CUE_APPEAR_SECONDS:
            return _smoothstep(elapsed / ACTION_CUE_APPEAR_SECONDS)
        if elapsed < ACTION_CUE_APPEAR_SECONDS + ACTION_CUE_HOLD_SECONDS:
            return 1.0
        fade_elapsed = elapsed - ACTION_CUE_APPEAR_SECONDS - ACTION_CUE_HOLD_SECONDS
        if fade_elapsed < ACTION_CUE_FADE_SECONDS:
            return 1.0 - _smoothstep(fade_elapsed / ACTION_CUE_FADE_SECONDS)
        return 0.0

    def _draw_action_cue(self, intensity: float) -> None:
        if intensity <= 0.001:
            return
        action = self._action_cue_action
        if action == ACTION_OFFER_HINT:
            self._draw_billboard_icon(TABLET_CUE_ANCHOR, 0.42, lambda: _draw_lightbulb_icon(intensity))
        elif action == ACTION_INTRODUCE_CHALLENGE:
            self._draw_billboard_icon(TABLET_CUE_ANCHOR, 0.42, lambda: _draw_starburst_icon(intensity))
        elif action == ACTION_REPEAT_SIMPLIFIED:
            self._draw_billboard_icon(TABLET_CUE_ANCHOR, 0.42, lambda: _draw_rewind_icon(intensity))
        elif action == ACTION_ADD_ENCOURAGEMENT:
            self._draw_billboard_icon(ENCOURAGEMENT_ANCHOR, 0.48, lambda: _draw_sparkle_icon(intensity))
        # increase/decrease_difficulty: no popup, see _draw_difficulty_gauge.
        # change_task_type/advance_topic: no popup, see _draw_tablet/_draw_blackboard.

    def _draw_billboard_icon(self, anchor, scale: float, draw_fn) -> None:
        glDisable(GL_LIGHTING)
        glPushMatrix()
        glTranslatef(*anchor)
        glMultMatrixf(self._billboard_matrix)
        glScalef(scale, scale, scale)
        draw_fn()
        glPopMatrix()
        glEnable(GL_LIGHTING)

    # -- difficulty gauge (Stage 3) -----------------------------------------

    def _gauge_displayed_value(self) -> float:
        if self._gauge_to is None:
            return float(MIN_DIFFICULTY)
        elapsed = time.time() - self._gauge_transition_start
        t = _smoothstep(min(1.0, elapsed / GAUGE_TRANSITION_SECONDS))
        return _lerp(float(self._gauge_from), float(self._gauge_to), t)

    def _draw_difficulty_gauge(self) -> None:
        value = self._gauge_displayed_value()
        frac = (value - MIN_DIFFICULTY) / (MAX_DIFFICULTY - MIN_DIFFICULTY)
        frac = max(0.0, min(1.0, frac))
        x, y0, z = GAUGE_ANCHOR

        glColor3f(*GAUGE_FRAME_COLOR)
        self._draw_box_at((x, y0 + GAUGE_MAX_HEIGHT / 2, z), (GAUGE_WIDTH, GAUGE_MAX_HEIGHT, GAUGE_DEPTH))

        fill_h = max(0.012, GAUGE_MAX_HEIGHT * frac)
        glColor3f(*GAUGE_FILL_COLOR)
        # Very slightly larger than the frame in all dimensions (not just
        # taller) so its faces are never exactly coplanar with the frame's
        # -- the same anti-z-fighting trick used elsewhere in this file.
        self._draw_box_at((x, y0 + fill_h / 2, z), (GAUGE_WIDTH * 1.04, fill_h, GAUGE_DEPTH * 1.04))

    # -- HUD (Stage 4) -------------------------------------------------------

    def _draw_hud(self, env, fb_width: int, fb_height: int, mood_color) -> None:
        """A bottom bar (per the brief: "does not obscure the classroom
        scene ... not overlapping the girl/desk") showing the last action
        taken, current engagement state, difficulty, reward, and a live
        engagement-over-time graph. All text is real, readable words (see
        _FONT_5X7) -- not icons -- since that's specifically what the brief
        asks the HUD for, unlike the in-scene cues."""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, fb_width, 0, fb_height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Sizes below are "design pixels" at a reference window height
        # (WINDOW_HEIGHT), scaled by ui_scale -- see the equivalent note in
        # the old path-renderer's HUD: without this, a retina framebuffer
        # (2x the logical window size) renders everything at half the
        # intended size.
        ui_scale = fb_height / WINDOW_HEIGHT
        panel_h = fb_height * HUD_HEIGHT_FRACTION

        glColor4f(HUD_BG_COLOR[0], HUD_BG_COLOR[1], HUD_BG_COLOR[2], HUD_BG_ALPHA)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(fb_width, 0)
        glVertex2f(fb_width, panel_h)
        glVertex2f(0, panel_h)
        glEnd()

        margin = 24 * ui_scale
        label_size = 10 * ui_scale
        value_size = 17 * ui_scale
        value_y = panel_h - margin - value_size
        label_y = value_y - label_size - 8 * ui_scale

        # Field columns are sized from actual text width (max of label/value,
        # both measured via `_text_width`), not guessed fixed gaps -- a
        # fixed gap silently overlaps as soon as any field's content (e.g. a
        # long action name) exceeds what was guessed, which is exactly what
        # happened here with "DIFFICULTY"/"REWARD" during initial testing.
        field_gap = 40 * ui_scale
        x = margin

        # Action
        action = self._action_cue_action
        action_text = ACTION_NAMES[action].replace("_", " ") if action is not None else "-"
        _draw_text_2d(x, label_y, label_size, "ACTION", HUD_LABEL_COLOR)
        _draw_text_2d(x, value_y, value_size, action_text, HUD_TEXT_COLOR)
        x += max(_text_width(label_size, "ACTION"), _text_width(value_size, action_text)) + field_gap

        # Engagement state (label + a dot matching the indicator orb's tweened color)
        state_text = env.engagement_state.name
        dot_offset = 22 * ui_scale
        _draw_text_2d(x, label_y, label_size, "STATE", HUD_LABEL_COLOR)
        glColor3f(*mood_color)
        _draw_filled_circle(x + 6 * ui_scale, value_y + value_size * 0.32, 6 * ui_scale, 12)
        _draw_text_2d(x + dot_offset, value_y, value_size, state_text, HUD_TEXT_COLOR)
        x += max(_text_width(label_size, "STATE"), dot_offset + _text_width(value_size, state_text)) + field_gap

        # Difficulty
        diff_text = str(env.difficulty)
        _draw_text_2d(x, label_y, label_size, "DIFFICULTY", HUD_LABEL_COLOR)
        _draw_text_2d(x, value_y, value_size, diff_text, HUD_TEXT_COLOR)
        x += max(_text_width(label_size, "DIFFICULTY"), _text_width(value_size, diff_text)) + field_gap

        # Reward
        reward_text = f"{env.last_reward:+.2f}"
        _draw_text_2d(x, label_y, label_size, "REWARD", HUD_LABEL_COLOR)
        _draw_text_2d(x, value_y, value_size, reward_text, HUD_TEXT_COLOR)
        x += max(_text_width(label_size, "REWARD"), _text_width(value_size, reward_text)) + field_gap

        # Engagement-over-time graph fills the remaining width.
        graph_label_h = label_size + 8 * ui_scale
        _draw_text_2d(x, panel_h - margin - label_size, label_size, "ENGAGEMENT OVER TIME", HUD_LABEL_COLOR)
        graph_w = fb_width - x - margin
        graph_h = panel_h - 2 * margin - graph_label_h
        _draw_engagement_graph(x, margin, graph_w, graph_h, self.history)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    # -- room shell -----------------------------------------------------

    def _draw_room(self) -> None:
        glColor3f(*FLOOR_COLOR)
        glBegin(GL_QUADS)
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(-ROOM_HALF_X, 0.0, -ROOM_HALF_Z)
        glVertex3f(ROOM_HALF_X, 0.0, -ROOM_HALF_Z)
        glVertex3f(ROOM_HALF_X, 0.0, ROOM_HALF_Z)
        glVertex3f(-ROOM_HALF_X, 0.0, ROOM_HALF_Z)
        glEnd()

        # Back wall (z = -ROOM_HALF_Z), facing +z into the room.
        glColor3f(*WALL_COLOR)
        glBegin(GL_QUADS)
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(-ROOM_HALF_X, 0.0, -ROOM_HALF_Z)
        glVertex3f(-ROOM_HALF_X, WALL_HEIGHT, -ROOM_HALF_Z)
        glVertex3f(ROOM_HALF_X, WALL_HEIGHT, -ROOM_HALF_Z)
        glVertex3f(ROOM_HALF_X, 0.0, -ROOM_HALF_Z)
        glEnd()

        # Left wall (x = -ROOM_HALF_X), facing +x into the room.
        glBegin(GL_QUADS)
        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(-ROOM_HALF_X, 0.0, -ROOM_HALF_Z)
        glVertex3f(-ROOM_HALF_X, 0.0, ROOM_HALF_Z)
        glVertex3f(-ROOM_HALF_X, WALL_HEIGHT, ROOM_HALF_Z)
        glVertex3f(-ROOM_HALF_X, WALL_HEIGHT, -ROOM_HALF_Z)
        glEnd()

        # No ceiling mesh: the camera is deliberately elevated *above* the
        # wall height for the Sims-style top-down framing (see module
        # docstring), so a ceiling plane would sit between the camera and
        # the room's interior and occlude everything -- the "open-top
        # dollhouse" look is intentional here, not a gap to fill in. The
        # neutral-dark clear color (see _init_gl_state) is what keeps the
        # area above the walls from reading as sky/outdoors.

        # A thin baseboard strip along both visible walls, for a touch of
        # grounded architectural detail.
        glColor3f(*BASEBOARD_COLOR)
        glBegin(GL_QUADS)
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(-ROOM_HALF_X, 0.0, -ROOM_HALF_Z + 0.01)
        glVertex3f(-ROOM_HALF_X, 0.12, -ROOM_HALF_Z + 0.01)
        glVertex3f(ROOM_HALF_X, 0.12, -ROOM_HALF_Z + 0.01)
        glVertex3f(ROOM_HALF_X, 0.0, -ROOM_HALF_Z + 0.01)
        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(-ROOM_HALF_X + 0.01, 0.0, -ROOM_HALF_Z)
        glVertex3f(-ROOM_HALF_X + 0.01, 0.0, ROOM_HALF_Z)
        glVertex3f(-ROOM_HALF_X + 0.01, 0.12, ROOM_HALF_Z)
        glVertex3f(-ROOM_HALF_X + 0.01, 0.12, -ROOM_HALF_Z)
        glEnd()

    def _draw_rug(self) -> None:
        cx, cz = RUG_CENTER
        hw, hd = RUG_SIZE[0] / 2, RUG_SIZE[1] / 2
        glColor3f(*RUG_COLOR)
        glBegin(GL_QUADS)
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(cx - hw, 0.008, cz - hd)
        glVertex3f(cx + hw, 0.008, cz - hd)
        glVertex3f(cx + hw, 0.008, cz + hd)
        glVertex3f(cx - hw, 0.008, cz + hd)
        glEnd()

    def _draw_window(self) -> None:
        cx, cy, cz = WINDOW_CENTER
        hw, hh = WINDOW_SIZE[0] / 2, WINDOW_SIZE[1] / 2
        # Sky pane
        glColor3f(*WINDOW_SKY_COLOR)
        glBegin(GL_QUADS)
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(cx - hw, cy - hh, cz)
        glVertex3f(cx - hw, cy + hh, cz)
        glVertex3f(cx + hw, cy + hh, cz)
        glVertex3f(cx + hw, cy - hh, cz)
        glEnd()
        # Simple cross frame + outer border, drawn as thin boxes
        glColor3f(*WINDOW_FRAME_COLOR)
        self._draw_box_at((cx, cy, cz - 0.01), (WINDOW_SIZE[0] + 0.1, WINDOW_SIZE[1] + 0.1, 0.03))
        glColor3f(*WINDOW_SKY_COLOR)  # redraw pane on top of the border box's front face footprint
        glBegin(GL_QUADS)
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(cx - hw, cy - hh, cz + 0.005)
        glVertex3f(cx - hw, cy + hh, cz + 0.005)
        glVertex3f(cx + hw, cy + hh, cz + 0.005)
        glVertex3f(cx + hw, cy - hh, cz + 0.005)
        glEnd()
        glColor3f(*WINDOW_FRAME_COLOR)
        glBegin(GL_QUADS)  # vertical mullion
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(cx - 0.02, cy - hh, cz + 0.01)
        glVertex3f(cx - 0.02, cy + hh, cz + 0.01)
        glVertex3f(cx + 0.02, cy + hh, cz + 0.01)
        glVertex3f(cx + 0.02, cy - hh, cz + 0.01)
        glEnd()
        glBegin(GL_QUADS)  # horizontal mullion
        glVertex3f(cx - hw, cy - 0.02, cz + 0.01)
        glVertex3f(cx - hw, cy + 0.02, cz + 0.01)
        glVertex3f(cx + hw, cy + 0.02, cz + 0.01)
        glVertex3f(cx + hw, cy - 0.02, cz + 0.01)
        glEnd()

    # -- furniture --------------------------------------------------------

    def _draw_desk(self) -> None:
        x, z = DESK_POS
        w, h, d = DESK_SIZE
        glColor3f(*DESK_COLOR)
        self._draw_box_at((x, h / 2, z), (w, h, d))

    def _draw_tablet(self, content_index: int, glow_intensity: float) -> None:
        x, y, z = TABLET_POS
        w, h, d = TABLET_SIZE
        glColor3f(*TABLET_BODY_COLOR)
        self._draw_box_at((x, y + h / 2, z), (w, h, d))

        screen_color = TABLET_SCREEN_PALETTE[content_index % len(TABLET_SCREEN_PALETTE)]
        glColor3f(*screen_color)
        glBegin(GL_QUADS)
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(x - w / 2 + 0.02, y + h + 0.002, z - d / 2 + 0.02)
        glVertex3f(x - w / 2 + 0.02, y + h + 0.002, z + d / 2 - 0.02)
        glVertex3f(x + w / 2 - 0.02, y + h + 0.002, z + d / 2 - 0.02)
        glVertex3f(x + w / 2 - 0.02, y + h + 0.002, z - d / 2 + 0.02)
        glEnd()

        if glow_intensity > 0.001:
            # change_task_type just fired: a brief glowing outline around
            # the screen's perimeter, so the content swap (otherwise an
            # instant change) doesn't go unnoticed. Deliberately an outline
            # loop, not a solid overlay -- an early version used a solid
            # translucent box and it washed out the very screen-color change
            # it was supposed to draw attention to, since the camera looks
            # down at the tablet almost face-on.
            glDisable(GL_LIGHTING)
            glMaterialfv(GL_FRONT, GL_EMISSION, (0.9, 0.9, 0.95, 1.0))
            glColor4f(1.0, 1.0, 1.0, glow_intensity)
            glLineWidth(5.0)
            glBegin(GL_LINE_LOOP)
            glVertex3f(x - w / 2 - 0.025, y + h + 0.004, z - d / 2 - 0.025)
            glVertex3f(x - w / 2 - 0.025, y + h + 0.004, z + d / 2 + 0.025)
            glVertex3f(x + w / 2 + 0.025, y + h + 0.004, z + d / 2 + 0.025)
            glVertex3f(x + w / 2 + 0.025, y + h + 0.004, z - d / 2 - 0.025)
            glEnd()
            glMaterialfv(GL_FRONT, GL_EMISSION, (0.0, 0.0, 0.0, 1.0))
            glEnable(GL_LIGHTING)

    def _draw_chair(self) -> None:
        x, z = CHAIR_POS
        glColor3f(*CHAIR_COLOR)
        self._draw_box_at((x, CHAIR_SEAT_Y / 2, z), (0.42, CHAIR_SEAT_Y, 0.42))
        # Backrest behind the girl (away from the desk, which is in the -z
        # direction), not between her and the desk.
        self._draw_box_at((x, CHAIR_SEAT_Y + 0.28, z + 0.19), (0.42, 0.56, 0.05))

    def _draw_blackboard(self, category_index: int, problem_indices: dict[str, int], glow_intensity: float) -> None:
        x, y, z = BLACKBOARD_CENTER
        h, w = BLACKBOARD_SIZE
        m = BLACKBOARD_FRAME_MARGIN
        glColor3f(*BLACKBOARD_FRAME_COLOR)
        self._draw_box_at((x - 0.015, y, z), (0.03, h + 2 * m, w + 2 * m))
        glColor3f(*BLACKBOARD_COLOR)
        glBegin(GL_QUADS)
        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(x + 0.005, y - h / 2, z - w / 2)
        glVertex3f(x + 0.005, y - h / 2, z + w / 2)
        glVertex3f(x + 0.005, y + h / 2, z + w / 2)
        glVertex3f(x + 0.005, y + h / 2, z - w / 2)
        glEnd()

        # Problem text, drawn directly on the board face -- the board is
        # axis-aligned (facing +x), so local (y, z) map straight onto it
        # without needing a billboard transform.
        category = BLACKBOARD_CATEGORIES[category_index]
        problem_list = BLACKBOARD_PROBLEMS[category]
        problem_text = problem_list[problem_indices[category] % len(problem_list)]
        glDisable(GL_LIGHTING)
        _draw_text_on_board(x + 0.012, y, z, BLACKBOARD_TEXT_CHAR_SIZE, problem_text, BLACKBOARD_TEXT_COLOR)
        glEnable(GL_LIGHTING)

        if glow_intensity > 0.001:
            # advance_topic just fired: a glowing outline around the board's
            # perimeter (see _draw_tablet's matching comment for why this is
            # an outline, not a solid overlay -- avoids washing out the icon).
            glDisable(GL_LIGHTING)
            glMaterialfv(GL_FRONT, GL_EMISSION, (0.55, 0.7, 0.9, 1.0))
            glColor4f(0.75, 0.88, 1.0, glow_intensity)
            glLineWidth(5.0)
            hh, hw = h / 2 + m + 0.02, w / 2 + m + 0.02
            glBegin(GL_LINE_LOOP)
            glVertex3f(x + 0.02, y - hh, z - hw)
            glVertex3f(x + 0.02, y - hh, z + hw)
            glVertex3f(x + 0.02, y + hh, z + hw)
            glVertex3f(x + 0.02, y + hh, z - hw)
            glEnd()
            glMaterialfv(GL_FRONT, GL_EMISSION, (0.0, 0.0, 0.0, 1.0))
            glEnable(GL_LIGHTING)

    def _draw_plant(self) -> None:
        x, z = PLANT_POS
        glColor3f(*POT_COLOR)
        self._draw_box_at((x, 0.14, z), (0.26, 0.28, 0.26))
        glColor3f(*LEAF_COLOR)
        for i, (dx, dz, dy) in enumerate([(0.0, 0.0, 0.5), (0.1, 0.08, 0.42), (-0.1, 0.06, 0.4), (0.02, -0.1, 0.46)]):
            glPushMatrix()
            glTranslatef(x + dx, 0.28 + dy, z + dz)
            _draw_sphere(0.14 - i * 0.012, 8, 6)
            glPopMatrix()

    # -- seated avatar ----------------------------------------------------

    def _draw_girl_seated(self, indicator_color, posture) -> None:
        """`indicator_color` is the tweened RGB for the state orb;
        `posture` is the tweened dict from `_current_mood` (torso_pitch,
        head_pitch, head_roll, idle_amp/freq/axis)."""
        clock = time.time() - self._t0
        idle_offset = posture["idle_amp"] * math.sin(clock * posture["idle_freq"])
        head_pitch = posture["head_pitch"] + (idle_offset if posture["idle_axis"] == "pitch" else 0.0)
        head_roll = posture["head_roll"] + (idle_offset if posture["idle_axis"] == "roll" else 0.0)

        x, z = GIRL_POS
        glPushMatrix()
        glTranslatef(x, CHAIR_SEAT_Y, z)
        glRotatef(GIRL_FACING_DEG, 0.0, 1.0, 0.0)

        # Legs: simple two-segment bent pose (thigh roughly horizontal,
        # extending forward toward the desk; shin down to the floor), mostly
        # occluded by the desk from the fixed camera -- kept simple rather
        # than fully modeled, and not affected by mood posture.
        glColor3f(*CLOTHES_COLOR)
        for side in (-1, 1):
            glPushMatrix()
            glTranslatef(side * 0.14, 0.0, 0.0)
            glPushMatrix()
            # y=+0.07 (clear of the chair seat's top surface, which sits
            # exactly at local y=0) -- keeping well above 0 avoids the
            # thigh box intersecting the seat box and z-fighting with it.
            glTranslatef(0.0, 0.07, -0.12)
            glRotatef(70.0, 1.0, 0.0, 0.0)
            glScalef(0.14, 0.36, 0.14)
            _draw_unit_box()
            glPopMatrix()
            glPushMatrix()
            glTranslatef(0.0, -0.42, -0.30)
            glScalef(0.14, 0.42, 0.14)
            _draw_unit_box()
            glPopMatrix()
            glPopMatrix()

        # Torso + arms + head, pitched forward/back at the hip -- this is
        # the "slumped" <-> "upright" mood cue.
        glPushMatrix()
        glRotatef(posture["torso_pitch"], 1.0, 0.0, 0.0)

        glColor3f(*CLOTHES_COLOR)
        glPushMatrix()
        glTranslatef(0.0, 0.35, 0.0)
        glScalef(0.5, 0.7, 0.32)
        _draw_unit_box()
        glPopMatrix()

        # Arms: resting toward the desk/tablet in front of her.
        glColor3f(*SKIN_COLOR)
        for side in (-1, 1):
            glPushMatrix()
            glTranslatef(side * 0.32, 0.65, 0.0)
            glRotatef(60.0, 1.0, 0.0, 0.0)
            glTranslatef(0.0, -0.22, 0.0)
            glScalef(0.12, 0.44, 0.12)
            _draw_unit_box()
            glPopMatrix()

        # Head group: pivots at the neck, on top of the torso lean above --
        # this is the "head tilt/nod/wobble" mood cue. Head/ponytail/orb
        # offsets below are all relative to NECK_Y rather than the seat root.
        glPushMatrix()
        glTranslatef(0.0, NECK_Y, 0.0)
        glRotatef(head_pitch, 1.0, 0.0, 0.0)
        glRotatef(head_roll, 0.0, 0.0, 1.0)

        glColor3f(*SKIN_COLOR)
        glPushMatrix()
        glTranslatef(0.0, 0.17, 0.0)  # head center: 0.95 - NECK_Y
        _draw_sphere(0.28, 12, 10)
        glPopMatrix()

        glColor3f(*HAIR_COLOR)
        for i in range(4):
            t = i / 3.0
            glPushMatrix()
            glTranslatef(0.0, 0.32 - 0.28 * t, 0.22 + 0.22 * t)  # 1.10 - NECK_Y, then tapering
            _draw_sphere(0.12 * (1.0 - 0.6 * t), 8, 6)
            glPopMatrix()

        # Engagement-state indicator orb: a small, glowing marker above her
        # head -- deliberately not a full-body tint, per the brief.
        glDisable(GL_LIGHTING)
        glMaterialfv(GL_FRONT, GL_EMISSION, (indicator_color[0] * 0.6, indicator_color[1] * 0.6, indicator_color[2] * 0.6, 1.0))
        glColor3f(*indicator_color)
        glPushMatrix()
        glTranslatef(0.0, 0.62 + 0.02 * math.sin(clock * 2.5), 0.0)
        _draw_sphere(0.075, 10, 8)
        glPopMatrix()
        glMaterialfv(GL_FRONT, GL_EMISSION, (0.0, 0.0, 0.0, 1.0))
        glEnable(GL_LIGHTING)

        glPopMatrix()  # end head group
        glPopMatrix()  # end torso group

        glPopMatrix()  # end root

    # -- small helpers -------------------------------------------------------

    @staticmethod
    def _draw_box_at(center, size) -> None:
        glPushMatrix()
        glTranslatef(center[0], center[1], center[2])
        glScalef(size[0], size[1], size[2])
        _draw_unit_box()
        glPopMatrix()

    def _read_pixels(self):
        import numpy as np

        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        raw = glReadPixels(0, 0, fb_width, fb_height, GL_RGB, GL_UNSIGNED_BYTE)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(fb_height, fb_width, 3)
        return np.flipud(arr)  # OpenGL's origin is bottom-left; image arrays expect top-left

    def _blank_frame(self):
        import numpy as np

        return np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)


# --------------------------------------------------------------------------
# Geometry primitives (no GLU dependency -- see module docstring)
# --------------------------------------------------------------------------


def _draw_unit_box() -> None:
    """A 1x1x1 box centered on the origin; callers use glScalef/glTranslatef
    to size and position it."""
    x = y = z = 0.5
    faces = [
        ((0, 0, 1), [(-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]),
        ((0, 0, -1), [(x, -y, -z), (-x, -y, -z), (-x, y, -z), (x, y, -z)]),
        ((0, 1, 0), [(-x, y, z), (x, y, z), (x, y, -z), (-x, y, -z)]),
        ((0, -1, 0), [(-x, -y, -z), (x, -y, -z), (x, -y, z), (-x, -y, z)]),
        ((1, 0, 0), [(x, -y, z), (x, -y, -z), (x, y, -z), (x, y, z)]),
        ((-1, 0, 0), [(-x, -y, -z), (-x, -y, z), (-x, y, z), (-x, y, -z)]),
    ]
    glBegin(GL_QUADS)
    for normal, verts in faces:
        glNormal3f(*normal)
        for v in verts:
            glVertex3f(*v)
    glEnd()


def _draw_sphere(radius: float, slices: int = 10, stacks: int = 8) -> None:
    """A low-poly UV sphere, reimplemented by hand to avoid a GLU dependency
    (gluSphere) -- see module docstring."""
    for i in range(stacks):
        lat0 = math.pi * (-0.5 + i / stacks)
        lat1 = math.pi * (-0.5 + (i + 1) / stacks)
        z0, zr0 = math.sin(lat0), math.cos(lat0)
        z1, zr1 = math.sin(lat1), math.cos(lat1)
        glBegin(GL_TRIANGLE_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * j / slices
            dx, dy = math.cos(lng), math.sin(lng)
            glNormal3f(dx * zr0, dy * zr0, z0)
            glVertex3f(radius * dx * zr0, radius * dy * zr0, radius * z0)
            glNormal3f(dx * zr1, dy * zr1, z1)
            glVertex3f(radius * dx * zr1, radius * dy * zr1, radius * z1)
        glEnd()


def _draw_filled_circle(cx: float, cy: float, r: float, segments: int) -> None:
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(cx, cy)
    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        glVertex2f(cx + r * math.cos(angle), cy + r * math.sin(angle))
    glEnd()


# --------------------------------------------------------------------------
# Action-cue icon shapes (Stage 3)
#
# Each of these draws in a local -1..1-ish 2D space (z=0) and is meant to be
# invoked either inside a billboard transform (see _draw_billboard_icon,
# always facing the camera -- used for the four popup cues) or directly at
# a fixed world x (used for the blackboard topic icon, which lies on an
# axis-aligned surface and so needs no billboarding). `intensity` (0..1)
# both fades the icon in/out (via alpha) and is baked into the color calls
# directly rather than threaded through every glVertex call.
# --------------------------------------------------------------------------


def _draw_lightbulb_icon(intensity: float) -> None:
    """offer_hint."""
    glColor4f(1.0, 0.92, 0.55, intensity)
    _draw_filled_circle(0.0, 0.08, 0.32, 14)
    glColor4f(0.55, 0.52, 0.50, intensity)
    glBegin(GL_QUADS)
    glVertex2f(-0.10, -0.28)
    glVertex2f(0.10, -0.28)
    glVertex2f(0.10, -0.08)
    glVertex2f(-0.10, -0.08)
    glEnd()
    glColor4f(1.0, 0.92, 0.55, intensity * 0.9)
    glLineWidth(2.5)
    for angle_deg in (55, 90, 125):
        a = math.radians(angle_deg)
        glBegin(GL_LINES)
        glVertex2f(0.38 * math.cos(a), 0.08 + 0.38 * math.sin(a))
        glVertex2f(0.58 * math.cos(a), 0.08 + 0.58 * math.sin(a))
        glEnd()


def _draw_starburst_icon(intensity: float) -> None:
    """introduce_challenge."""
    glColor4f(1.0, 0.75, 0.15, intensity)
    pts = []
    for i in range(10):
        r = 0.45 if i % 2 == 0 else 0.18
        a = math.pi / 2 + i * math.pi / 5
        pts.append((r * math.cos(a), r * math.sin(a)))
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(0.0, 0.0)
    for p in pts + [pts[0]]:
        glVertex2f(*p)
    glEnd()


def _draw_rewind_icon(intensity: float) -> None:
    """repeat_simplified."""
    glColor4f(0.55, 0.80, 0.95, intensity)
    glLineWidth(3.0)
    glBegin(GL_LINE_STRIP)
    for i in range(11):
        a = math.radians(40 + i * 280 / 10)
        glVertex2f(0.35 * math.cos(a), 0.35 * math.sin(a))
    glEnd()
    a0 = math.radians(40)
    tip = (0.35 * math.cos(a0), 0.35 * math.sin(a0))
    glBegin(GL_TRIANGLES)
    glVertex2f(tip[0] + 0.14, tip[1] + 0.04)
    glVertex2f(tip[0] - 0.05, tip[1] + 0.17)
    glVertex2f(tip[0] - 0.02, tip[1] - 0.11)
    glEnd()


def _draw_4_point_sparkle(cx: float, cy: float, s: float, intensity: float) -> None:
    glColor4f(1.0, 0.85, 0.45, intensity)
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(cx, cy)
    pts = [
        (cx, cy + s), (cx + s * 0.28, cy + s * 0.28),
        (cx + s, cy), (cx + s * 0.28, cy - s * 0.28),
        (cx, cy - s), (cx - s * 0.28, cy - s * 0.28),
        (cx - s, cy), (cx - s * 0.28, cy + s * 0.28),
        (cx, cy + s),
    ]
    for p in pts:
        glVertex2f(*p)
    glEnd()


def _draw_sparkle_icon(intensity: float) -> None:
    """add_encouragement: a small cluster of sparkles near the girl."""
    for dx, dy, s in [(0.0, 0.0, 0.20), (0.30, 0.14, 0.11), (-0.27, 0.20, 0.10), (0.08, -0.24, 0.09)]:
        _draw_4_point_sparkle(dx, dy, s, intensity)


def _draw_text_on_board(x: float, y_center: float, z_center: float, char_size: float, text: str, color) -> None:
    """advance_topic / change_task_type: real problem text drawn directly on
    the blackboard face, using the same 5x7 bitmap font as the HUD (see
    _FONT_5X7 / _draw_text_2d) but laid out in the board's local (y, z)
    plane instead of 2D screen space -- the board is axis-aligned (facing
    +/-x), so local (y, z) map directly onto it without a billboard
    transform. Text is laid out in a local u ("screen-right") coordinate
    and converted to world z via z = z_center - u: for this fixed camera
    (CAMERA_EYE/CAMERA_TARGET above), the view's right-vector has a
    *negative* z component, i.e. increasing world z reads as screen-LEFT
    here -- the opposite of _draw_text_2d's plain 2D ortho space. Laying
    the text out directly along +z (an earlier version of this function
    did that) draws the whole string mirrored. The glyph rows run top to
    bottom along -y, matching _draw_text_2d's row order. Centered on
    (y_center, z_center)."""
    glColor3f(*color)
    px = char_size / 5.0
    total_w = _text_width(char_size, text)
    start_u = -total_w / 2.0
    top_y = y_center + 3.5 * px  # vertically center the 7-row glyph block
    cursor_u = start_u
    for ch in text.upper():
        glyph = _FONT_5X7.get(ch)
        if glyph is not None:
            glBegin(GL_QUADS)
            for row in range(7):
                for col in range(5):
                    if glyph[row][col] == "#":
                        u0 = cursor_u + col * px
                        u1 = u0 + px
                        gy = top_y - row * px
                        gz0 = z_center - u0
                        gz1 = z_center - u1
                        glVertex3f(x, gy, gz0)
                        glVertex3f(x, gy, gz1)
                        glVertex3f(x, gy - px, gz1)
                        glVertex3f(x, gy - px, gz0)
            glEnd()
        cursor_u += char_size + px


# --------------------------------------------------------------------------
# HUD text (Stage 4): a small self-authored 5x7 bitmap font, drawn as
# filled quads. No system font / GLUT dependency -- see module docstring's
# general design philosophy of not requiring anything beyond what `uv sync`
# installs. Only the characters actually needed by HUD labels are defined
# (uppercase A-Z, 0-9, space, and a few punctuation marks); an undefined
# character is skipped (rendered as blank) rather than raising, so a typo
# in a label degrades gracefully instead of crashing the renderer.
# --------------------------------------------------------------------------

_FONT_5X7: dict[str, tuple[str, ...]] = {
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".####", "#....", "#....", "#....", "#....", "#....", ".####"),
    "D": ("####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "G": (".####", "#....", "#....", "#.###", "#...#", "#...#", ".####"),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"),
    "J": ("..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."),
    "K": ("#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#...#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "Q": (".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "V": ("#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": ("#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"),
    "X": ("#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    "Y": ("#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."),
    "Z": ("#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"),
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", "#####"),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": (".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": ("..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."),
    " ": (".....", ".....", ".....", ".....", ".....", ".....", "....."),
    ":": (".....", "..#..", ".....", ".....", ".....", "..#..", "....."),
    ".": (".....", ".....", ".....", ".....", ".....", ".##..", ".##.."),
    "+": (".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."),
    "-": (".....", ".....", ".....", "#####", ".....", ".....", "....."),
    "%": ("#...#", "....#", "...#.", "..#..", ".#...", "#....", "#...#"),
    "/": ("....#", "...#.", "..#..", "..#..", ".#...", ".#...", "#...."),
    "=": (".....", ".....", "#####", ".....", "#####", ".....", "....."),
    "?": (".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."),
    ",": (".....", ".....", ".....", ".....", ".....", "..##.", ".#..."),
    "_": (".....", ".....", ".....", ".....", ".....", ".....", "#####"),
}


def _draw_text_2d(x: float, y: float, char_size: float, text: str, color) -> None:
    """Draws `text` (upper-cased automatically) at (x, y) in whatever 2D
    ortho space is currently active, `char_size` tall per character. Meant
    for the HUD only (Stage 4) -- in-scene cues stay purely iconographic
    (see the action-cue functions above)."""
    glColor3f(*color)
    px = char_size / 5.0  # size of one "pixel" of the 5x7 grid
    cursor_x = x
    for ch in text.upper():
        glyph = _FONT_5X7.get(ch)
        if glyph is not None:
            glBegin(GL_QUADS)
            for row in range(7):
                for col in range(5):
                    if glyph[row][col] == "#":
                        gx = cursor_x + col * px
                        gy = y + (6 - row) * px  # row 0 is the top of the glyph
                        glVertex2f(gx, gy)
                        glVertex2f(gx + px, gy)
                        glVertex2f(gx + px, gy + px)
                        glVertex2f(gx, gy + px)
            glEnd()
        cursor_x += char_size + px  # advance one glyph width + 1px gap


def _text_width(char_size: float, text: str) -> float:
    """Pixel width `_draw_text_2d` would occupy for `text` -- used to
    right-align or center HUD fields."""
    if not text:
        return 0.0
    px = char_size / 5.0
    return len(text) * (char_size + px) - px


def _draw_engagement_graph(x: float, y: float, w: float, h: float, history) -> None:
    """A live line graph of engagement state over the episode so far
    (0=FRUSTRATED, 1=BORED, 2=CONFUSED, 3=ENGAGED -- matching
    EngagementState's own integer order), with each row tinted by that
    state's color for a self-explanatory axis instead of a text legend."""
    if w <= 0 or h <= 0:
        return
    band_h = h / 4.0
    for state_value, color in (
        (0, STATE_COLOR[EngagementState.FRUSTRATED]),
        (1, STATE_COLOR[EngagementState.BORED]),
        (2, STATE_COLOR[EngagementState.CONFUSED]),
        (3, STATE_COLOR[EngagementState.ENGAGED]),
    ):
        glColor4f(color[0], color[1], color[2], GRAPH_BAND_ALPHA)
        glBegin(GL_QUADS)
        glVertex2f(x, y + state_value * band_h)
        glVertex2f(x + w, y + state_value * band_h)
        glVertex2f(x + w, y + (state_value + 1) * band_h)
        glVertex2f(x, y + (state_value + 1) * band_h)
        glEnd()

    if len(history) < 2:
        return
    window = history[-40:]  # show at most the most recent 40 steps
    n = len(window)
    glLineWidth(2.5)
    glBegin(GL_LINE_STRIP)
    for i, rec in enumerate(window):
        px = x + w * (i / max(1, n - 1))
        py = y + (int(rec.state) + 0.5) * band_h
        glColor3f(*STATE_COLOR[rec.state])
        glVertex2f(px, py)
    glEnd()

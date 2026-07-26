"""Manual visual smoke test for the PyOpenGL renderer.

Not part of the pytest suite (rendering requires a real display/GL context,
which CI/headless environments don't have -- see environment/rendering.py's
module docstring). Run by hand and inspect the saved PNGs:

    uv run python tests/render_smoke_test.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from environment.custom_env import AdaptLearnEnv

OUT_DIR = Path(__file__).parent / "_render_smoke_output"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    env = AdaptLearnEnv(render_mode="rgb_array")
    env.reset(seed=0)

    # A deliberately varied action sequence, so distinct visual features
    # (incline, fork, bridge, glow, flow zone) each get exercised at least
    # once across the saved frames.
    actions = [0, 0, 3, 4, 2, 5, 0, 0, 1, 6, 0, 0, 4, 2, 0, 3, 0, 0, 0, 0, 0]

    frame = env.render()
    Image.fromarray(frame).save(OUT_DIR / "frame_000_reset.png")
    print(f"saved frame_000_reset.png {frame.shape}")

    for i, action in enumerate(actions, start=1):
        env.step(action)
        frame = env.render()
        Image.fromarray(frame).save(OUT_DIR / f"frame_{i:03d}.png")
        print(f"saved frame_{i:03d}.png action={action} shape={frame.shape}")

    env.close()
    print(f"\nDone. Inspect PNGs in {OUT_DIR}")


if __name__ == "__main__":
    main()

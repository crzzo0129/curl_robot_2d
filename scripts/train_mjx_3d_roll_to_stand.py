"""Primitive-only dynamic Roll to Stand PPO entry; mesh is never trained here."""

import sys
from scripts.train_mjx_3d_transition_ppo import main as transition_main


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    # Defaults can be overridden, but the underlying task rejects mesh training.
    transition_main(["--geometry", "rollingquad_2_primitive",
                     "--stage", "brake_full", "--dynamic-roll-to-stand",
                     "--physics-profile", "cg12",
                     "--initial-policy-std", "0.2",
                     "--out", "results/mjx_3d_roll_to_stand", *args])


if __name__ == "__main__":
    main()

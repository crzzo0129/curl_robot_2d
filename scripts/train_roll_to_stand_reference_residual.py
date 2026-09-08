"""Train residual corrections around the successful +90deg handcrafted transition."""

import sys
from scripts.train_mjx_3d_transition_ppo import main as transition_main


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    transition_main([
        "--geometry", "rollingquad_2_abd10_no_self_collision",
        "--stage", "brake_full", "--dynamic-roll-to-stand",
        "--handcrafted-reference-residual", "--stand-abduction-zero",
        "--physics-profile", "accurate", "--initial-policy-std", "0.05",
        "--out", "results/roll_to_stand_reference_residual", *args,
    ])


if __name__ == "__main__":
    main()

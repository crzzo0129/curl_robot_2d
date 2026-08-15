import argparse
import json
import math
from pathlib import Path

from tensorboardX import SummaryWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    output_dir = args.out or args.result_dir / "tensorboard"
    writer = SummaryWriter(str(output_dir))

    converted = 0

    for filename in ("metrics_history.json", "reward_history.json"):
        path = args.result_dir / filename
        if not path.exists():
            print(f"skip missing: {path}")
            continue

        rows = json.loads(path.read_text(encoding="utf-8"))

        for row in rows:
            step = int(row["step"])

            for name, value in row.items():
                if name == "step" or not isinstance(value, (int, float)):
                    continue
                if not math.isfinite(value):
                    continue

                writer.add_scalar(name, value, step)
                converted += 1

    writer.flush()
    writer.close()

    print(f"converted_scalars={converted}")
    print(f"tensorboard_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
from pathlib import Path

from curl_robot_2d.model import write_mjcf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_2d.xml"


if __name__ == "__main__":
    path = write_mjcf(MODEL_PATH)
    print(path)

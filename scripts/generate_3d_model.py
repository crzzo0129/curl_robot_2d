from pathlib import Path

from curl_robot_2d.model_3d import write_mjcf_3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_3d.xml"


if __name__ == "__main__":
    path = write_mjcf_3d(MODEL_PATH)
    print(path)

from pathlib import Path

from curl_robot_2d.model import write_mjcf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_2d.xml"
NO_SELF_COLLISION_MODEL_PATH = (
    PROJECT_ROOT / "assets" / "curl_robot_2d_no_self_collision.xml"
)


if __name__ == "__main__":
    path = write_mjcf(MODEL_PATH)
    print(path)
    no_self_collision_path = write_mjcf(
        NO_SELF_COLLISION_MODEL_PATH,
        enable_self_collision=False,
    )
    print(no_self_collision_path)

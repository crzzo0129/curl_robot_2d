from pathlib import Path

from curl_robot_2d.model import write_mjcf
from curl_robot_2d.parameters import REAL_GEOMETRY_PARAMETERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_2d.xml"
NO_SELF_COLLISION_MODEL_PATH = (
    PROJECT_ROOT / "assets" / "curl_robot_2d_no_self_collision.xml"
)
REAL_GEOMETRY_MODEL_PATH = (
    PROJECT_ROOT / "assets" / "curl_robot_2d_real_geometry.xml"
)


if __name__ == "__main__":
    path = write_mjcf(MODEL_PATH)
    print(path)
    no_self_collision_path = write_mjcf(
        NO_SELF_COLLISION_MODEL_PATH,
        enable_self_collision=False,
    )
    print(no_self_collision_path)
    real_geometry_path = write_mjcf(
        REAL_GEOMETRY_MODEL_PATH,
        parameters=REAL_GEOMETRY_PARAMETERS,
        include_rolling_shell=True,
        detailed_structure=True,
    )
    print(real_geometry_path)

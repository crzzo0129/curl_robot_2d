from pathlib import Path

from curl_robot_2d.model_3d import write_mjcf_3d
from curl_robot_2d.parameters import REAL_GEOMETRY_PARAMETERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_3d.xml"
REAL_GEOMETRY_MODEL_PATH = (
    PROJECT_ROOT / "assets" / "curl_robot_3d_real_geometry.xml"
)


if __name__ == "__main__":
    path = write_mjcf_3d(MODEL_PATH)
    print(path)
    real_geometry_path = write_mjcf_3d(
        REAL_GEOMETRY_MODEL_PATH,
        REAL_GEOMETRY_PARAMETERS,
        detailed_structure=True,
    )
    print(real_geometry_path)

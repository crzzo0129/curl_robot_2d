"""Cheap CPU reference diagnostic before MJX collection; never loads mesh collisions."""

import json
from pathlib import Path
from curl_robot_2d_mjx.environment_3d import (
    ROLLINGQUAD_2_PRIMITIVE_CEM_CONTROLLER, model_path_3d,
)
from scripts.view_3d_cem_reference import run


def main():
    path = model_path_3d("rollingquad_2_primitive")
    rows = []
    for profile in ("newton4", "reference", "cg12"):
        row = run(["--xml", str(path), "--controller", str(ROLLINGQUAD_2_PRIMITIVE_CEM_CONTROLLER),
                   "--geometry", "rollingquad_2", "--physics-profile", profile,
                   "--duration", "10", "--headless"])
        # The legacy CPU tool uses the rollingquad family name for kinematics.
        row.update(geometry="rollingquad_2_primitive", model_xml=str(path),
                   backend="cpu_mujoco", mjx_parity_verified=False)
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = Path("results/roll_to_stand_reference")
    out.mkdir(parents=True, exist_ok=True)
    (out / "cpu_reference_probe.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

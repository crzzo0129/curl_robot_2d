from __future__ import annotations

import math
import unittest
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d_mjx.terrain_3d import (
    SlopeTerrainConfig,
    column_centers_x,
    flat_hfield_data_3d,
    hfield_data_3d,
    hfield_data_flat_column_major,
    inject_hfield_into_mjcf,
    slope_terrain_config_from_task,
    terrain_height,
    terrain_height_array,
    validate_slope_terrain_config,
)
from curl_robot_2d_mjx.config_3d import Rolling3DConfig, validate_3d_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MESH_MODEL = (
    PROJECT_ROOT
    / "assets"
    / "rollingquad_description_2"
    / "mjcf"
    / "rollingquad.xml"
)


class SlopeTerrainProfileTest(unittest.TestCase):
    def setUp(self):
        self.config = SlopeTerrainConfig(
            slope_angle_deg=2.0,
            slope_start_distance_m=3.0,
            transition_length_m=0.5,
            slope_length_m=2.0,
        )

    def test_flat_config_yields_zero_height(self):
        flat = SlopeTerrainConfig(slope_angle_deg=0.0)
        x = np.linspace(-5.0, 5.0, 101)
        np.testing.assert_array_equal(
            terrain_height_array(x, flat), np.zeros_like(x)
        )

    def test_uphill_profile_reaches_total_rise_and_is_smooth(self):
        rise = math.tan(math.radians(2.0))
        config = self.config

        self.assertAlmostEqual(terrain_height(0.0, config), 0.0, places=9)
        self.assertAlmostEqual(
            terrain_height(config.slope_start_distance_m, config), 0.0, places=9
        )
        # Mid entry transition is half of a half: rise * L / 2 accumulated over
        # the full transition, so at the midpoint of the entry the height is
        # rise * L * smoothstep_integral(0.5).
        entry_mid = config.slope_start_distance_m + 0.5 * config.transition_length_m
        expected_entry_mid = rise * config.transition_length_m * (
            0.5**3 - 0.5 * 0.5**4
        )
        self.assertAlmostEqual(
            terrain_height(entry_mid, config), expected_entry_mid, places=9
        )
        # Top plateau.
        top = config.slope_start_distance_m + config.transition_length_m + config.slope_length_m + config.transition_length_m
        self.assertAlmostEqual(
            terrain_height(top, config), config.total_rise, places=9
        )
        self.assertAlmostEqual(config.total_rise, rise * (0.5 + 2.0), places=9)

    def test_downhill_profile_is_negative_and_symmetric(self):
        up = SlopeTerrainConfig(slope_angle_deg=3.0)
        down = SlopeTerrainConfig(slope_angle_deg=-3.0)
        x = np.linspace(-2.0, 12.0, 200)
        np.testing.assert_allclose(
            terrain_height_array(x, down),
            -terrain_height_array(x, up),
            atol=1e-12,
        )

    def test_profile_is_monotonic_for_uphill(self):
        x = np.linspace(-2.0, 12.0, 500)
        heights = terrain_height_array(x, self.config)
        self.assertTrue(np.all(np.diff(heights) >= -1e-12))


class SlopeTerrainHfieldTest(unittest.TestCase):
    def setUp(self):
        self.config = SlopeTerrainConfig(
            slope_angle_deg=2.0,
            ncol=64,
            nrow=4,
        )

    def test_hfield_data_shape_and_constant_along_y(self):
        data = hfield_data_3d(self.config)
        self.assertEqual(data.shape, (self.config.nrow, self.config.ncol))
        for row in range(self.config.nrow):
            np.testing.assert_array_equal(data[row], data[0])

    def test_hfield_data_surface_matches_profile(self):
        data = hfield_data_3d(self.config)
        centers = column_centers_x(self.config)
        surface = self.config.hfield_base_z + self.config.hfield_scale_z * data[0]
        expected = terrain_height_array(centers, self.config)
        shifted = expected - float(np.min(expected))
        np.testing.assert_allclose(surface, shifted, atol=1e-9)

    def test_flat_hfield_data_is_constant(self):
        data = flat_hfield_data_3d(self.config)
        self.assertEqual(data.shape, (self.config.nrow, self.config.ncol))
        self.assertTrue(np.all(data == data[0, 0]))

    def test_column_major_flatten_layout(self):
        data_2d = np.arange(self.config.nrow * self.config.ncol, dtype=np.float64).reshape(
            self.config.nrow, self.config.ncol
        )
        flat = hfield_data_flat_column_major(self.config, data_2d)
        self.assertEqual(flat.shape, (self.config.nrow * self.config.ncol,))
        # MuJoCo order: column (x) is the outer index.
        self.assertEqual(flat[0], data_2d[0, 0])
        self.assertEqual(flat[1], data_2d[1, 0])
        self.assertEqual(flat[self.config.nrow], data_2d[0, 1])

    def test_task_derived_config_uses_flat_fields(self):
        task = Rolling3DConfig(
            terrain_enabled=True,
            terrain_slope_angle_deg=-2.0,
            terrain_slope_start_distance_m=4.0,
        )
        validate_3d_config(task)
        config = slope_terrain_config_from_task(task)
        self.assertEqual(config.slope_angle_deg, -2.0)
        self.assertEqual(config.slope_start_distance_m, 4.0)


class SlopeTerrainMjcfBakeTest(unittest.TestCase):
    def test_bake_compiles_and_keeps_self_collision_contract(self):
        from curl_robot_2d_mjx.environment_3d import (
            validate_rollingquad_self_collision_contract_3d,
        )

        # Bake next to the source so the relative ../meshes references resolve.
        output = MESH_MODEL.with_name("_terrain_test_bake.xml")
        config = SlopeTerrainConfig(slope_angle_deg=2.0)
        try:
            inject_hfield_into_mjcf(MESH_MODEL, output, config, force=True)
            model = mujoco.MjModel.from_xml_path(str(output))
            floor_id = model.geom("floor").id
            self.assertEqual(model.geom_type[floor_id], mujoco.mjtGeom.mjGEOM_HFIELD)
            self.assertEqual(model.nhfield, 1)
            self.assertEqual(model.hfield_nrow[0], config.nrow)
            self.assertEqual(model.hfield_ncol[0], config.ncol)
            # The floor's contype/conaffinity must remain (1, 0).
            self.assertEqual(int(model.geom_contype[floor_id]), 1)
            self.assertEqual(int(model.geom_conaffinity[floor_id]), 0)
            validate_rollingquad_self_collision_contract_3d(model, "rollingquad_2")
        finally:
            if output.exists():
                output.unlink()

    def test_bake_rejects_existing_output_without_force(self):
        output = MESH_MODEL.with_name("_terrain_test_existing.xml")
        output.write_text("<mujoco/>", encoding="utf-8")
        try:
            with self.assertRaises(FileExistsError):
                inject_hfield_into_mjcf(MESH_MODEL, output, SlopeTerrainConfig())
        finally:
            if output.exists():
                output.unlink()


if __name__ == "__main__":
    unittest.main()

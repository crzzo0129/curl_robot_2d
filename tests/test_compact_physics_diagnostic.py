"""Dependency-light fixture, trajectory and trigger tests."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from scripts.diagnose_compact_physics import (
    fixed_fixture_xml, fold_progress, parse_args, sustained_triggers,
)

SOURCE = Path(__file__).resolve().parents[1] / 'assets/rollingquad_description_2/mjcf/rollingquad_primitive.xml'


class PhysicsDiagnosticTest(unittest.TestCase):
    def args(self, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            return parse_args(['--out', str(Path(tmp) / 'run'), *extra])

    def test_fixture_removes_only_root_motion_and_keeps_contacts_and_actuators(self):
        source = ET.parse(SOURCE).getroot()
        fixture = ET.fromstring(fixed_fixture_xml(SOURCE, .8))
        self.assertIsNone(fixture.find('./worldbody/body/freejoint'))
        self.assertEqual(len(fixture.findall('./worldbody//joint')), 12)
        for path in ('./actuator', './contact'):
            self.assertEqual(ET.tostring(source.find(path)), ET.tostring(fixture.find(path)))
        self.assertEqual([x.attrib for x in source.findall('./worldbody//geom')],
                         [x.attrib for x in fixture.findall('./worldbody//geom')])
        for key in fixture.findall('./keyframe/key'):
            original = source.find(f'./keyframe/key[@name="{key.get("name")}"]')
            self.assertEqual(key.get('qpos').split(), original.get('qpos').split()[7:])
            self.assertEqual(key.get('ctrl'), original.get('ctrl'))
        for asset in fixture.findall('./asset/*'):
            if asset.get('file'):
                self.assertTrue(Path(asset.get('file')).is_absolute())
                self.assertTrue(Path(asset.get('file')).exists())

    def test_incremental_motion_has_dwell_and_small_increments(self):
        args = self.args()
        delta = np.array([-.789, -.241])
        p0, duration = fold_progress(-1., delta, args)
        self.assertEqual(p0, 0.)
        self.assertEqual(duration, 8.)
        mid, _ = fold_progress(.25, delta, args)
        dwell, _ = fold_progress(.49, delta, args)
        self.assertEqual(mid, dwell)
        self.assertLessEqual(np.max(np.abs(delta * mid)), args.increment_rad)
        self.assertEqual(fold_progress(duration, delta, args)[0], 1.)
        self.assertEqual(fold_progress(duration + 10., delta, args)[0], 1.)

    def test_smooth_motion_clamps_and_is_monotonic(self):
        args = self.args('--trajectory', 'smooth')
        samples = [fold_progress(t, np.ones(8), args)[0] for t in np.linspace(-1, 7, 100)]
        self.assertEqual(samples[0], 0.)
        self.assertEqual(samples[-1], 1.)
        self.assertTrue(np.all(np.diff(samples) >= 0))

    def test_trigger_requires_contiguous_duration(self):
        timers = {}
        for _ in range(9):
            self.assertEqual(sustained_triggers({'upward': True}, timers, .001, .01), [])
        sustained_triggers({'upward': False}, timers, .001, .01)
        for _ in range(9):
            self.assertEqual(sustained_triggers({'upward': True}, timers, .001, .01), [])
        self.assertEqual(sustained_triggers({'upward': True}, timers, .001, .01), ['upward'])


if __name__ == '__main__':
    unittest.main()

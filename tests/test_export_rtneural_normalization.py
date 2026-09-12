"""Pure NumPy regression checks for tiny-variance deployment observations."""
import json
import unittest
import numpy as np
from scripts.export_rtneural import convert, _run_layers


class ExportNormalizationTest(unittest.TestCase):
    def test_constant_channel_preserves_small_dynamic_signal(self):
        checkpoint = (
            {"mean": np.array([1., 0.], dtype=np.float32),
             "std": np.array([1e-6, 1.], dtype=np.float32)},
            {"params": {"hidden_0": {
                "kernel": np.array([[1., 0.], [1., 0.]], dtype=np.float32),
                "bias": np.array([0.003, 0.], dtype=np.float32)}}}, {})
        doc = convert(checkpoint, {"action_scale": [0.5]}, normalization="batchnorm")
        doc = json.loads(json.dumps(doc))
        observations = np.array([[1., 0.01], [1., -0.01], [1.000001, 0.]], dtype=np.float32)
        expected = np.tanh((observations[:, 0] - 1.) / np.float32(1e-6)
                           + observations[:, 1] + np.float32(0.003))[:, None]
        np.testing.assert_allclose(_run_layers(observations, doc["layers"]), expected,
                                   rtol=1e-6, atol=1e-6)
        self.assertEqual(doc["in_shape"], [1, 2])
        self.assertEqual(doc["out_shape"], [1, 1])

    def test_unknown_normalization_rejected(self):
        with self.assertRaisesRegex(ValueError, "normalization"):
            convert(None, {}, normalization="unknown")


if __name__ == "__main__":
    unittest.main()

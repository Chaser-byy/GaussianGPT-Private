import importlib.util
import unittest


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch is not installed in this environment.")

import torch

from gaussiangpt.autoencoder.training.losses import (
    _as_batched_coords,
    _kernel_map_out_indices,
    _sparse_occupancy_targets,
)


try:
    import MinkowskiEngine as ME

    HAS_MINKOWSKI = True
except ImportError:
    HAS_MINKOWSKI = False


class LossCoordinateTests(unittest.TestCase):
    def test_as_batched_coords_adds_leading_batch_index(self):
        coords = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

        batched = _as_batched_coords(coords, torch.device("cpu"))

        expected = torch.tensor(
            [[0, 1, 2, 3], [0, 4, 5, 6]],
            dtype=torch.long,
        )
        self.assertEqual(batched.dtype, torch.long)
        self.assertTrue(torch.equal(batched, expected))

    def test_as_batched_coords_keeps_existing_batch_column(self):
        coords = torch.tensor(
            [[0, 1, 2, 3], [1, 4, 5, 6]],
            dtype=torch.int32,
        )

        batched = _as_batched_coords(coords, torch.device("cpu"))

        self.assertEqual(batched.dtype, torch.long)
        self.assertTrue(torch.equal(batched, coords.long()))

    def test_as_batched_coords_handles_empty_single_sample_coords(self):
        coords = torch.empty((0, 3), dtype=torch.long)

        batched = _as_batched_coords(coords, torch.device("cpu"))

        self.assertEqual(tuple(batched.shape), (0, 4))
        self.assertEqual(batched.dtype, torch.long)

    def test_kernel_map_out_indices_reads_documented_dict_format(self):
        mapping = {
            0: torch.tensor(
                [
                    [0, 2, 4],
                    [1, 3, 5],
                ],
                dtype=torch.int32,
            )
        }

        out_indices = _kernel_map_out_indices(mapping, torch.device("cpu"))

        self.assertEqual(out_indices.dtype, torch.long)
        self.assertTrue(torch.equal(out_indices, torch.tensor([1, 3, 5])))

    @unittest.skipIf(HAS_MINKOWSKI, "MinkowskiEngine is available.")
    def test_sparse_occupancy_targets_requires_minkowski(self):
        class FakeOcc:
            F = torch.zeros((1, 1), dtype=torch.float32)
            tensor_stride = 1

        with self.assertRaisesRegex(RuntimeError, "MinkowskiEngine"):
            _sparse_occupancy_targets(
                FakeOcc(),
                torch.zeros((1, 4), dtype=torch.long),
                stage_idx=0,
                n_stages=1,
                device=torch.device("cpu"),
            )

    @unittest.skipUnless(HAS_MINKOWSKI, "MinkowskiEngine is not available.")
    def test_sparse_occupancy_targets_uses_minkowski_coordinate_map(self):
        occ_coords = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 1],
                [1, 2, 2, 2],
                [1, 3, 3, 3],
            ],
            dtype=torch.int32,
        )
        occ_features = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
        occ = ME.SparseTensor(
            features=occ_features,
            coordinates=occ_coords,
            tensor_stride=2,
        )
        gt_coords = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 2, 2, 2],
                [1, 4, 4, 4],
            ],
            dtype=torch.long,
        )

        occ_logits, targets, stride = _sparse_occupancy_targets(
            occ,
            gt_coords,
            stage_idx=0,
            n_stages=1,
            device=torch.device("cpu"),
        )

        self.assertEqual(stride, 2)
        self.assertTrue(torch.equal(occ_logits, occ_features.squeeze(-1)))
        self.assertTrue(
            torch.equal(
                targets,
                torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=targets.dtype),
            )
        )


if __name__ == "__main__":
    unittest.main()

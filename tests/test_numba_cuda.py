"""Run on Colab for hardware coverage; simulator deliberately excludes shuffles."""
import importlib
import csv
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numba import config, cuda

from benchmarks.sweep_k import K_VALUES, run_sweep
from benchmarks.numba_configs import configurations
from benchmarks.validation import cpu_reference, validate_result
from benchmarks.plot_k import plot_results
from implementations.numba_cuda.common import require_cuda

MODULES = ("v2_1_block_tuning", "v2_2_coalescing", "v2_3_warp",
           "v2_4_shared_memory", "v2_5_reduction", "v2_6_warp_reduction")
SIMULATOR = bool(config.ENABLE_CUDASIM)
HARDWARE = not SIMULATOR and cuda.is_available()
if os.environ.get("NUMBA_TEST_REQUIRE_GPU") == "1":
    require_cuda()  # Fail loudly instead of reporting a skipped GPU suite.


class ReferenceTests(unittest.TestCase):
    def test_reference_and_validator(self):
        X = np.array([[0, 0], [0, 2], [10, 10], [10, 12]], dtype=np.float32)
        reference = cpu_reference(X, X[[0, 2]])
        np.testing.assert_array_equal(reference[0], [0, 0, 1, 1])
        np.testing.assert_allclose(reference[1], [[0, 1], [10, 11]])
        self.assertEqual(reference[-1], 4.0)
        result = (reference[0], reference[1].astype(np.float32), *reference[2:])
        validate_result(X, result, 2, reference_inertia=4.0)
        with self.assertRaises(AssertionError):
            validate_result(X, (1 - result[0], *result[1:]), 2)
        with self.assertRaises(AssertionError):
            validate_result(X, (*result[:-1], 0.0), 2)
        with self.assertRaises(AssertionError):
            validate_result(X, result, 2, reference_inertia=100.0)

    def test_reference_empty_cluster_retains_center(self):
        X = np.array([[0, 0], [0, 0], [10, 10]], dtype=np.float32)
        initial = X.copy()
        result = cpu_reference(X, initial)
        np.testing.assert_array_equal(result[1], initial)

    def test_configuration_and_plot_contract(self):
        configs = configurations((128, 256, 512))
        self.assertEqual(len(configs), 9)
        self.assertEqual(K_VALUES, (2, 4, 8, 16, 32, 64))
        rows = [dict(implementation=name, k=k, run=run, seconds=float(k + run))
                for name, _, _ in configs for k in K_VALUES for run in range(1, 6)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runtime.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            medians = plot_results(path, speedup=True)
            self.assertEqual(medians['V2.0 baseline'], {k: k + 3 for k in K_VALUES})
            self.assertEqual({p.name for p in Path(folder).glob('*.png')},
                             {'runtime_vs_k.png', 'speedup_vs_k.png'})

    def test_sweep_rejects_nonhardware(self):
        if HARDWARE:
            self.skipTest('Hardware present')
        with self.assertRaises(RuntimeError):
            require_cuda()


@unittest.skipUnless(SIMULATOR or HARDWARE, 'Needs CUDA simulator or NVIDIA CUDA')
class KernelTests(unittest.TestCase):
    def fits(self):
        for name in MODULES:
            if name in ('v2_3_warp', 'v2_6_warp_reduction') and SIMULATOR:
                continue
            fit = importlib.import_module(f'implementations.numba_cuda.{name}').kmeans
            yield name, fit

    def test_fixed_work_and_empty_clusters(self):
        X = np.array([[0, 0, 0], [0, 2, 0], [10, 10, 10], [10, 12, 10],
                      [0, 0, 0]], dtype=np.float32)
        initial = X[[0, 0, 2]].copy()  # Duplicate centroid must remain finite/unchanged.
        reference = cpu_reference(X, initial)
        for name, fit in self.fits():
            with self.subTest(name=name):
                result = fit(X, 3, num_iterations=20, initial_centroids=initial,
                             early_stopping=False, threads_per_block=32)
                validate_result(X, result, 3, reference_inertia=reference[-1])
                np.testing.assert_allclose(result[1], reference[1], atol=1e-6)
                np.testing.assert_array_equal(initial, X[[0, 0, 2]])
                np.testing.assert_array_equal(result[1][1], initial[1])
                early = fit(X, 3, initial_centroids=initial, threads_per_block=32)
                self.assertTrue(early[3])
                self.assertLess(early[2], 20)

    def test_partial_image_feature_and_centroid_tiles(self):
        # 35 images: partial block/warp; 131 features: partial shared/warp feature
        # tile; 5 centroids: partial shared centroid tile.
        X = np.random.default_rng(5).normal(size=(35, 131)).astype(np.float32)
        initial = X[:5].copy()
        reference = cpu_reference(X, initial, iterations=2)
        for name, fit in self.fits():
            with self.subTest(name=name):
                options = {'shards': 3} if name in ('v2_5_reduction', 'v2_6_warp_reduction') else {}
                result = fit(X, 5, num_iterations=2, initial_centroids=initial,
                             early_stopping=False, threads_per_block=32, **options)
                validate_result(X, result, 5, iterations=2, reference_inertia=reference[-1])
                np.testing.assert_allclose(result[1], reference[1], atol=2e-6)

    def test_argument_checks(self):
        X = np.zeros((3, 2), dtype=np.float32)
        for name, fit in self.fits():
            for options in ({'threads_per_block': 33}, {'num_iterations': 0},
                            {'num_clusters': 4}, {'initial_centroids': np.zeros((1, 2))},
                            {'threshold': -1}):
                with self.subTest(name=name, options=options), self.assertRaises(ValueError):
                    fit(X, **dict({'num_clusters': 2}, **options))

    def test_warp_simulator_is_explicitly_unsupported(self):
        if not SIMULATOR:
            self.skipTest('Only relevant to simulator')
        for name in ('v2_3_warp', 'v2_6_warp_reduction'):
            fit = importlib.import_module(f'implementations.numba_cuda.{name}').kmeans
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'real NVIDIA'):
                fit(np.zeros((3, 2), dtype=np.float32), 2)


@unittest.skipUnless(HARDWARE, 'Real NVIDIA CUDA required, no simulator warp emulation')
class HardwareTests(unittest.TestCase):
    def test_unified_gpu_smoke(self):
        # Real PyTorch and Numba through the exact same timing/validation runner.
        with tempfile.TemporaryDirectory() as folder:
            rows = run_sweep(folder, dataset='synthetic', suite='gpu', smoke=True,
                             speedup=True, no_progress=True)
            self.assertEqual(len(rows), 2 * 10 * 5)
            self.assertEqual(len({row['implementation'] for row in rows}), 10)

    def test_combo_matches_warp_with_different_shard_counts(self):
        from implementations.numba_cuda.v2_3_warp import kmeans as warp
        from implementations.numba_cuda.v2_6_warp_reduction import kmeans as combo
        X = np.random.default_rng(23).normal(size=(35, 131)).astype(np.float32)
        initial = X[:5].copy()
        kwargs = dict(num_clusters=5, num_iterations=20, initial_centroids=initial,
                      early_stopping=False, threads_per_block=64)
        reference = cpu_reference(X, initial)
        plain = warp(X, **kwargs)
        for shards in (1, 3, 32):
            with self.subTest(shards=shards):
                result = combo(X, shards=shards, **kwargs)
                validate_result(X, result, 5, reference_inertia=reference[-1])
                np.testing.assert_array_equal(result[0], plain[0])
                np.testing.assert_allclose(result[1], plain[1], rtol=1e-5, atol=2e-6)
        np.testing.assert_array_equal(initial, X[:5])

    def test_all_k_and_block_configurations_at_784_features(self):
        X = np.random.default_rng(19).random((137, 784), dtype=np.float32)
        for k in K_VALUES:
            initial = X[np.random.RandomState(42).choice(len(X), k, replace=False)].copy()
            reference = cpu_reference(X, initial)
            for name, fit, options in configurations((128, 256, 512)):
                with self.subTest(k=k, name=name):
                    result = fit(X, k, num_iterations=20, initial_centroids=initial,
                                 early_stopping=False, **options)
                    validate_result(X, result, k, reference_inertia=reference[-1])


if __name__ == '__main__':
    unittest.main()

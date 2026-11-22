import tempfile
import csv
import json
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from implementations import vectorized_implementation as numpy_impl
from implementations import scikit_implementation as sklearn_impl
from implementations import pytorch_implementation as pytorch_impl
from benchmarks.sweep_k import as_result, validate_result, run_sweep, load_benchmark_data, main


class FixedWorkTests(unittest.TestCase):
    def setUp(self):
        self.limit = threadpool_limits(limits=1)
        self.addCleanup(self.limit.restore_original_limits)
        torch.set_num_threads(1)

    def test_early_convergence_still_executes_twenty_updates(self):
        X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=np.float32)
        initial = X[[0, 2]].copy()
        expected = np.array([[0, .5], [10, 10.5]], dtype=np.float32)
        for name in ('numpy', 'sklearn', 'pytorch'):
            with self.subTest(name=name):
                kwargs = dict(num_clusters=2, num_iterations=20, initial_centroids=initial,
                              early_stopping=False)
                if name == 'numpy':
                    result = numpy_impl.kmeans(X, **kwargs)
                elif name == 'sklearn':
                    result = as_result(sklearn_impl.kmeans(X, **kwargs))
                else:
                    result = pytorch_impl.kmeans(torch.from_numpy(X), **kwargs)
                    result = (result[0].numpy(), result[1].numpy(), *result[2:])
                validate_result(X, result, 2, 20)
                np.testing.assert_allclose(result[1], expected, atol=1e-6)
                np.testing.assert_array_equal(initial, X[[0, 2]])

    def test_sklearn_performs_twenty_compiled_updates_after_convergence(self):
        from sklearn.cluster import _k_means_lloyd
        X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=np.float32)
        with patch.object(_k_means_lloyd, 'lloyd_iter_chunked_dense',
                          wraps=_k_means_lloyd.lloyd_iter_chunked_dense) as step:
            model = sklearn_impl.kmeans(X, 2, num_iterations=20,
                                        initial_centroids=X[[0, 2]], early_stopping=False)
        self.assertEqual(step.call_count, 21)
        self.assertTrue(all(call.kwargs.get('update_centers', True)
                            for call in step.call_args_list[:20]))
        self.assertFalse(step.call_args_list[-1].kwargs['update_centers'])
        np.testing.assert_array_equal(model.predict(X), model.labels_)

    def test_random_fixed_work_matches_reference(self):
        X = np.random.default_rng(7).normal(size=(300, 8)).astype(np.float32)
        initial = X[[0, 80, 150, 230]].copy()
        kwargs = dict(num_clusters=4, num_iterations=20, initial_centroids=initial, early_stopping=False)
        reference = numpy_impl.kmeans(X, **kwargs)
        for fit in (sklearn_impl.kmeans, pytorch_impl.kmeans):
            with self.subTest(fit=fit.__module__):
                if fit is sklearn_impl.kmeans:
                    result = as_result(fit(X, **kwargs))
                else:
                    result = fit(torch.from_numpy(X), **kwargs)
                    result = (result[0].numpy(), result[1].numpy(), *result[2:])
                validate_result(X, result, 4, 20, reference[-1])
                np.testing.assert_allclose(result[1], reference[1], rtol=1e-4, atol=1e-5)

    def test_empty_clusters_remain_finite(self):
        X = np.array([[0, 0], [0, 0], [10, 10], [10, 10]], dtype=np.float32)
        initial = np.array([[0, 0], [0, 0], [10, 10]], dtype=np.float32)
        for fit in (numpy_impl.kmeans, sklearn_impl.kmeans, pytorch_impl.kmeans):
            with self.subTest(fit=fit.__module__):
                is_torch = fit in (pytorch_impl.kmeans,)
                result = fit(torch.from_numpy(X) if is_torch else X, num_clusters=3,
                             num_iterations=20, initial_centroids=initial, early_stopping=False)
                if fit is sklearn_impl.kmeans:
                    result = as_result(result)
                elif is_torch:
                    result = (result[0].numpy(), result[1].numpy(), *result[2:])
                validate_result(X, result, 3, 20)

    def test_default_calls_still_stop_early(self):
        X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=np.float32)
        self.assertLess(numpy_impl.kmeans(X, 2)[2], 20)
        self.assertLess(sklearn_impl.kmeans(X, 2).n_iter_, 20)
        for fit in (pytorch_impl.kmeans,):
            self.assertLess(fit(torch.from_numpy(X), 2)[2], 20)

    def test_validator_rejects_wrong_assignment(self):
        X = np.array([[0, 0], [10, 10]], dtype=np.float32)
        with self.assertRaises(AssertionError):
            validate_result(X, (np.array([1, 0]), X, 20, False, 400.), 2, 20)

    def test_smoke_produces_only_requested_plots_and_five_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = run_sweep(folder, dataset="synthetic", cpu_only=True, smoke=True, no_progress=True, speedup=True)
            self.assertEqual(len(rows), 20)
            self.assertEqual({p.name for p in Path(folder).glob('*.png')},
                             {'runtime_vs_k.png', 'speedup_vs_k.png'})
            from benchmarks.plot_k import plot_results
            plot_results(Path(folder) / 'runtime.csv')
            self.assertEqual({p.name for p in Path(folder).glob('*.png')}, {'runtime_vs_k.png'})
            notes = Path(folder) / 'notes.txt'
            notes.write_text('keep this file')
            for filename in ('runtime.csv', 'environment.json', 'runtime_vs_k.png', 'speedup_vs_k.png'):
                (Path(folder) / filename).write_text('stale output')
            rerun = run_sweep(folder, dataset="synthetic", cpu_only=True, smoke=True, no_progress=True)
            self.assertEqual(len(rerun), 20)
            with (Path(folder) / 'runtime.csv').open(newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 20)
            metadata = json.loads((Path(folder) / 'environment.json').read_text())
            self.assertEqual(metadata['dataset'], 'synthetic')
            self.assertTrue((Path(folder) / 'runtime_vs_k.png').read_bytes().startswith(b'\x89PNG'))
            self.assertFalse((Path(folder) / 'speedup_vs_k.png').exists())
            self.assertEqual(notes.read_text(), 'keep this file')

    def test_cli_defaults_to_fashion_mnist(self):
        with patch('sys.argv', ['sweep_k']), patch('benchmarks.sweep_k.run_sweep') as run:
            main()
        self.assertEqual(run.call_args.kwargs['dataset'], 'fashion-mnist')

    def test_default_dataset_uses_fashion_loader_and_preserves_features(self):
        X = np.random.default_rng(2).random((300, 784), dtype=np.float32)
        with patch.object(numpy_impl, 'load_fashion_mnist', return_value=(X, np.zeros(300))) as load:
            full, _ = load_benchmark_data()
            small, _ = load_benchmark_data(smoke=True)
        self.assertEqual(load.call_count, 2)
        np.testing.assert_array_equal(full, X)
        np.testing.assert_array_equal(small, X[:256])
        self.assertEqual(small.shape, (256, 784))

    def test_synthetic_option_does_not_fetch_data(self):
        with patch.object(numpy_impl, 'load_fashion_mnist') as load:
            X, _ = load_benchmark_data('synthetic')
        load.assert_not_called()
        self.assertEqual(X.shape, (100000, 32))
        self.assertEqual(X.dtype, np.float32)

    def test_fashion_sweep_records_actual_dataset_and_shape(self):
        # Small loader fixture exercises the real runner without a download.
        X = np.random.default_rng(3).random((40, 7), dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder, patch.object(
                numpy_impl, 'load_fashion_mnist', return_value=(X, np.zeros(40))):
            rows = run_sweep(folder, cpu_only=True, smoke=True, no_progress=True)
            metadata = json.loads((Path(folder) / 'environment.json').read_text())
        self.assertEqual(metadata['dataset'], 'fashion-mnist')
        self.assertEqual((metadata['n'], metadata['d']), X.shape)
        self.assertIn('pixels / 255', metadata['preprocessing'])
        self.assertEqual(len(rows), 20)

    def test_fashion_download_failure_does_not_fall_back(self):
        with patch.object(numpy_impl, 'load_fashion_mnist', side_effect=OSError('offline')):
            with self.assertRaisesRegex(OSError, 'offline'):
                load_benchmark_data()

    @unittest.skipUnless(torch.cuda.is_available(), 'NVIDIA CUDA unavailable')
    def test_cuda_fixed_work(self):
        X = np.random.default_rng(1).normal(size=(256, 4)).astype(np.float32)
        kwargs = dict(num_clusters=4, num_iterations=20, initial_centroids=X[:4], early_stopping=False)
        reference = numpy_impl.kmeans(X, **kwargs)
        for fit in (pytorch_impl.kmeans,):
            result = fit(torch.as_tensor(X, device='cuda'), **kwargs)
            torch.cuda.synchronize()
            result = (result[0].cpu().numpy(), result[1].cpu().numpy(), *result[2:])
            validate_result(X, result, 4, 20, reference[-1])



if __name__ == '__main__':
    unittest.main()

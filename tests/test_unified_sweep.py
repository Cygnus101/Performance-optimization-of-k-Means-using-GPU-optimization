"""Regression coverage for complete-fit timing and unified orchestration."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from benchmarks import sweep_k
from implementations import vectorized_implementation as numpy_impl
from implementations import pytorch_implementation as torch_impl


class UnifiedSweepTests(unittest.TestCase):
    def test_transfers_are_between_timer_boundaries(self):
        events = []
        X = np.zeros((4, 2), dtype=np.float32)
        initial = X[:2].copy()

        class ResultTensor:
            def __init__(self, value):
                self.value = value

            def cpu(self):
                events.append('D2H')
                return self

            def numpy(self):
                return self.value

        def transfer(array, *, device):
            events.append('H2D')
            return array

        def fit(samples, *, initial_centroids):
            events.append('fit')
            return (ResultTensor(np.zeros(4, dtype=np.int32)),
                    ResultTensor(initial_centroids), 20, False, 0.0)

        clock = iter((10.0, 12.0))

        def now():
            events.append('clock')
            return next(clock)

        with patch.object(sweep_k, 'perf_counter', side_effect=now):
            result, seconds = sweep_k.measure_fit(
                lambda: sweep_k.torch_complete_fit(
                    X, initial, torch=SimpleNamespace(as_tensor=transfer), fit=fit, device='cuda'),
                lambda: events.append('sync'))
        self.assertEqual(events, ['sync', 'clock', 'H2D', 'H2D', 'fit', 'D2H', 'D2H', 'sync', 'clock'])
        self.assertEqual(seconds, 2.0)
        self.assertIsInstance(result[0], np.ndarray)

    def test_all_suites_share_warmup_runs_reference_and_outputs(self):
        # Exercise all adapters without claiming CUDA execution: CPU tensors and
        # instrumented NumPy stand-ins for two independent Numba configurations.
        calls, synchronizations = [], []
        configs = []
        for name, options in [('V2.0 baseline', {}), ('V2.1 threads=128', {'threads_per_block': 128})]:
            def fit(X, *, name=name, **kwargs):
                calls.append((name, kwargs.pop('threads_per_block', None), kwargs['num_clusters']))
                return numpy_impl.kmeans(X, **kwargs)
            configs.append((name, fit, options))
        fake_cuda = SimpleNamespace(synchronize=lambda: synchronizations.append('numba'))
        host_torch = SimpleNamespace(as_tensor=torch.as_tensor,
                                     cuda=SimpleNamespace(synchronize=lambda device: None))
        backends = {'torch': (host_torch, torch_impl.kmeans, torch.device('cpu')),
                    'numba': (fake_cuda, configs)}
        X = np.random.default_rng(8).normal(size=(32, 5)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder, patch.object(
                sweep_k, 'prepare_backends', return_value=(backends, {'gpu_status': 'test doubles'})), patch.object(
                sweep_k, 'load_benchmark_data', return_value=(X, 'test fixture')):
            rows = sweep_k.run_sweep(folder, suite='all', smoke=True, no_progress=True, speedup=True)
            metadata = json.loads((Path(folder) / 'environment.json').read_text())
            self.assertEqual(metadata['status'], 'complete')
            self.assertEqual(metadata['timing_protocol'], 'complete_fit_v1')
            self.assertEqual(metadata['speedup_baseline'], 'NumPy CPU')
            self.assertEqual({p.name for p in Path(folder).iterdir()}, set(sweep_k.ARTIFACTS))
        self.assertEqual(len(rows), 2 * 5 * 5)
        for name, _, options in configs:
            recorded = [c for c in calls if c[0] == name]
            self.assertEqual(len(recorded), 2 * 6)  # Warmup + five runs, at each K.
            self.assertEqual({c[1] for c in recorded}, {options.get('threads_per_block')})
        self.assertEqual(len(synchronizations), 2 * 2 * 11)  # 1 warmup + 2 per timed run.
        self.assertEqual({r['run'] for r in rows}, {1, 2, 3, 4, 5})
        self.assertTrue(all('reference_inertia' in r for r in rows))

    def test_custom_k_list_overrides_smoke_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = sweep_k.run_sweep(folder, dataset='synthetic', suite='cpu', smoke=True,
                                     k_values=[3, 6], no_progress=True)
            metadata = json.loads((Path(folder) / 'environment.json').read_text())
            self.assertEqual(metadata['k'], [3, 6])
            self.assertEqual({r['k'] for r in rows}, {3, 6})
            self.assertEqual(len(rows), 20)
        for values in ([], [0], [-1], [2, 2], [1.5], [True], [257]):
            with self.subTest(values=values), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    sweep_k.run_sweep(folder, dataset='synthetic', suite='cpu', smoke=True,
                                      k_values=values, no_progress=True)

    def test_cli_passes_custom_k_list(self):
        with patch('sys.argv', ['sweep_k', '--k', '2', '8', '32']), patch.object(
                sweep_k, 'run_sweep') as run:
            sweep_k.main()
        self.assertEqual(run.call_args.kwargs['k_values'], (2, 8, 32))

    def test_failure_marks_partial_run_and_removes_final_plots(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
                sweep_k, 'cpu_reference', side_effect=AssertionError('bad reference')):
            with self.assertRaisesRegex(AssertionError, 'bad reference'):
                sweep_k.run_sweep(folder, dataset='synthetic', suite='cpu', smoke=True, no_progress=True)
            metadata = json.loads((Path(folder) / 'environment.json').read_text())
            self.assertEqual(metadata['status'], 'failed')
            self.assertFalse((Path(folder) / 'runtime_vs_k.png').exists())


if __name__ == '__main__':
    unittest.main()

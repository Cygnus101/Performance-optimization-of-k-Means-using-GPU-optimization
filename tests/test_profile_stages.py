"""Profiling accounting, actual-fit equivalence and output failure behavior."""
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from implementations import vectorized_implementation as numpy_impl
from implementations import scikit_implementation as sklearn_impl
from implementations.profiling import Collector, stage
from benchmarks import profile_stages as profile
from benchmarks.sweep_k import as_result


class ProfileTests(unittest.TestCase):
    def test_actual_cpu_paths_keep_results_and_fixed_work(self):
        X = np.random.default_rng(42).normal(size=(37, 5)).astype(np.float32)
        initial = X[[0, 1, 2]].copy()
        for name, fit in (("numpy", numpy_impl.kmeans), ("sklearn", sklearn_impl.kmeans)):
            def run():
                result = fit(X, 3, 4, initial_centroids=initial, early_stopping=False)
                return as_result(result) if name=="sklearn" else result
            plain = run()
            measured, elapsed, records = Collector().measure(run)
            np.testing.assert_array_equal(plain[0], measured[0])
            np.testing.assert_array_equal(plain[1], measured[1])
            self.assertEqual(plain[2:], measured[2:])
            self.assertAlmostEqual(sum(r["wall_seconds"] for r in records), elapsed, places=12)
            iterative = [r for r in records if r["stage"] == ("assignment" if name=="numpy" else "lloyd_fused")]
            self.assertEqual([r["step"] for r in iterative], [1,2,3,4])
            self.assertTrue(all(r["event_seconds"] is None for r in records))

    def test_pytorch_host_logic_with_profiling(self):
        import torch
        from implementations.pytorch_implementation import kmeans
        X = torch.tensor([[0., 1.], [1., 2.], [8., 7.], [9., 8.]])
        initial = X[[0, 2]].clone()
        def fit():
            return kmeans(X, 2, 4, initial_centroids=initial, early_stopping=False)
        plain = fit()
        measured, elapsed, records = Collector().measure(fit)
        torch.testing.assert_close(plain[0], measured[0], rtol=0, atol=0)
        torch.testing.assert_close(plain[1], measured[1], rtol=0, atol=0)
        self.assertEqual(plain[2:], measured[2:])
        self.assertEqual([r["step"] for r in records if r["stage"]=="assignment"], [1,2,3,4])
        self.assertAlmostEqual(sum(r["wall_seconds"] for r in records), elapsed, places=12)

    def test_events_resolve_after_final_fence_without_stage_fences(self):
        calls = []
        class Event:
            def record(self): calls.append("record")
            def elapsed_time(self, other):
                self_test.assertEqual(calls[-1], "fence")
                return 2.5
        class Backend:
            @staticmethod
            def Event(enable_timing): return Event()
        self_test = self
        def fit():
            with stage("assignment", device=True, step=1): calls.append("launch")
            with stage("centroid_update", device=True, step=1): calls.append("launch")
            return "result"
        result, elapsed, rows = Collector(Backend()).measure(fit, lambda:calls.append("fence"))
        self.assertEqual(result, "result")
        self.assertEqual(calls.count("fence"), 2)
        self.assertEqual(calls, ["fence","record","launch","record","record","launch","record","fence"])
        self.assertEqual(rows[0]["event_seconds"], .0025)
        self.assertAlmostEqual(sum(r["wall_seconds"] for r in rows), elapsed, places=12)

    def test_nested_scopes_rejected_and_collector_restored(self):
        def bad():
            with stage("assignment"):
                with stage("centroid_update"): pass
        with self.assertRaisesRegex(RuntimeError, "Nested"):
            Collector().measure(bad)
        with stage("outside"): pass  # Active collector restored even on exception.

    def test_outputs_na_and_failure_state(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            runtime, steps, stages = profile.run_profile(output, dataset="synthetic", smoke=True,
                    backend="cpu", k_values=[2], iterations=3, runs=2)
            self.assertEqual(len(runtime), 8)
            self.assertEqual(json.loads((output/"environment.json").read_text())["status"], "complete")
            self.assertEqual(len(json.loads((output/"environment.json").read_text())["warmup"]), 2)
            self.assertTrue(all(r["wall_seconds"] is None for r in stages if r["stage"]=="h2d"))
            self.assertTrue(all(r["wall_seconds"] is None for r in stages if
                                r["implementation"]=="scikit-learn CPU" and r["stage"]=="assignment"))
            for name in profile.ARTIFACTS:
                self.assertTrue((output/name).exists(), name)
            with (output/"stages.csv").open() as stream:
                h2d = [r for r in csv.DictReader(stream) if r["stage"]=="h2d"]
            self.assertEqual(h2d[0]["wall_seconds"], "")
            with patch.object(profile, "validate_result", side_effect=ValueError("test rejection")):
                with self.assertRaisesRegex(ValueError, "test rejection"):
                    profile.run_profile(output, dataset="synthetic", smoke=True, backend="cpu", runs=1)
            self.assertEqual(json.loads((output/"environment.json").read_text())["status"], "failed")
            self.assertFalse((output/"elapsed.png").exists())
            self.assertFalse((output/"summary.csv").exists())

    def test_invalid_selection_fails_before_backend_initialization(self):
        with patch.object(profile, "prepare_backends", side_effect=AssertionError("backend initialized")):
            for options in (dict(versions=["v0"]), dict(versions=["pytorch"],backend="cpu"),
                            dict(k_values=[2,2]), dict(runs=0)):
                with self.assertRaises(ValueError): profile.run_profile("unused", **options)


class CUDAProfileTests(unittest.TestCase):
    def test_real_gpu_all_current_configurations(self):
        import torch
        if not torch.cuda.is_available():
            if os.environ.get("NUMBA_TEST_REQUIRE_GPU")=="1":
                self.fail("Real NVIDIA CUDA required by NUMBA_TEST_REQUIRE_GPU")
            self.skipTest("Real NVIDIA CUDA unavailable")
        with tempfile.TemporaryDirectory() as directory:
            runtime, steps, stages = profile.run_profile(directory, dataset="synthetic", smoke=True,
                backend="gpu", k_values=[2], runs=2)
            self.assertEqual(len({r["implementation"] for r in runtime}), 10)
            for name in {r["implementation"] for r in runtime}:
                for run in (1,2):
                    records = [r for r in steps if r["implementation"]==name and r["run"]==run]
                    elapsed = next(r["seconds"] for r in runtime if r["implementation"]==name and
                                   r["run"]==run and r["mode"]=="profiled")
                    self.assertAlmostEqual(sum(r["wall_seconds"] for r in records), elapsed, places=10)
                    assignments = [r for r in records if r["stage"]=="assignment"]
                    self.assertEqual(len(assignments), 20)
                    self.assertTrue(all(r["event_seconds"] is not None and r["event_seconds"]>=0 for r in assignments))


if __name__ == "__main__":
    unittest.main()

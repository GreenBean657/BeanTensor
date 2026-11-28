#!/usr/bin/env python3
"""btui_logic_tests.py — stdlib-unittest suite for the btui core (no curses).

Covers python/btui.py's non-UI surface: the ModelSpec/LayerSpec dataclasses,
build_net's statement-sequenced deterministic construction, the name+shape
weight-reuse copy (byte-level), the SharedState lock-guarded metrics store,
and run_training's stop/error/epoch semantics on tiny synthetic CSV fixtures.

The btui module must import without a terminal, so the first test pins the
lazy-curses contract: importing btui must not pull curses into sys.modules.

Run directly (bindings on PYTHONPATH) or through the btui_logic ctest:
    PYTHONPATH=scratch/build-tui/Bindings python3 -m unittest tests.btui_logic_tests
"""

import os
import sys
import threading
import time
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import numpy as np  # noqa: E402

import beantensor as bt  # noqa: E402

import btui  # noqa: E402

F32 = bt.DType.Float32
CPU = bt.Device.CPU

SCRATCH_BTUI = os.path.join(REPO_ROOT, "scratch", "btui")

# 12 rows, 2 features, 2 classes, perfectly separable; batch 2 -> 6 batches.
CSV_12ROW = (
    "x1,x2,cls\n"
    "0.0,0.0,a\n"
    "0.1,0.0,a\n"
    "0.0,0.1,a\n"
    "0.2,0.1,a\n"
    "0.1,0.2,a\n"
    "0.0,0.2,a\n"
    "1.0,1.0,b\n"
    "1.0,0.9,b\n"
    "0.9,1.0,b\n"
    "1.1,1.0,b\n"
    "1.0,1.1,b\n"
    "0.9,0.9,b\n"
)


def write_fixture(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def tiny_spec() -> "btui.ModelSpec":
    return btui.ModelSpec(
        layers=[
            btui.LayerSpec("Linear", {"in_features": 4, "out_features": 8}),
            btui.LayerSpec("Relu", {}),
            btui.LayerSpec("Dropout", {"p": 0.25}),
            btui.LayerSpec("Linear", {"in_features": 8, "out_features": 3}),
        ]
    )


def param_bytes(model) -> dict:
    return {name: t.to_numpy().tobytes() for name, t in model.named_parameters().items()}


class LazyCursesTest(unittest.TestCase):
    def test_import_btui_does_not_import_curses(self):
        self.assertNotIn("curses", sys.modules, "import btui must keep curses lazy")


class BuildNetTest(unittest.TestCase):
    def test_named_parameter_keys_and_shapes(self):
        net = btui.build_net(tiny_spec(), seed=1234)
        params = net.named_parameters()
        self.assertEqual(set(params.keys()), {"0.weight", "0.bias", "3.weight", "3.bias"})
        self.assertEqual(params["0.weight"].shape, (4, 8))
        self.assertEqual(params["0.bias"].shape, (8,))
        self.assertEqual(params["3.weight"].shape, (8, 3))
        self.assertEqual(params["3.bias"].shape, (3,))
        for tensor in params.values():
            self.assertEqual(tensor.dtype, F32)
            self.assertEqual(tensor.device, CPU)

    def test_same_seed_same_spec_is_byte_identical(self):
        first = btui.build_net(tiny_spec(), seed=99)
        second = btui.build_net(tiny_spec(), seed=99)
        self.assertEqual(param_bytes(first), param_bytes(second))

    def test_different_seed_reinitializes(self):
        first = btui.build_net(tiny_spec(), seed=1)
        second = btui.build_net(tiny_spec(), seed=2)
        self.assertNotEqual(param_bytes(first), param_bytes(second))

    def test_unknown_layer_type_raises_value_error(self):
        spec = btui.ModelSpec(layers=[btui.LayerSpec("NoSuchLayer", {})])
        with self.assertRaises(ValueError):
            btui.build_net(spec, seed=1)

    def test_missing_param_raises_value_error(self):
        spec = btui.ModelSpec(layers=[btui.LayerSpec("Linear", {"in_features": 4})])
        with self.assertRaises(ValueError):
            btui.build_net(spec, seed=1)


class DiffKeepWeightsTest(unittest.TestCase):
    def test_same_spec_copies_everything_byte_identical(self):
        old = btui.build_net(tiny_spec(), seed=1234)
        # A different seed proves the copy, not the init, produced the match.
        new = btui.build_net(tiny_spec(), seed=5678)
        self.assertNotEqual(param_bytes(old), param_bytes(new))
        copied = btui.diff_keep_weights(old, new)
        self.assertEqual(sorted(copied), ["0.bias", "0.weight", "3.bias", "3.weight"])
        self.assertEqual(param_bytes(old), param_bytes(new))

    def test_changed_shape_copies_matching_prefix_only(self):
        old = btui.build_net(tiny_spec(), seed=1234)
        changed = btui.ModelSpec(
            layers=[
                btui.LayerSpec("Linear", {"in_features": 4, "out_features": 8}),
                btui.LayerSpec("Relu", {}),
                btui.LayerSpec("Dropout", {"p": 0.25}),
                btui.LayerSpec("Linear", {"in_features": 8, "out_features": 5}),
            ]
        )
        new = btui.build_net(changed, seed=5678)
        fresh = param_bytes(btui.build_net(changed, seed=5678))
        old_params = param_bytes(old)
        copied = btui.diff_keep_weights(old, new)
        self.assertEqual(copied, ["0.weight", "0.bias"])
        new_params = param_bytes(new)
        # Copied entries carry the old bytes.
        self.assertEqual(new_params["0.weight"], old_params["0.weight"])
        self.assertEqual(new_params["0.bias"], old_params["0.bias"])
        # Shape-mismatched entries were reinitialized (untouched by the copy):
        # they are still byte-identical to a fresh same-seed build, and they
        # cannot equal the old tensors whose shapes differ.
        self.assertEqual(new_params["3.weight"], fresh["3.weight"])
        self.assertEqual(new_params["3.bias"], fresh["3.bias"])
        self.assertNotEqual(new_params["3.weight"], old_params["3.weight"])
        self.assertNotEqual(new_params["3.bias"], old_params["3.bias"])

    def test_disjoint_specs_copy_nothing(self):
        old = btui.build_net(tiny_spec(), seed=1)
        other_spec = btui.ModelSpec(layers=[btui.LayerSpec("Linear", {"in_features": 2, "out_features": 2})])
        new = btui.build_net(other_spec, seed=1)
        self.assertEqual(btui.diff_keep_weights(old, new), [])


class ScriptedStop:
    """threading.Event-shaped fake: is_set() is False for the first
    `true_after` calls, True from then on. Models a stop request that lands
    while batch `true_after` + 1 is in flight, so the run stops at that
    batch's boundary."""

    def __init__(self, true_after: int):
        self._calls = 0
        self._true_after = true_after

    def is_set(self) -> bool:
        self._calls += 1
        return self._calls > self._true_after


class RunTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_path = os.path.join(SCRATCH_BTUI, "train_12row.csv")
        write_fixture(cls.csv_path, CSV_12ROW)

    def data_cfg(self, **overrides) -> "btui.DataConfig":
        cfg = {
            "source": "csv",
            "csv_path": self.csv_path,
            "train_rows": None,
            "test_rows": 2000,
        }
        cfg.update(overrides)
        return btui.DataConfig(**cfg)

    def hp_cfg(self, **overrides) -> "btui.HyperConfig":
        cfg = {
            "lr": 1e-2,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "epochs": 1,
            "batch": 2,
            "seed": 1234,
            "optimizer": "adam",
        }
        cfg.update(overrides)
        return btui.HyperConfig(**cfg)

    def csv_spec(self) -> "btui.ModelSpec":
        return btui.ModelSpec(
            layers=[
                btui.LayerSpec("Linear", {"in_features": 2, "out_features": 8}),
                btui.LayerSpec("Relu", {}),
                btui.LayerSpec("Linear", {"in_features": 8, "out_features": 2}),
            ]
        )

    def test_stop_after_three_batches_stops_at_batch_four(self):
        shared = btui.SharedState()
        stop = ScriptedStop(true_after=3)
        net = btui.run_training(self.csv_spec(), self.data_cfg(), self.hp_cfg(), shared, stop)
        self.assertIsNotNone(net)
        self.assertIsNone(shared.error)
        self.assertFalse(shared.running)
        # 12 rows / batch 2 = 6 batches; the stop lands at batch 4's boundary.
        self.assertEqual(len(shared.losses), 4)
        self.assertEqual(shared.batches_done, 4)
        # Stopped mid-epoch: no per-epoch evaluate stats were pushed.
        self.assertEqual(shared.epoch_stats, [])

    def test_full_epoch_pushes_losses_and_epoch_stats(self):
        shared = btui.SharedState()
        net = btui.run_training(
            self.csv_spec(), self.data_cfg(), self.hp_cfg(), shared, threading.Event()
        )
        self.assertIsNotNone(net)
        self.assertIsNone(shared.error)
        self.assertFalse(shared.running)
        self.assertEqual(len(shared.losses), 6)
        self.assertTrue(all(np.isfinite(list(shared.losses))))
        self.assertEqual(shared.batches_done, 6)
        self.assertEqual(len(shared.epoch_stats), 1)
        stats = shared.epoch_stats[0]
        self.assertEqual(stats["epoch"], 1)
        self.assertEqual(stats["epochs"], 1)
        self.assertTrue(np.isfinite(stats["train_mean_loss"]))
        self.assertGreaterEqual(stats["test_accuracy"], 0.0)
        self.assertLessEqual(stats["test_accuracy"], 1.0)
        self.assertTrue(np.isfinite(stats["test_mean_loss"]))
        self.assertGreater(shared.batches_per_s, 0.0)
        # The evaluating flag never leaks past a finished run (try/finally).
        self.assertFalse(shared.evaluating)

    def test_sgd_optimizer_runs(self):
        shared = btui.SharedState()
        net = btui.run_training(
            self.csv_spec(), self.data_cfg(), self.hp_cfg(optimizer="sgd"),
            shared, threading.Event(),
        )
        self.assertIsNotNone(net)
        self.assertIsNone(shared.error)
        self.assertEqual(len(shared.losses), 6)

    def test_unknown_optimizer_raises_value_error(self):
        shared = btui.SharedState()
        with self.assertRaises(ValueError):
            btui.run_training(
                self.csv_spec(), self.data_cfg(), self.hp_cfg(optimizer="rmsprop"),
                shared, threading.Event(),
            )
        # A config error is raised to the caller, not captured as a run failure.
        self.assertIsNone(shared.error)

    def test_failure_captures_traceback_and_clears_running(self):
        shared = btui.SharedState()
        missing = self.data_cfg(csv_path=os.path.join(SCRATCH_BTUI, "__no_such__.csv"))
        net = btui.run_training(self.csv_spec(), missing, self.hp_cfg(), shared, threading.Event())
        self.assertIsNone(net)
        self.assertFalse(shared.running)
        self.assertIsNotNone(shared.error)
        self.assertIn("Traceback", shared.error)

    def test_weight_reuse_seeds_matching_params(self):
        old = btui.build_net(self.csv_spec(), seed=777)
        widened = btui.ModelSpec(
            layers=[
                btui.LayerSpec("Linear", {"in_features": 2, "out_features": 8}),
                btui.LayerSpec("Relu", {}),
                btui.LayerSpec("Linear", {"in_features": 8, "out_features": 4}),
            ]
        )
        hp = self.hp_cfg(epochs=1)
        shared = btui.SharedState()
        net = btui.run_training(
            widened, self.data_cfg(), hp, shared,
            threading.Event(), seed_weights_from=old,
        )
        self.assertIsNotNone(net)
        self.assertIsNone(shared.error)
        # The seeding window is always cleared when the copy finishes, even
        # though this run used it.
        self.assertFalse(shared.seeding)
        # The reuse itself is pinned exactly: the run's first-batch loss is a
        # pure function of the reused starting params and the (seeded) first
        # batch, so it must equal the loss of a manually built net that had
        # the same diff_keep_weights copy applied, bit-for-bit.
        opts = bt.CsvOptions()
        opts.feature_dtype = F32
        ds = bt.load_csv(self.csv_path, opts)
        loader = bt.DataLoader(ds.features, ds.labels, hp.batch, True, hp.seed, False)
        # run_training resets the loader at the top of every epoch, so the
        # first trained batch is the SECOND shuffle off the loader's seeded
        # stream (the constructor shuffles once already, Data/DataLoader.cpp).
        loader.reset()
        features, labels = loader.next_batch()
        ref = btui.build_net(widened, hp.seed)
        ref_fresh_loss = bt.softmax_cross_entropy(ref.forward(features), labels).item()
        copied = btui.diff_keep_weights(old, ref)
        self.assertEqual(copied, ["0.weight", "0.bias"])
        ref_reused_loss = bt.softmax_cross_entropy(ref.forward(features), labels).item()
        self.assertNotEqual(ref_fresh_loss, ref_reused_loss)
        self.assertEqual(shared.losses[0], ref_reused_loss)

    def test_data_config_validation(self):
        with self.assertRaises(ValueError):
            btui.DataConfig(source="bogus").validate()
        with self.assertRaises(ValueError):
            btui.DataConfig(source="csv", csv_path="").validate()
        with self.assertRaises(ValueError):
            btui.DataConfig(source="csv", csv_path=self.csv_path, test_rows=0).validate()
        with self.assertRaises(ValueError):
            btui.DataConfig(source="csv", csv_path=self.csv_path, csv_nulls="bogus").validate()
        with self.assertRaises(ValueError):
            btui.DataConfig(source="csv", csv_path=self.csv_path, csv_normalize="bogus").validate()
        btui.DataConfig(source="csv", csv_path=self.csv_path).validate()
        btui.DataConfig(source="csv", csv_path=self.csv_path,
                        csv_nulls="FillMedian", csv_normalize="MinMax").validate()
        btui.DataConfig(source="mnist_flat").validate()
        btui.DataConfig(source="mnist_nchw").validate()

    def test_csv_policies_reach_the_loader(self):
        # A null cell under the default KeepNaN stays NaN; FillMean fills it.
        # Both map through the config's policy strings onto the CsvOptions
        # enums (CsvNormalize.None via getattr).
        null_csv = os.path.join(SCRATCH_BTUI, "null_policy.csv")
        write_fixture(null_csv, "f1,f2,cls\n1.0,,a\n2.0,4.0,b\n")
        keep = btui.DataConfig(source="csv", csv_path=null_csv)
        (train_x, _), _ = btui._load_data(keep)
        self.assertTrue(np.isnan(train_x.to_numpy()).any())
        fill = btui.DataConfig(source="csv", csv_path=null_csv,
                               csv_nulls="FillMean", csv_normalize="Standardize")
        (filled_x, _), _ = btui._load_data(fill)
        self.assertFalse(np.isnan(filled_x.to_numpy()).any())

    def test_hyperconfig_rejects_nonfinite_lr_eps(self):
        # float("inf") > 0.0 is True; direct callers of HyperConfig.validate
        # must get the same rejection as the UI forms.
        for bad in (float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                btui.HyperConfig(lr=bad).validate()
            with self.assertRaises(ValueError):
                btui.HyperConfig(eps=bad).validate()

    def test_hyperconfig_validates_betas(self):
        for bad in (0.0, 1.0, -0.5, 1.5, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                btui.HyperConfig(beta1=bad).validate()
            with self.assertRaises(ValueError):
                btui.HyperConfig(beta2=bad).validate()
        btui.HyperConfig(beta1=0.9, beta2=0.999).validate()

    def test_trainjob_validates_config_at_construction(self):        # A bad run must never vanish silently: TrainJob rejects an invalid
        # config synchronously at construction, before any thread starts.
        with self.assertRaises(ValueError):
            btui.TrainJob(
                self.csv_spec(), self.data_cfg(), self.hp_cfg(optimizer="bogus"),
                btui.SharedState(),
            )
        with self.assertRaises(ValueError):
            btui.TrainJob(
                self.csv_spec(), btui.DataConfig(source="bogus"), self.hp_cfg(),
                btui.SharedState(),
            )


class TrainJobTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_path = os.path.join(SCRATCH_BTUI, "train_12row.csv")
        write_fixture(cls.csv_path, CSV_12ROW)

    def test_threaded_run_and_stop(self):
        spec = btui.ModelSpec(
            layers=[
                btui.LayerSpec("Linear", {"in_features": 2, "out_features": 8}),
                btui.LayerSpec("Relu", {}),
                btui.LayerSpec("Linear", {"in_features": 8, "out_features": 2}),
            ]
        )
        data_cfg = btui.DataConfig(source="csv", csv_path=self.csv_path)
        hp_cfg = btui.HyperConfig(lr=1e-2, epochs=100, batch=2, seed=42)
        shared = btui.SharedState()
        job = btui.TrainJob(spec, data_cfg, hp_cfg, shared)
        job.start()
        # Wait for the run to go live, then stop at the next batch boundary.
        deadline = 10.0
        start = time.perf_counter()
        while not shared.running and time.perf_counter() - start < deadline:
            time.sleep(0.001)
        self.assertTrue(shared.running, "training thread never went live")
        job.stop()
        job.join(timeout=30.0)
        self.assertFalse(job.is_alive(), "training thread did not finish after stop")
        self.assertFalse(shared.running)
        self.assertIsNone(shared.error)
        self.assertIsNotNone(job.net)
        self.assertLess(shared.batches_done, 100 * 6)


class DeviceConfigTest(unittest.TestCase):
    def test_probe_returns_consistent_bool(self):
        first = btui.probe_gpu()
        self.assertIsInstance(first, bool)
        self.assertEqual(first, btui.probe_gpu())

    def test_gpu_device_rejected_without_gpu(self):
        cfg = btui.DataConfig(source="csv", csv_path="x.csv", device="gpu")
        with self.assertRaises(ValueError) as ctx:
            cfg.validate(has_gpu=False)
        self.assertIn("no gpu in this build", str(ctx.exception))
        # Accepted when the probe says a GPU exists.
        cfg.validate(has_gpu=True)

    def test_device_value_validation(self):
        with self.assertRaises(ValueError):
            btui.DataConfig(source="csv", csv_path="x.csv", device="tpu").validate(
                has_gpu=True
            )
        btui.DataConfig(source="csv", csv_path="x.csv", device="cpu").validate(
            has_gpu=False
        )

    def test_trainjob_rejects_gpu_without_gpu(self):
        spec = btui.ModelSpec(layers=[
            btui.LayerSpec("Linear", {"in_features": 2, "out_features": 2}),
        ])
        cfg = btui.DataConfig(source="csv", csv_path="x.csv", device="gpu")
        with self.assertRaises(ValueError) as ctx:
            btui.TrainJob(spec, cfg, btui.HyperConfig(), btui.SharedState(),
                          has_gpu=False)
        self.assertIn("no gpu in this build", str(ctx.exception))


class GpuPathTest(unittest.TestCase):
    """GPU-path coverage. Runs for real only on a GPU-backend build — the
    probe gates it, so the CPU build's ctest reports SKIPPED; the CUDA
    build's ctest executes it (under the gpu-test wrapper)."""

    @classmethod
    def setUpClass(cls):
        if not btui.probe_gpu():
            raise unittest.SkipTest("SKIP: no gpu in this build")
        cls.csv_path = os.path.join(SCRATCH_BTUI, "gpu_path_12row.csv")
        write_fixture(cls.csv_path, CSV_12ROW)

    def spec(self) -> "btui.ModelSpec":
        return btui.ModelSpec(layers=[
            btui.LayerSpec("Linear", {"in_features": 2, "out_features": 8}),
            btui.LayerSpec("Relu", {}),
            btui.LayerSpec("Linear", {"in_features": 8, "out_features": 2}),
        ])

    def test_diff_keep_weights_cpu_to_gpu_byte_identical(self):
        spec = self.spec()
        old = btui.build_net(spec, seed=777)  # CPU net, stands in for a prior run
        old_bytes = {
            name: p.to_numpy().tobytes()
            for name, p in old.named_parameters().items()
        }
        # A previous GPU run's net: same values, GPU-resident params.
        for param in old.named_parameters().values():
            param.to(bt.Device.GPU)
        new = btui.build_net(spec, seed=999, device="gpu")
        copied = btui.diff_keep_weights(old, new)
        self.assertEqual(sorted(copied), sorted(old_bytes.keys()))
        for name, param in new.named_parameters().items():
            self.assertEqual(param.device, bt.Device.GPU)
            param.to(bt.Device.CPU)  # read back for the byte comparison
            self.assertEqual(param.to_numpy().tobytes(), old_bytes[name],
                             f"{name} diverged across the cpu->gpu reuse copy")
        # Documented side effect: the copy moved the old params to CPU in
        # place (the old net is discarded post-restart).
        for param in old.named_parameters().values():
            self.assertEqual(param.device, bt.Device.CPU)

    def test_run_training_on_gpu(self):
        cfg = btui.DataConfig(source="csv", csv_path=self.csv_path,
                              device="gpu", test_rows=12)
        hp = btui.HyperConfig(lr=1e-2, epochs=1, batch=2, seed=1234)
        shared = btui.SharedState()
        net = btui.run_training(self.spec(), cfg, hp, shared, threading.Event(),
                                has_gpu=True)
        self.assertIsNotNone(net)
        self.assertIsNone(shared.error)
        self.assertFalse(shared.running)
        self.assertEqual(len(shared.losses), 6)
        self.assertTrue(all(np.isfinite(shared.losses)))
        self.assertEqual(len(shared.epoch_stats), 1)
        stats = shared.epoch_stats[0]
        self.assertTrue(np.isfinite(stats["train_mean_loss"]))
        self.assertTrue(np.isfinite(stats["test_mean_loss"]))
        self.assertGreaterEqual(stats["test_accuracy"], 0.0)

    def test_device_mismatch_ready_net_rebuilds(self):
        # A CPU-resident ready net (a CPU-saved Full checkpoint stands in)
        # resumed onto GPU must NOT take the ready path: Adam moments live
        # in an address-keyed map and do NOT move with Tensor.to — the run
        # rebuilds and re-seeds instead, honestly resetting optimizer state.
        spec = self.spec()
        ready = btui.build_net(spec, seed=5)
        ready.train()
        x = bt.Tensor.from_numpy(np.zeros((4, 2), dtype=np.float32))
        y = bt.Tensor.from_numpy(np.array([0, 1, 0, 1], dtype=np.int64))
        for _ in range(3):
            ready.zero_grad()
            loss = bt.softmax_cross_entropy(ready.forward(x), y)
            loss.backward(F32)
            ready.adam_step(1e-2, 0.9, 0.999, 1e-8)
        ready_count = ready.named_parameters()["0.weight"].adam_step_count()
        self.assertEqual(ready_count, 3)
        ready_bytes = {name: p.to_numpy().tobytes()
                       for name, p in ready.named_parameters().items()}

        cfg = btui.DataConfig(source="csv", csv_path=self.csv_path,
                              device="gpu", test_rows=12)
        hp = btui.HyperConfig(lr=1e-2, epochs=1, batch=2, seed=1234)
        shared = btui.SharedState()
        net = btui.run_training(spec, cfg, hp, shared, threading.Event(),
                                seed_weights_from=ready, ready_net=ready,
                                has_gpu=True)
        self.assertIsNotNone(net)
        self.assertIsNone(shared.error)
        # Rebuilt: a different module object, fresh optimizer state
        # (6 batches in the fixture epoch, no carried count), and the ready
        # net itself untouched — its weights were only READ for the seed.
        self.assertIsNot(net, ready)
        new_count = net.named_parameters()["0.weight"].adam_step_count()
        self.assertEqual(new_count, 6)
        self.assertEqual(
            ready.named_parameters()["0.weight"].adam_step_count(), 3)
        for name, param in ready.named_parameters().items():
            self.assertEqual(param.to_numpy().tobytes(), ready_bytes[name])

    def test_same_device_ready_net_resumes(self):
        # Control: a GPU-resident ready net resumed onto GPU DOES take the
        # ready path and continues its step count.
        spec = self.spec()
        ready = btui.build_net(spec, seed=5, device="gpu")
        ready.train()
        x = bt.Tensor.from_numpy(np.zeros((4, 2), dtype=np.float32))
        y = bt.Tensor.from_numpy(np.array([0, 1, 0, 1], dtype=np.int64))
        x.to(bt.Device.GPU)
        y.to(bt.Device.GPU)
        for _ in range(3):
            ready.zero_grad()
            loss = bt.softmax_cross_entropy(ready.forward(x), y)
            loss.backward(F32)
            ready.adam_step(1e-2, 0.9, 0.999, 1e-8)
        cfg = btui.DataConfig(source="csv", csv_path=self.csv_path,
                              device="gpu", test_rows=12)
        hp = btui.HyperConfig(lr=1e-2, epochs=1, batch=2, seed=1234)
        shared = btui.SharedState()
        net = btui.run_training(spec, cfg, hp, shared, threading.Event(),
                                ready_net=ready, has_gpu=True)
        self.assertIsNotNone(net)
        self.assertIsNone(shared.error)
        self.assertIs(net, ready)
        self.assertEqual(
            net.named_parameters()["0.weight"].adam_step_count(), 3 + 6)


class DetectInputShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_path = os.path.join(SCRATCH_BTUI, "detect_3feat.csv")
        write_fixture(
            cls.csv_path,
            "a,b,c,cls\n1.0,2.0,3.0,x\n4.0,5.0,6.0,y\n",
        )

    def test_mnist_shapes_without_loading(self):
        flat = btui.detect_input_shape(btui.DataConfig(source="mnist_flat"))
        self.assertEqual(flat, (784,))
        nchw = btui.detect_input_shape(btui.DataConfig(source="mnist_nchw"))
        self.assertEqual(nchw, (1, 28, 28))

    def test_csv_shape_reads_features(self):
        cfg = btui.DataConfig(source="csv", csv_path=self.csv_path)
        self.assertEqual(btui.detect_input_shape(cfg), (3,))

    def test_csv_bad_path_returns_none(self):
        cfg = btui.DataConfig(
            source="csv",
            csv_path=os.path.join(SCRATCH_BTUI, "__no_such__.csv"),
        )
        self.assertIsNone(btui.detect_input_shape(cfg))
        cfg = btui.DataConfig(source="csv", csv_path="")
        self.assertIsNone(btui.detect_input_shape(cfg))


class SpecToLayerSpecTest(unittest.TestCase):
    def test_roundtrip_types_and_params(self):
        spec = btui.ModelSpec(layers=[
            btui.LayerSpec("Linear", {"in_features": 4, "out_features": 8}),
            btui.LayerSpec("Relu", {}),
            btui.LayerSpec("Dropout", {"p": 0.25}),
            btui.LayerSpec("Linear", {"in_features": 8, "out_features": 3}),
        ])
        net = btui.build_net(spec, seed=42)
        exported = bt.export_spec(net)
        self.assertEqual(exported["type"], "Sequential")
        back, warnings = btui.spec_to_layer_spec(exported)
        self.assertEqual(
            [(layer.type, layer.params) for layer in back.layers],
            [(layer.type, layer.params) for layer in spec.layers],
        )
        self.assertEqual(warnings, [])

    def test_conv_stack_roundtrip(self):
        spec = btui.ModelSpec(layers=[
            btui.LayerSpec("Conv2d", {"in_channels": 1, "out_channels": 4,
                                      "kernel": 3, "stride": 1, "pad": 0}),
            btui.LayerSpec("BatchNorm", {"channels": 4}),
            btui.LayerSpec("MaxPool2d", {"kernel": 2}),
            btui.LayerSpec("Flatten", {}),
            btui.LayerSpec("Gelu", {}),
        ])
        net = btui.build_net(spec, seed=42)
        back, warnings = btui.spec_to_layer_spec(bt.export_spec(net))
        self.assertEqual(
            [(layer.type, layer.params) for layer in back.layers],
            [(layer.type, layer.params) for layer in spec.layers],
        )
        # BatchNorm's momentum/eps have no app-side expression: the import
        # must SAY they were dropped, not drop them silently.
        self.assertEqual(warnings, ["layer 1 BatchNorm momentum/eps"])

    def test_unknown_layer_type_rejected(self):
        bogus = {
            "type": "Sequential",
            "hyperparameters": {},
            "children": [
                {"name": "0", "type": "Linear",
                 "hyperparameters": {"in_features": "4", "out_features": "8",
                                     "bias": "1"},
                 "children": []},
                {"name": "1", "type": "NoSuchLayer",
                 "hyperparameters": {}, "children": []},
            ],
        }
        with self.assertRaises(ValueError) as ctx:
            btui.spec_to_layer_spec(bogus)
        self.assertIn("unknown layer type", str(ctx.exception))


class ArchDataFitTest(unittest.TestCase):
    """Pre-run architecture/data fit validation: refuse a mismatch with a
    friendly reason BEFORE the engine can raise its raw rank error."""

    def _spec(self, first_type, params=None):
        layers = [btui.LayerSpec(first_type, params or {})]
        if first_type == "Conv2d":
            layers.append(btui.LayerSpec("Flatten", {}))
        layers.append(btui.LayerSpec("Linear", {"in_features": 4, "out_features": 2}))
        return btui.ModelSpec(layers=layers)

    def test_validation_table(self):
        nchw = btui.DataConfig(source="mnist_nchw")
        flat = btui.DataConfig(source="mnist_flat")
        csv_cfg = btui.DataConfig(source="csv", csv_path="x.csv")
        conv = self._spec("Conv2d", {"in_channels": 1, "out_channels": 4, "kernel": 3})
        # Conv2d-first needs rank-4 NCHW; flat/CSV are refused with the
        # shape contract, not a dataset prescription.
        reason = btui.validate_arch_data_fit(conv, flat)
        self.assertIsNotNone(reason)
        self.assertIn("Conv2d", reason)
        self.assertIn("rank-4 NCHW", reason)
        self.assertIsNotNone(btui.validate_arch_data_fit(conv, csv_cfg))
        # Conv2d-first with NCHW is allowed.
        self.assertIsNone(btui.validate_arch_data_fit(conv, nchw))
        # Linear-first works with flat and CSV, not with 4-D data.
        linear = self._spec("Linear", {"in_features": 784, "out_features": 8})
        self.assertIsNone(btui.validate_arch_data_fit(linear, flat))
        self.assertIsNone(btui.validate_arch_data_fit(linear, csv_cfg))
        reason = btui.validate_arch_data_fit(linear, nchw)
        self.assertIsNotNone(reason)
        self.assertIn("Linear", reason)
        self.assertIn("rank-2", reason)
        # Flatten reshapes any contiguous input to rank 2, so it accepts both.
        flatten = self._spec("Flatten")
        self.assertIsNone(btui.validate_arch_data_fit(flatten, flat))
        self.assertIsNone(btui.validate_arch_data_fit(flatten, nchw))
        # Rank-agnostic first layers pass either way.
        relu = self._spec("Relu")
        self.assertIsNone(btui.validate_arch_data_fit(relu, nchw))
        self.assertIsNone(btui.validate_arch_data_fit(relu, flat))
        # Empty spec.
        self.assertEqual(
            btui.validate_arch_data_fit(btui.ModelSpec(), flat),
            "model has no layers",
        )

    def test_trainjob_refuses_mismatch(self):
        conv = self._spec("Conv2d", {"in_channels": 1, "out_channels": 4, "kernel": 3})
        with self.assertRaises(ValueError) as ctx:
            btui.TrainJob(conv, btui.DataConfig(source="mnist_flat"),
                          btui.HyperConfig(), btui.SharedState())
        self.assertIn("rank-4 NCHW", str(ctx.exception))


class VgraphMathTest(unittest.TestCase):
    """_vgraph maps a series to `height` rows of eighth-block bars,
    bottom row = base, normalization mirroring the sparkline family:
    zero renders blank, nonzero floors at one level, flat-busy sits mid."""

    def test_empty_and_zero_series(self):
        self.assertEqual(btui._vgraph([], 3), ["", "", ""])
        rows = btui._vgraph([0, 0, 0], 2)
        self.assertEqual(rows, ["   ", "   "])
        self.assertEqual(btui._vgraph([], 0), [])

    def test_flat_busy_renders_mid_height_not_blank(self):
        rows = btui._vgraph([5, 5, 5, 5], 4)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(len(row) == 4 for row in rows))
        # Flat nonzero must read as activity: the bottom half is full.
        self.assertEqual(rows, ["    ", "    ", "████", "████"])

    def test_rising_series_ascends(self):
        # h=1: each column is its own normalized level.
        self.assertEqual(btui._vgraph([1, 2, 3, 4], 1), ["▁▂▅█"])
        # h=2: the top row carries only the partial blocks of tall columns.
        rows = btui._vgraph([0, 1, 2, 3, 4], 2)
        self.assertEqual(rows, ["   ▄█", " ▄███"])

    def test_row_lengths_match_sample_count(self):
        rows = btui._vgraph([0, 1, 0, 2, 0, 3], 3)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(row) == 6 for row in rows))


class DgraphMathTest(unittest.TestCase):
    """_dgraph renders a centered delta graph with a RESERVED baseline row:
    the middle row is always the centerline ('─'), positive deltas bar UP
    from it (bottom-up fill), negative deltas bar DOWN from it (top-down
    fill, partial cells hugging the line so bars never float). Half blocks
    (▄/▀) are one level, full blocks (█) two, per side-cell. The first
    visible sample deltas against the window mean. Heights are scaled by
    the largest |delta| under a shared per-side budget."""

    def test_empty_and_flat(self):
        self.assertEqual(btui._dgraph([], 3), ["", "", ""])
        self.assertEqual(btui._dgraph([], 0), [])
        # Flat series: centerline row only, sides blank.
        rows = btui._dgraph([2.0, 2.0, 2.0, 2.0], 5)
        self.assertEqual(rows, ["    ", "    ", "────", "    ", "    "])

    def test_rising_series_bars_above_center(self):
        # [1,2,3,4]: mean 2.5; deltas [-1.5, +1, +1, +1]; scale 1.5,
        # side budget 4 levels (2 rows). Column 0 dives two cells down
        # (full + partial hugging the line); columns 1-3 rise full+partial.
        rows = btui._dgraph([1.0, 2.0, 3.0, 4.0], 5)
        self.assertEqual(rows, [" ▄▄▄", " ███", "────", "█   ", "█   "])

    def test_falling_series_bars_below_center(self):
        rows = btui._dgraph([4.0, 3.0, 2.0, 1.0], 5)
        self.assertEqual(rows, ["█   ", "█   ", "────", " ███", " ▀▀▀"])

    def test_small_deltas_hug_the_centerline(self):
        # The readability pin: sub-full steps sit DIRECTLY adjacent to the
        # line — a step up as ▄ in the row above (bottom half touching the
        # line), a step down as ▀ in the row below (top half touching it).
        # Series [1, 1.5, 1.0]: c0 anchors below-mean (▀), c1 steps up (█ at
        # this window's scale), c2 steps down (█); no bar floats a row away
        # from the baseline.
        rows = btui._dgraph([1.0, 1.5, 1.0], 3)
        self.assertEqual(rows, [" █ ", "───", "▀ █"])

    def test_row_lengths_match_sample_count(self):
        rows = btui._dgraph([1.0, 0.5, 2.0, 0.1], 3)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(row) == 4 for row in rows))
        # The baseline row is the centerline in every finite column.
        self.assertEqual(rows[1], "────")

    def test_nonfinite_marks_center(self):
        rows = btui._dgraph([1.0, float("nan"), 2.0], 3)
        self.assertEqual(rows[1], "─??")


class PauseTest(unittest.TestCase):
    """p pauses between batches: the loss ring freezes, resume continues the
    SAME run (epoch numbering never resets), and s still stops fully even
    while paused."""

    @classmethod
    def setUpClass(cls):
        cls.csv_path = os.path.join(SCRATCH_BTUI, "pause_12row.csv")
        write_fixture(cls.csv_path, CSV_12ROW)

    def _wait_until(self, predicate, deadline_seconds=10.0):
        deadline = time.perf_counter() + deadline_seconds
        while time.perf_counter() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_pause_freezes_resume_continues_stop_works(self):
        shared = btui.SharedState()
        spec = btui.ModelSpec(layers=[
            btui.LayerSpec("Linear", {"in_features": 2, "out_features": 8}),
            btui.LayerSpec("Relu", {}),
            btui.LayerSpec("Linear", {"in_features": 8, "out_features": 2}),
        ])
        cfg = btui.DataConfig(source="csv", csv_path=self.csv_path)
        hp = btui.HyperConfig(lr=1e-2, epochs=2000, batch=2, seed=1234)
        job = btui.TrainJob(spec, cfg, hp, shared)
        job.start()
        try:
            self.assertTrue(self._wait_until(lambda: shared.batches_done >= 2),
                            "run never produced batches")
            shared.set_paused(True)
            # Batch-boundary semantics: a batch already in flight when p
            # lands still completes and records. Let it settle, THEN freeze.
            time.sleep(0.15)
            frozen = shared.batches_done
            time.sleep(0.3)
            self.assertEqual(shared.batches_done, frozen,
                             "loss ring advanced while paused")
            # The ring caps at loss_capacity entries: on a fast runner the
            # settle window can complete more batches than the ring holds, so
            # the length tracks frozen only up to the capacity.
            self.assertEqual(len(shared.losses), min(frozen, shared.loss_capacity))
            shared.set_paused(False)
            self.assertTrue(self._wait_until(lambda: shared.batches_done > frozen),
                            "run did not resume after unpause")
            # s lands even while paused (the wait loop polls the stop event).
            shared.set_paused(True)
            job.stop()
            job.join(timeout=30.0)
            self.assertFalse(job.is_alive(), "stop while paused did not land")
            self.assertFalse(shared.running)
            self.assertFalse(shared.paused, "end_run must clear the paused flag")
            self.assertIsNone(shared.error)
            # Epoch numbering never reset: consecutive, starting at 1.
            epochs_seen = [s["epoch"] for s in shared.epoch_stats]
            self.assertEqual(epochs_seen, list(range(1, len(epochs_seen) + 1)))
            self.assertTrue(len(epochs_seen) >= 1)
        finally:
            job.stop()
            job.join(timeout=30.0)


class GpuMemoryTest(unittest.TestCase):
    def test_gpu_memory_matches_probe(self):
        mem = bt.gpu_memory()
        if btui.probe_gpu():
            self.assertIsInstance(mem, dict)
            self.assertGreater(mem["total_bytes"], 0)
            self.assertGreaterEqual(mem["used_bytes"], 0)
            self.assertLessEqual(mem["used_bytes"], mem["total_bytes"])
        else:
            self.assertIsNone(mem)


class GpuUtilTest(unittest.TestCase):
    """_gpu_utilization parses nvidia-smi's CSV probe and maps every failure
    mode to None: missing binary, timeout, non-zero exit, unparseable or
    empty output."""

    class _Proc:
        def __init__(self, returncode, stdout):
            self.returncode = returncode
            self.stdout = stdout

    def setUp(self):
        # btui.subprocess IS the stdlib module; save/restore, never leak.
        self._real = btui.subprocess.run

    def tearDown(self):
        btui.subprocess.run = self._real

    def _patch(self, fn):
        btui.subprocess.run = fn

    def test_parses_first_gpu_percent(self):
        self._patch(lambda *a, **k: GpuUtilTest._Proc(0, "37\n"))
        self.assertEqual(btui._gpu_utilization(), 37)
        # Multi-GPU hosts list one line per device; the dashboard reports
        # device 0, matching gpu_memory's contract.
        self._patch(lambda *a, **k: GpuUtilTest._Proc(0, "37\n82\n"))
        self.assertEqual(btui._gpu_utilization(), 37)

    def test_failure_modes_return_none(self):
        def missing(*a, **k):
            raise FileNotFoundError("nvidia-smi")
        self._patch(missing)
        self.assertIsNone(btui._gpu_utilization())

        def slow(*a, **k):
            raise btui.subprocess.TimeoutExpired("nvidia-smi", 2)
        self._patch(slow)
        self.assertIsNone(btui._gpu_utilization())

        self._patch(lambda *a, **k: GpuUtilTest._Proc(1, ""))
        self.assertIsNone(btui._gpu_utilization())

        self._patch(lambda *a, **k: GpuUtilTest._Proc(0, "not-a-number\n"))
        self.assertIsNone(btui._gpu_utilization())

        self._patch(lambda *a, **k: GpuUtilTest._Proc(0, ""))
        self.assertIsNone(btui._gpu_utilization())


class GilReleaseHeartbeatTest(unittest.TestCase):
    """The blocking bindings (item/evaluate/to_numpy/write_numpy) drop the
    GIL across their sync regions, so a polling thread stays responsive
    while the training thread waits on the pool — and can actually observe
    the pool mid-work instead of only after it goes idle."""

    def test_pool_polling_heartbeat_during_training(self):
        rng = np.random.RandomState(0)
        features = bt.Tensor.from_numpy(rng.randn(4096, 784).astype(np.float32))
        labels = bt.Tensor.from_numpy(
            rng.randint(0, 10, size=4096).astype(np.int64)
        )
        bt.set_seed(20261119)
        net = bt.Sequential([
            bt.Linear(784, 512, F32, CPU),
            bt.Relu(),
            bt.Linear(512, 512, F32, CPU),
            bt.Relu(),
            bt.Linear(512, 10, F32, CPU),
        ])
        net.train()
        stop = threading.Event()

        def train():
            # Both GIL-released stretches repeat: the per-step item() sync,
            # and a full evaluate pass (a long, continuously pool-busy call).
            loader = bt.DataLoader(features, labels, 128, False, None, False)
            while not stop.is_set():
                net.zero_grad()
                logits = net.forward(features)
                loss = bt.softmax_cross_entropy(logits, labels)
                loss.backward(F32)
                net.adam_step(1e-3, 0.9, 0.999, 1e-8)
                loss.item()  # the blocking sync the GIL release covers
                bt.evaluate(net, loader)

        worker = threading.Thread(target=train)
        worker.start()
        max_poll_seconds = 0.0
        saw_activity = False
        deadline = time.perf_counter() + 60.0
        try:
            while not saw_activity and time.perf_counter() < deadline:
                poll_start = time.perf_counter()
                snap = bt.pool_snapshot()
                max_poll_seconds = max(max_poll_seconds,
                                       time.perf_counter() - poll_start)
                if snap["pending_tasks_total"] > 0 or any(
                    w["status"] != "Idle" for w in snap["workers"]
                ):
                    saw_activity = True
                time.sleep(0.1)  # ~10 Hz, the Pool panel's poll rate
        finally:
            stop.set()
            worker.join(timeout=120.0)
        self.assertFalse(worker.is_alive(), "training thread did not stop")
        self.assertTrue(
            saw_activity,
            "no snapshot during training showed a busy pool "
            "(pending > 0 or a non-idle worker)",
        )
        self.assertLess(
            max_poll_seconds, 1.0,
            f"pool_snapshot poll stalled behind the GIL for {max_poll_seconds:.3f}s",
        )


if __name__ == "__main__":
    unittest.main()

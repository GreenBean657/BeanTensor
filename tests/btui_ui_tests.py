#!/usr/bin/env python3
"""btui_ui_tests.py — stdlib-unittest suite for the btui curses front-end.

Drives the dependency-injected BufferScreen backend only — no real terminal
needed. Covers the form-validation table (per layer type and per
hyperparameter: bad input produces the log-strip error text, leaves state
unchanged, and never raises), layer add/delete, and a headless end-to-end
run that builds Linear(2,2)+Relu+Linear(2,2) over a tiny CSV fixture, runs
one epoch, and asserts the final screen carries the Training box and a loss.

Run directly (bindings on PYTHONPATH) or through the btui_ui ctest:
    PYTHONPATH=scratch/build-tui/Bindings python3 -m unittest tests.btui_ui_tests
"""

import os
import re
import subprocess
import sys
import threading
import time
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import beantensor as bt  # noqa: E402

import btui  # noqa: E402

SCRATCH_BTUI = os.path.join(REPO_ROOT, "scratch", "btui")

# 16 rows, 2 features, 2 classes, separable; batch 8 -> 2 batches/epoch.
CSV_16ROW = (
    "x1,x2,cls\n"
    + "\n".join(
        ["0.0,0.0,a", "0.1,0.0,a", "0.0,0.1,a", "0.2,0.1,a",
         "0.1,0.2,a", "0.0,0.2,a", "0.2,0.0,a", "0.2,0.2,a",
         "1.0,1.0,b", "1.0,0.9,b", "0.9,1.0,b", "1.1,1.0,b",
         "1.0,1.1,b", "0.9,0.9,b", "1.1,0.9,b", "0.9,1.1,b"]
    )
    + "\n"
)


def type_text(app, text):
    for ch in text:
        app.handle_key(ch)


def feed(app, *keys):
    for key in keys:
        app.handle_key(key)


class BufferScreenTest(unittest.TestCase):
    def test_addstr_text_and_clipping(self):
        screen = btui.BufferScreen(10, 20)
        screen.addstr(0, 0, "hello")
        screen.addstr(1, 18, "clipped-write")  # must not raise at the edge
        screen.addstr(9, 19, "x")  # bottom-right corner must not raise
        text = screen.text()
        self.assertIn("hello", text.splitlines()[0])
        screen.erase()
        self.assertNotIn("hello", screen.text())


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.screen = btui.BufferScreen(34, 110)
        self.app = btui.BtuiApp(self.screen)

    def tearDown(self):
        self.app.shutdown()

    def add_layer(self, type_name):
        """Drive the add-layer flow, accepting every default param."""
        before = len(self.app.spec.layers)
        self.app.handle_key("a")
        type_text(self.app, type_name)
        self.app.handle_key("ENTER")
        while self.app.form is not None:
            self.app.handle_key("ENTER")  # empty keeps the default
        self.assertEqual(len(self.app.spec.layers), before + 1)
        self.assertEqual(self.app.spec.layers[-1].type, type_name)
        return self.app.spec.layers[-1]


class FormValidationTest(AppTestCase):
    def test_layer_param_table_and_bad_input(self):
        expected = {
            "Linear": ["in_features", "out_features"],
            "Conv2d": ["in_channels", "out_channels", "kernel", "stride", "pad"],
            "BatchNorm": ["channels"],
            "MaxPool2d": ["kernel"],
            "Dropout": ["p"],
            "Flatten": [],
            "Relu": [],
            "Gelu": [],
        }
        for type_name, params in expected.items():
            with self.subTest(type=type_name):
                layer = self.add_layer(type_name)
                self.assertEqual(list(layer.params.keys()), params)
                if not params:
                    continue
                before = dict(layer.params)
                # The selection follows the appended layer; the param cursor
                # starts on the first param. ENTER opens its edit form.
                self.app.handle_key("ENTER")
                self.assertIsNotNone(self.app.form)
                type_text(self.app, "not_a_number")
                self.app.handle_key("ENTER")
                self.assertIsNone(self.app.form)
                self.assertEqual(dict(self.app.spec.layers[-1].params), before)
                self.assertTrue(
                    any("invalid" in line and params[0] in line
                        for line in self.app.log_lines),
                    f"log must carry the invalid-{params[0]} line: {self.app.log_lines}",
                )

    def test_bad_value_table(self):
        cases = [
            ("Linear", 0, "-3"),      # in_features must be >= 1
            ("Linear", 1, "0"),       # out_features must be >= 1
            ("MaxPool2d", 0, "0"),    # kernel must be >= 1
            ("Dropout", 0, "1.5"),    # p must be in [0, 1)
            ("Dropout", 0, "-0.1"),
        ]
        for type_name, param_index, bad_text in cases:
            with self.subTest(type=type_name, param=param_index, bad=bad_text):
                screen = btui.BufferScreen(34, 110)
                app = btui.BtuiApp(screen)
                try:
                    before_count = len(app.log_lines)
                    app.handle_key("a")
                    type_text(app, type_name)
                    app.handle_key("ENTER")
                    while app.form is not None:
                        app.handle_key("ENTER")
                    layer = app.spec.layers[-1]
                    before = dict(layer.params)
                    feed(app, *["RIGHT"] * param_index, "ENTER")
                    type_text(app, bad_text)
                    app.handle_key("ENTER")
                    self.assertEqual(dict(app.spec.layers[-1].params), before)
                    new_lines = app.log_lines[before_count:]
                    self.assertTrue(any("invalid" in line for line in new_lines),
                                    f"no invalid line logged: {new_lines}")
                finally:
                    app.shutdown()

    def test_unknown_layer_type_is_rejected(self):
        self.app.handle_key("a")
        type_text(self.app, "Bogus")
        self.app.handle_key("ENTER")
        self.assertEqual(self.app.spec.layers, [])
        self.assertTrue(
            any("unknown layer type" in line for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )

    def test_hyperparam_form_validation(self):
        # Focus the params panel and edit lr (field index 8) with garbage.
        feed(self.app, "TAB", *["DOWN"] * 8, "ENTER")
        type_text(self.app, "abc")
        self.app.handle_key("ENTER")
        self.assertEqual(self.app.hp_cfg.lr, 1e-3)
        self.assertTrue(
            any("invalid lr" in line for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )
        # A good value commits and marks the config pending.
        self.app.handle_key("ENTER")
        type_text(self.app, "0.01")
        self.app.handle_key("ENTER")
        self.assertEqual(self.app.hp_cfg.lr, 0.01)
        self.assertTrue(self.app.shared.pending)
        # epochs (field index 12) rejects 0.
        feed(self.app, *["DOWN"] * 4, "ENTER")
        type_text(self.app, "0")
        self.app.handle_key("ENTER")
        self.assertEqual(self.app.hp_cfg.epochs, 3)
        self.assertTrue(
            any("invalid epochs" in line for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )

    def test_data_source_enum_validation(self):
        feed(self.app, "TAB", "ENTER")  # first field is the data source
        type_text(self.app, "bogus")
        self.app.handle_key("ENTER")
        self.assertEqual(self.app.data_cfg.source, btui.MNIST_FLAT)
        self.assertTrue(
            any("invalid data_source" in line for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )
        self.app.handle_key("ENTER")
        type_text(self.app, "csv")
        self.app.handle_key("ENTER")
        self.assertEqual(self.app.data_cfg.source, btui.CSV)

    def test_csv_policy_fields(self):
        # csv nulls (index 3) and csv normalize (index 4) are validated enums.
        feed(self.app, "TAB", "DOWN", "DOWN", "DOWN", "ENTER")
        type_text(self.app, "FillEverything")
        self.app.handle_key("ENTER")
        self.assertEqual(self.app.data_cfg.csv_nulls, "KeepNaN")
        self.assertTrue(
            any("invalid csv_nulls" in line for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )
        self.app.handle_key("ENTER")
        type_text(self.app, "FillMean")
        self.app.handle_key("ENTER")
        self.assertEqual(self.app.data_cfg.csv_nulls, "FillMean")

        feed(self.app, "DOWN", "ENTER")
        type_text(self.app, "ZScore")
        self.app.handle_key("ENTER")
        self.assertEqual(self.app.data_cfg.csv_normalize, "None")
        self.assertTrue(
            any("invalid csv_normalize" in line for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )
        self.app.handle_key("ENTER")
        type_text(self.app, "Standardize")
        self.app.handle_key("ENTER")
        self.assertEqual(self.app.data_cfg.csv_normalize, "Standardize")

    def test_lr_eps_reject_nonfinite(self):
        # float("inf") > 0.0 is True, so inf spellings must be rejected
        # explicitly — a run with lr=inf silently trains to NaN otherwise.
        for field_index, slug in ((8, "lr"), (11, "eps")):
            for bad in ("inf", "1e999"):
                with self.subTest(field=slug, bad=bad):
                    screen = btui.BufferScreen(34, 110)
                    app = btui.BtuiApp(screen)
                    try:
                        before = app.hp_cfg
                        feed(app, "TAB", *["DOWN"] * field_index, "ENTER")
                        type_text(app, bad)
                        app.handle_key("ENTER")
                        self.assertEqual(app.hp_cfg, before)
                        self.assertTrue(
                            any(f"invalid {slug}" in line for line in app.log_lines),
                            f"log lines: {app.log_lines}",
                        )
                    finally:
                        app.shutdown()

    def test_esc_cancels_without_changes(self):
        self.add_layer("Linear")
        before = dict(self.app.spec.layers[-1].params)
        self.app.handle_key("ENTER")
        type_text(self.app, "99")
        self.app.handle_key("ESC")
        self.assertIsNone(self.app.form)
        self.assertEqual(dict(self.app.spec.layers[-1].params), before)


class LayerEditingTest(AppTestCase):
    def test_add_edit_delete(self):
        self.add_layer("Linear")
        self.add_layer("Relu")
        self.add_layer("Linear")
        self.assertEqual(
            [layer.type for layer in self.app.spec.layers],
            ["Linear", "Relu", "Linear"],
        )
        # Edit the selected (last) layer's out_features via LEFT/RIGHT + form.
        feed(self.app, "RIGHT", "ENTER")
        type_text(self.app, "5")
        self.app.handle_key("ENTER")
        self.assertEqual(self.app.spec.layers[-1].params["out_features"], 5)
        # Delete the selected layer; the selection clamps to the new tail.
        self.app.handle_key("d")
        self.assertEqual(
            [layer.type for layer in self.app.spec.layers], ["Linear", "Relu"]
        )
        self.assertTrue(self.app.shared.pending)


class HeadlessEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_path = os.path.join(SCRATCH_BTUI, "ui_fixture_16row.csv")
        os.makedirs(os.path.dirname(cls.csv_path), exist_ok=True)
        with open(cls.csv_path, "w", encoding="utf-8") as fh:
            fh.write(CSV_16ROW)
        cls.keys_path = os.path.join(SCRATCH_BTUI, "ui_end_to_end.keys")

    def test_headless_run_to_idle(self):
        keys = [
            # Data: tiny CSV fixture, 16 train rows, batch 8 (2 batches).
            "SET:data_source=csv",
            f"SET:csv_path={self.csv_path}",
            "SET:train_rows=16",
            "SET:batch=8",
            # lr and epochs go through the real form path (field indices
            # 8 and 12 in the params panel) to prove the form drive.
            "TAB", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN",
            "ENTER", "TYPE:0.01", "ENTER",
            "DOWN", "DOWN", "DOWN", "DOWN",
            "ENTER", "TYPE:1", "ENTER",
            "TAB",
            # Build Linear(2,2) + Relu + Linear(2,2); ENTERs accept defaults.
            "a", "TYPE:Linear", "ENTER", "ENTER", "ENTER",
            "a", "TYPE:Relu", "ENTER",
            "a", "TYPE:Linear", "ENTER", "ENTER", "ENTER",
            "r",
        ]
        with open(self.keys_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(keys) + "\n")
        code, text = btui.run_headless(self.keys_path)
        self.assertEqual(code, 0)
        self.assertIn("Training", text, f"final screen:\n{text}")
        self.assertRegex(text, r"last=\d+\.\d+")
        self.assertIn("epoch 1/1", text, f"final screen:\n{text}")
        self.assertIn("CPU pool", text, f"final screen:\n{text}")


class ReorderAndDisplayTest(AppTestCase):
    def test_reorder_layers_with_j_and_k(self):
        self.add_layer("Linear")
        self.add_layer("Relu")
        self.add_layer("Dropout")
        self.assertEqual(
            [layer.type for layer in self.app.spec.layers],
            ["Linear", "Relu", "Dropout"],
        )
        # Selection follows the last add (Dropout at index 2); k moves it up.
        self.app.handle_key("k")
        self.assertEqual(
            [layer.type for layer in self.app.spec.layers],
            ["Linear", "Dropout", "Relu"],
        )
        self.assertEqual(self.app.layer_sel, 1)
        self.assertTrue(self.app.shared.pending)
        # j moves it back down (restoring the original order); the second j
        # clamps silently at the bottom edge.
        self.app.handle_key("j")
        self.app.handle_key("j")
        self.assertEqual(
            [layer.type for layer in self.app.spec.layers],
            ["Linear", "Relu", "Dropout"],
        )
        self.assertEqual(self.app.layer_sel, 2)
        self.app.handle_key("k")
        self.app.handle_key("k")
        self.app.handle_key("k")  # clamped at the top
        self.assertEqual(self.app.layer_sel, 0)
        self.assertEqual(
            [layer.type for layer in self.app.spec.layers],
            ["Dropout", "Linear", "Relu"],
        )

    def test_device_indicator_visible(self):
        app = btui.BtuiApp(btui.BufferScreen(34, 110), has_gpu=False)
        try:
            app.redraw()
            self.assertIn("device: cpu (no gpu in this build)",
                          app.screen.text())
        finally:
            app.shutdown()


class DeviceFieldTest(AppTestCase):
    def setUp(self):
        self.screen = btui.BufferScreen(34, 110)
        self.app = btui.BtuiApp(self.screen, has_gpu=False)

    def test_device_row_greyed_and_locked_without_gpu(self):
        # The device field (last row, index 15) shows the no-gpu message and
        # ENTER logs it instead of opening a form; the value stays cpu.
        feed(self.app, "TAB", *["DOWN"] * 15, "ENTER")
        self.assertIsNone(self.app.form)
        self.assertTrue(
            any("no gpu in this build" in line for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )
        self.assertEqual(self.app.data_cfg.device, "cpu")
        # The headless SET path rejects it too, with the same message.
        self.app.apply_setting("device", "gpu")
        self.assertEqual(self.app.data_cfg.device, "cpu")
        self.assertTrue(
            any("invalid device" in line and "no gpu in this build" in line
                for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )

    def test_device_field_editable_with_gpu(self):
        app = btui.BtuiApp(btui.BufferScreen(34, 110), has_gpu=True)
        try:
            feed(app, "TAB", *["DOWN"] * 15, "ENTER")
            self.assertIsNotNone(app.form)
            type_text(app, "gpu")
            app.handle_key("ENTER")
            self.assertEqual(app.data_cfg.device, "gpu")
            self.assertTrue(app.shared.pending)
            app.handle_key("ENTER")
            type_text(app, "tpu")
            app.handle_key("ENTER")
            self.assertEqual(app.data_cfg.device, "gpu")
            self.assertTrue(
                any("invalid device" in line for line in app.log_lines),
                f"log lines: {app.log_lines}",
            )
        finally:
            app.shutdown()


class ArchFitRefusalTest(AppTestCase):
    def test_r_refuses_conv2d_first_on_flat_data(self):
        self.add_layer("Conv2d")  # defaults in=1 out=4 kernel=3
        self.add_layer("Flatten")
        self.add_layer("Linear")
        # Data source is still the default mnist_flat.
        self.app.handle_key("r")
        self.assertIsNone(self.app.job, "the run must be refused, not started")
        self.assertTrue(
            any("first layer Conv2d needs rank-4 NCHW" in line
                and "batch, channels, height, width" in line
                for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )
        # Switching the source to mnist_nchw passes the fit check (it then
        # fails later only for want of the IDX files, not the fit check).
        self.app.apply_setting("data_source", "mnist_nchw")
        self.app.handle_key("r")
        self.assertFalse(
            any("first layer Conv2d needs rank-4 NCHW" in line
                for line in self.app.log_lines[-2:]),
            f"log lines: {self.app.log_lines}",
        )


class ColorTest(AppTestCase):
    def test_error_line_carries_error_attr(self):
        # Force a run failure: a valid Linear stack pointed at a missing CSV.
        self.app.apply_setting("data_source", "csv")
        self.app.apply_setting(
            "csv_path", os.path.join(SCRATCH_BTUI, "__missing__.csv"))
        self.add_layer("Linear")
        self.app.handle_key("r")
        deadline = time.perf_counter() + 30.0
        while self.app.busy and time.perf_counter() < deadline:
            self.app.redraw()
            time.sleep(0.005)
        self.assertFalse(self.app.busy)
        self.app.redraw()
        rows = self.app.screen.text().splitlines()
        row = next(i for i, line in enumerate(rows) if "run failed:" in line)
        col = rows[row].index("run failed:")
        self.assertEqual(self.app.screen.attr_at(row, col), btui.STYLE_ERROR)
        # The sibling log-strip row (the line above) is a non-error line
        # ("run started") and must render dim, not error-red.
        screen_rows, _cols = self.app.screen.getmaxyx()
        sibling = rows[screen_rows - 3]
        self.assertTrue(sibling.strip(), "expected a non-error log line above")
        first_col = sibling.index(sibling.strip()[0])
        self.assertEqual(self.app.screen.attr_at(screen_rows - 3, first_col),
                         btui.STYLE_DIM)

    def test_pool_cells_carry_status_attrs(self):
        self.app.redraw()
        rows = self.app.screen.text().splitlines()
        row = next(i for i, line in enumerate(rows) if "#00 " in line)
        tag_col = rows[row].index("#00 ") + 4
        tag = rows[row][tag_col:].split()[0]
        expected = {
            "IDLE": btui.STYLE_IDLE,
            "OWN": btui.STYLE_OWN,
            "STEAL": btui.STYLE_STEAL,
            "HELP": btui.STYLE_HELP,
            "WAIT": btui.STYLE_WAIT,
        }
        self.assertIn(tag, expected, f"unexpected pool tag {tag!r}")
        self.assertEqual(self.app.screen.attr_at(row, tag_col), expected[tag])

    def test_header_state_carries_state_attr(self):
        self.app.job = SlowJob()
        self.app.shared.begin_run()
        self.app.redraw()
        rows = self.app.screen.text().splitlines()
        col = rows[0].index("running")
        self.assertEqual(self.app.screen.attr_at(0, col), btui.STYLE_STATE_RUNNING)
        self.app.job.release()

    def test_monochrome_records_no_attrs(self):
        screen = btui.BufferScreen(34, 110, color=False)
        app = btui.BtuiApp(screen)
        try:
            app.redraw()
            self.assertTrue(
                all(attr == 0 for attr in screen.all_attrs()),
                "monochrome fallback must record attr 0 everywhere",
            )
        finally:
            app.shutdown()


def _fake_pool_snapshot(completed):
    """A pool_snapshot-shaped dict with the given per-worker
    tasks_completed counters (everyone instantaneously Idle, the common
    microtask case)."""
    return {
        "worker_count": len(completed),
        "pending_tasks_total": 0,
        "stopping": False,
        "workers": [
            {
                "worker_id": i, "status": "Idle", "pinned_group": 0,
                "task_sequence": 0, "task_priority": "?",
                "task_running_ms": 0, "futures_awaited": 0,
                "tasks_completed": count, "tasks_stolen": 0,
                "tasks_helped": 0, "local_queue_depth": 0,
            }
            for i, count in enumerate(completed)
        ],
    }


class PoolDeltaActivityTest(AppTestCase):
    """Instantaneous worker status structurally misses microsecond pool
    tasks; the Pool panel must derive activity from tasks_completed
    deltas between polls instead."""

    def _poll_with(self, snapshot):
        self._snapshot = snapshot
        self.app._pool_at = 0.0  # force a refetch on the next redraw
        self.app.redraw()

    def setUp(self):
        super().setUp()
        self._real_snapshot = btui.bt.pool_snapshot
        self._snapshot = _fake_pool_snapshot([0] * 6)
        btui.bt.pool_snapshot = lambda: self._snapshot

    def tearDown(self):
        btui.bt.pool_snapshot = self._real_snapshot
        super().tearDown()

    def test_first_snapshot_renders_instantaneous_resting(self):
        self.app.redraw()  # first poll: no previous snapshot
        rows = self.screen.text().splitlines()
        header_row = next(i for i, line in enumerate(rows) if "workers=" in line)
        # No interval readout yet, and workers render resting (Idle dim).
        self.assertNotIn("done=", rows[header_row])
        grid_row = next(i for i, line in enumerate(rows) if "#00 " in line)
        col = rows[grid_row].index("#00 ")
        self.assertEqual(self.screen.attr_at(grid_row, col), btui.STYLE_IDLE)
        self.assertNotIn("+", rows[grid_row])

    def test_delta_workers_render_active_with_counts(self):
        self.app.redraw()  # first poll (baseline)
        advanced = [0, 0, 0, 12, 0, 1]
        self._poll_with(_fake_pool_snapshot(advanced))
        rows = self.screen.text().splitlines()
        # Header carries the interval total (12 + 1).
        header_row = next(i for i, line in enumerate(rows) if "workers=" in line)
        self.assertIn("done=13", rows[header_row])
        grid_row = next(i for i, line in enumerate(rows) if "#00 " in line)
        # Resting workers RECEDE: dim lowercase "idle", never the IDLE shout.
        col0 = rows[grid_row].index("#00 ")
        self.assertIn("#00 idle", rows[grid_row])
        self.assertEqual(self.screen.attr_at(grid_row, col0), btui.STYLE_DIM)
        # Active workers DOMINATE: bright busy style, ~count, and crucially
        # the string IDLE never appears in an active cell.
        col3 = rows[grid_row].index("#03 ")
        self.assertIn("~12", rows[grid_row])
        self.assertNotIn("IDLE", rows[grid_row])
        self.assertEqual(self.screen.attr_at(grid_row, col3), btui.STYLE_BUSY)
        row5 = next(i for i, line in enumerate(rows) if "#05 " in line)
        col5 = rows[row5].index("#05 ")
        self.assertIn("~1", rows[row5])
        self.assertEqual(self.screen.attr_at(row5, col5), btui.STYLE_BUSY)
        # A third poll with no progress returns everyone to resting.
        self._poll_with(_fake_pool_snapshot(advanced))
        rows = self.screen.text().splitlines()
        grid_row = next(i for i, line in enumerate(rows) if "#00 " in line)
        col3 = rows[grid_row].index("#03 ")
        self.assertNotIn("~", rows[grid_row])
        self.assertEqual(self.screen.attr_at(grid_row, col3), btui.STYLE_DIM)
        header_row = next(i for i, line in enumerate(rows) if "workers=" in line)
        self.assertIn("done=0", rows[header_row])


class DeltaGraphPanelTest(AppTestCase):
    """The key readability pin: a falling loss sequence puts bar chars BELOW
    the dim centerline row; a rising one puts them ABOVE."""

    def _training_graph_rows(self):
        rows = self.screen.text().splitlines()
        title_row = next(i for i, line in enumerate(rows) if "─ Training" in line)
        # The Training box is the right-hand panel. The graph block starts
        # three rows below the title (readout, mean, then the graph); the
        # baseline row is LOCATED BY CONTENT — the interior row carrying a
        # run of the centerline glyph — so test geometry changes can't
        # silently retarget the pin.
        interior = rows[title_row].rindex("┌") + 2
        graph_top = title_row + 3
        center = next(
            r for r in range(graph_top, graph_top + 9)
            if "───" in rows[r][interior:]
        )
        return rows, graph_top, center, interior  # (rows, graph_top, center, col0)

    def _feed_losses(self, losses):
        self.app.shared.begin_run()
        for loss in losses:
            self.app.shared.record_batch(loss)
        self.app.redraw()

    def test_falling_losses_render_below_centerline(self):
        # Column 0 deltas against the window mean BY DESIGN, so the pin
        # counts columns >= 1 (pure step-to-step deltas). Bars hang STRICTLY
        # below the baseline row in the new encoding.
        self._feed_losses([0.8, 0.8, 0.6, 0.4, 0.2])
        rows, _top, center, col0 = self._training_graph_rows()
        blocks = "▄▀█"
        above = sum(rows[r][col0:].count(b) for b in blocks
                    for r in range(_top, center))
        below = sum(rows[r][col0:].count(b) for b in blocks
                    for r in range(center + 1, center + 4))
        self.assertGreaterEqual(below, 3,
                                f"falling losses must bar below the center:\n"
                                f"{self.screen.text()}")
        self.assertEqual(above, 0,
                         "falling step-deltas must not rise above the center")
        # The centerline glyph row exists in the training graph.
        self.assertIn("─", rows[center])

    def test_rising_losses_render_above_centerline(self):
        self._feed_losses([0.2, 0.2, 0.4, 0.6, 0.8])
        rows, _top, center, col0 = self._training_graph_rows()
        blocks = "▄▀█"
        above = sum(rows[r][col0:].count(b) for b in blocks
                    for r in range(_top, center))
        below = sum(rows[r][col0:].count(b) for b in blocks
                    for r in range(center + 1, center + 4))
        self.assertGreaterEqual(above, 3,
                                f"rising losses must bar above the center:\n"
                                f"{self.screen.text()}")
        self.assertEqual(below, 0,
                         "rising step-deltas must not dip below the center")

    def test_flat_losses_render_centerline_only(self):
        self._feed_losses([0.7] * 8)
        rows, _top, center, _col0 = self._training_graph_rows()
        blocks = "▄▀█"
        self.assertIn("─", rows[center])
        self.assertFalse(any(b in rows[center] for b in blocks),
                         "a flat series must not render bars on the centerline")


class PoolVerticalGraphTest(AppTestCase):
    def setUp(self):
        super().setUp()
        self._real_snapshot = btui.bt.pool_snapshot
        self._snapshot = _fake_pool_snapshot([0] * 6)
        btui.bt.pool_snapshot = lambda: self._snapshot

    def tearDown(self):
        btui.bt.pool_snapshot = self._real_snapshot
        super().tearDown()

    def _poll_with(self, snapshot):
        self._snapshot = snapshot
        self.app._pool_at = 0.0
        self.app.redraw()

    def test_vertical_graph_replaces_header_spark(self):
        # Baseline poll: header keeps the readouts, graph rows are blank.
        self.app.redraw()
        rows = self.screen.text().splitlines()
        header_row = next(i for i, line in enumerate(rows) if "workers=" in line)
        self.assertIn("pending=", rows[header_row])
        self.assertNotIn("pool |", rows[header_row])
        # Activity lands as vertical block bars under the header.
        self._poll_with(_fake_pool_snapshot([50, 0, 0, 30, 0, 0]))
        self._poll_with(_fake_pool_snapshot([50, 0, 0, 30, 0, 0]))
        self._poll_with(_fake_pool_snapshot([90, 0, 0, 90, 0, 0]))
        rows = self.screen.text().splitlines()
        header_row = next(i for i, line in enumerate(rows) if "workers=" in line)
        self.assertIn("done=", rows[header_row])
        # The screen rows span BOTH panels; slice out the pool box interior
        # (its left border is the LAST '┌' on the title row — the Params
        # box owns the first one).
        title_row = next(i for i, line in enumerate(rows) if "CPU pool" in line)
        pool_col = rows[title_row].rindex("┌")
        graph_text = "".join(
            row[pool_col + 1:] for row in rows[header_row + 1: header_row + 4]
        )
        self.assertIn("█", graph_text,
                      f"no full-block bar in the pool graph:\n{self.screen.text()}")
        self.assertTrue(set(graph_text) <= set(" ▁▂▃▄▅▆▇█│"),
                        f"pool graph must use only the block ramp: {graph_text!r}")
        self.assertNotIn("·", graph_text, "the middot stopgap must be gone")

    def test_graph_rows_fit_box_width_as_ring_fills(self):
        """The graph columns track the ring tail but never exceed the box's
        inner width — the old growing-gap concern is gone by construction."""
        deltas = [90, 0, 0, 90, 0, 0]
        for _ in range(70):  # beyond the 60-sample ring
            self._poll_with(_fake_pool_snapshot(deltas))
        rows = self.screen.text().splitlines()
        _rows, cols = self.screen.getmaxyx()
        self.assertTrue(all(len(row) <= cols for row in rows))
        title_row = next(i for i, line in enumerate(rows) if "CPU pool" in line)
        pool_col = rows[title_row].rindex("┌")
        blocks = sum(
            row[pool_col + 1:].count(block)
            for row in rows[title_row + 2:]
            for block in "▁▂▃▄▅▆▇█"
        )
        self.assertLessEqual(blocks, 60 * 3,
                             "graph columns must not exceed the 60-sample ring")

    def test_graph_never_overwrites_right_border_at_80x24(self):
        """The ring holds 60 samples but the Pool box interior at 80x24 is
        narrower: once the ring passes the interior width, an unclipped
        graph row would write over the box's right border."""
        app = btui.BtuiApp(btui.BufferScreen(24, 80), has_gpu=True)
        real = btui.bt.pool_snapshot
        snapshot = _fake_pool_snapshot([0] * 6)
        btui.bt.pool_snapshot = lambda: snapshot
        try:
            app.apply_setting("device", "gpu")
            count = 0
            # Well past the ~44-sample interior: 60 ring entries, all active.
            for _ in range(60):
                count += 5
                snapshot = _fake_pool_snapshot([count] * 6)
                app._pool_at = 0.0
                app.redraw()
            rows = app.screen.text().splitlines()
            title_row = next(i for i, line in enumerate(rows) if "CPU pool" in line)
            pool_col = rows[title_row].rindex("┌")
            border_col = 80 - 1  # the box spans to the screen's last column
            for row_index in range(title_row + 1, len(rows)):
                # text() rstrips each row; pad so an overwritten (spaced-out)
                # border is detectable instead of an IndexError.
                line = rows[row_index].ljust(80)
                if line[pool_col] == "└":
                    break  # bottom border reached
                self.assertEqual(
                    line[border_col], "│",
                    f"row {row_index}: right border overwritten: {line!r}",
                )
        finally:
            btui.bt.pool_snapshot = real
            app.shutdown()


class TrainingVerticalGraphTest(AppTestCase):
    def test_loss_graph_rows_render_blocks(self):
        self.app.shared.begin_run()
        for loss in (1.0, 0.8, 0.9, 0.6, 0.7, 0.5, 0.55, 0.4):
            self.app.shared.record_batch(loss)
        self.app.redraw()
        text = self.screen.text()
        self.assertNotIn("spark:", text)
        self.assertIn("last=0.4", text)
        self.assertTrue(any(block in text for block in "▁▂▃▄▅▆▇█"),
                        f"no block bars in the training box:\n{text}")
        self.assertNotIn("·", text, "middot stopgap must be gone")


class Geometry80x24Test(AppTestCase):
    def setUp(self):
        self.screen = btui.BufferScreen(24, 80)
        self.app = btui.BtuiApp(self.screen)

    def test_borders_intact_at_80x24(self):
        self.app.redraw()
        rows = self.screen.text().splitlines()
        self.assertIn("CPU pool", self.screen.text())
        self.assertIn("└", self.screen.text())
        # Every drawn row fits the terminal width; borders were not clipped.
        self.assertTrue(all(len(row) <= 80 for row in rows))
        bottom = max(i for i, line in enumerate(rows) if "└" in line)
        self.assertLessEqual(bottom, 23)

    def test_gpu_hint_still_fits_at_80x24(self):
        app = btui.BtuiApp(btui.BufferScreen(24, 80), has_gpu=True)
        try:
            app.apply_setting("device", "gpu")
            app.redraw()
            text = app.screen.text()
            self.assertIn("gpu run: pool enqueues kernels", text)
            self.assertIn("└", text)
        finally:
            app.shutdown()

    def test_delta_graph_keeps_direction_at_80x24(self):
        # The minimum supported geometry must not collapse the delta graph
        # to a directionless single row: falling and rising sequences of
        # the same magnitudes render DIFFERENT rows, each on its own side
        # of the centerline.
        def graph_rows_for(losses):
            app = btui.BtuiApp(btui.BufferScreen(24, 80))
            try:
                app.shared.begin_run()
                for loss in losses:
                    app.shared.record_batch(loss)
                app.redraw()
                rows = app.screen.text().splitlines()
                title = next(i for i, line in enumerate(rows)
                             if "─ Training" in line)
                return rows[title + 3: title + 6]
            finally:
                app.shutdown()

        falling = graph_rows_for([0.8, 0.8, 0.6, 0.4, 0.2])
        rising = graph_rows_for([0.2, 0.2, 0.4, 0.6, 0.8])
        self.assertNotEqual(falling, rising,
                            "falling and rising rendered identically at 80x24")
        blocks = "▁▂▃▄▅▆▇█"
        falling_below = sum(falling[2].count(b) for b in blocks)
        rising_above = sum(rising[0].count(b) for b in blocks)
        self.assertGreaterEqual(falling_below, 1,
                                f"no below-center bar at 80x24: {falling}")
        self.assertGreaterEqual(rising_above, 1,
                                f"no above-center bar at 80x24: {rising}")


class SaveBusyTest(AppTestCase):
    """While a restart seeding copy is in flight, diff_keep_weights mutates
    the old net's params in place (GPU params move CPU-ward), and last_net
    aliases that net until the new run lands — so w must refuse rather than
    race the buffer swap with a checkpoint write."""

    def test_save_refused_during_restart_copy(self):
        self.app.last_net = btui.build_net(
            btui.ModelSpec(layers=[
                btui.LayerSpec("Linear", {"in_features": 2, "out_features": 2}),
            ]),
            seed=1,
        )
        # Simulate the copy window: the training thread holds the flag
        # around diff_keep_weights.
        self.app.shared.set_seeding(True)
        self.app.handle_key("w")
        self.assertIsNone(self.app.form, "save form must not open mid-copy")
        self.assertTrue(
            any("save busy: restart copy in flight" in line
                for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )
        # Once the copy lands, saving opens the form as usual.
        self.app.shared.set_seeding(False)
        self.app.handle_key("w")
        self.assertIsNotNone(self.app.form)


class SlowJob:
    """TrainJob-shaped double whose join blocks until the test releases it —
    a stop that takes arbitrarily long, without the wait."""

    def __init__(self):
        self.net = None
        self._alive = True
        self._gate = threading.Event()

    def stop(self):
        pass

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self._gate.wait(timeout)

    def release(self):
        self._alive = False
        self._gate.set()


class ShutdownTest(AppTestCase):
    def test_shutdown_waits_for_slow_join_and_stays_responsive(self):
        job = SlowJob()
        self.app.job = job
        refreshes_before = self.screen.refresh_count
        done = threading.Event()

        def run_shutdown():
            self.app.shutdown()
            done.set()

        worker = threading.Thread(target=run_shutdown)
        worker.start()
        time.sleep(0.4)
        self.assertFalse(done.is_set(), "shutdown returned before the job joined")
        self.assertGreater(self.screen.refresh_count, refreshes_before,
                           "the UI stopped redrawing during the wait")
        job.release()
        worker.join(timeout=10.0)
        self.assertFalse(worker.is_alive(), "shutdown hung after the join landed")
        self.assertTrue(done.is_set())
        self.assertIsNone(self.app.job)

    def test_first_q_stops_second_q_forces(self):
        self.app.job = SlowJob()
        self.app.handle_key("q")
        self.assertFalse(self.app.quit_requested)
        self.app.handle_key("q")
        self.assertTrue(self.app.quit_requested)
        # tearDown's shutdown must not block on the still-live double: the
        # force flag short-circuits the wait loop.
        self.app.shutdown()
        job = self.app.job
        self.assertIsNotNone(job)  # left running by design (documented hatch)


class CliExitCodeTest(unittest.TestCase):
    """Subprocess coverage of the real --headless CLI branch: a passing run
    exits 0, a run that errors engine-side exits nonzero."""

    @classmethod
    def setUpClass(cls):
        os.makedirs(SCRATCH_BTUI, exist_ok=True)
        cls.csv_path = os.path.join(SCRATCH_BTUI, "cli_fixture_16row.csv")
        with open(cls.csv_path, "w", encoding="utf-8") as fh:
            fh.write(CSV_16ROW)
        cls.pass_keys = os.path.join(SCRATCH_BTUI, "cli_pass.keys")
        with open(cls.pass_keys, "w", encoding="utf-8") as fh:
            fh.write("\n".join([
                "SET:data_source=csv",
                f"SET:csv_path={cls.csv_path}",
                "SET:train_rows=16",
                "SET:batch=8",
                "SET:epochs=1",
                "a", "TYPE:Linear", "ENTER", "ENTER", "ENTER",
                "a", "TYPE:Relu", "ENTER",
                "a", "TYPE:Linear", "ENTER", "ENTER", "ENTER",
                "r",
            ]) + "\n")
        cls.fail_keys = os.path.join(SCRATCH_BTUI, "cli_fail.keys")
        with open(cls.fail_keys, "w", encoding="utf-8") as fh:
            fh.write("\n".join([
                "SET:data_source=csv",
                f"SET:csv_path={SCRATCH_BTUI}/__missing__.csv",
                "a", "TYPE:Linear", "ENTER", "ENTER", "ENTER",
                "r",
            ]) + "\n")

    def run_cli(self, keys_path):
        env = dict(os.environ)
        mod_dir = os.path.dirname(os.path.abspath(bt.__file__))
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (mod_dir, env.get("PYTHONPATH", "")) if p
        )
        return subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "python", "btui.py"),
             "--headless", keys_path],
            capture_output=True, text=True, timeout=180, env=env, cwd=REPO_ROOT,
        )

    def test_passing_run_exits_zero(self):
        proc = self.run_cli(self.pass_keys)
        self.assertEqual(proc.returncode, 0,
                         f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("Training", proc.stdout)
        self.assertRegex(proc.stdout, r"last=\d+\.\d+")

    def test_failed_run_exits_nonzero(self):
        proc = self.run_cli(self.fail_keys)
        self.assertNotEqual(proc.returncode, 0,
                            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("run failed", proc.stdout)


class DetectedInputTest(AppTestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_path = os.path.join(SCRATCH_BTUI, "ui_detect_2feat.csv")
        os.makedirs(SCRATCH_BTUI, exist_ok=True)
        with open(cls.csv_path, "w", encoding="utf-8") as fh:
            fh.write("x1,x2,cls\n0.0,0.0,a\n1.0,1.0,b\n")

    def set_csv_source(self):
        self.app.apply_setting("data_source", "csv")
        self.app.apply_setting("csv_path", self.csv_path)

    def test_model_panel_info_line(self):
        # Default mnist_flat source: static shape, no file access.
        self.app.redraw()
        self.assertIn("data input: 784", self.screen.text())
        # NCHW shows the tensor shape.
        self.app.apply_setting("data_source", "mnist_nchw")
        self.app.redraw()
        self.assertIn("data input: (1, 28, 28)", self.screen.text())
        # CSV reads the fixture's feature count.
        self.set_csv_source()
        self.app.redraw()
        self.assertIn("data input: 2", self.screen.text())
        # A bad path reports unknown instead of raising.
        self.app.apply_setting("csv_path", os.path.join(SCRATCH_BTUI, "__missing__.csv"))
        self.app.redraw()
        self.assertIn("data input: unknown", self.screen.text())

    def test_first_layer_input_prefill(self):
        self.set_csv_source()
        self.add_layer("Linear")  # defaults in_features=2, out_features=2
        self.assertEqual(self.app.layer_sel, 0)
        # ENTER on the first layer's first param (in_features) pre-fills the
        # detected scalar, stated in the prompt, still user-editable.
        self.app.handle_key("ENTER")
        self.assertIsNotNone(self.app.form)
        self.assertEqual(self.app.form.text, "2")
        self.assertIn("detected 2", self.app.form.prompt)
        # Clear the prefill and replace it.
        self.app.handle_key("BACKSPACE")
        type_text(self.app, "7")
        self.app.handle_key("ENTER")
        self.assertEqual(self.app.spec.layers[0].params["in_features"], 7)
        # Non-first layers get no prefill.
        self.add_layer("Linear")
        self.app.handle_key("ENTER")
        self.assertEqual(self.app.form.text, "")


class ImportArchitectureTest(AppTestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(SCRATCH_BTUI, exist_ok=True)
        cls.bt_path = os.path.join(SCRATCH_BTUI, "import_fixture.bt")

    def _make_checkpoint(self):
        spec = btui.ModelSpec(layers=[
            btui.LayerSpec("Linear", {"in_features": 2, "out_features": 8}),
            btui.LayerSpec("Relu", {}),
            btui.LayerSpec("Dropout", {"p": 0.25}),
            btui.LayerSpec("Linear", {"in_features": 8, "out_features": 2}),
        ])
        net = btui.build_net(spec, seed=2026)
        bt.save(net, self.bt_path, True, bt.CheckpointLevel.Full)
        return spec, {name: t.to_numpy().tobytes()
                      for name, t in net.named_parameters().items()}

    def test_import_replaces_spec_and_seeds_weights(self):
        spec, net_bytes = self._make_checkpoint()
        self.assertEqual(self.app.spec.layers, [])
        self.app.handle_key("L")
        self.assertIsNotNone(self.app.form)
        type_text(self.app, self.bt_path)
        self.app.handle_key("ENTER")
        self.assertIsNone(self.app.form)
        # The imported spec matches the saved architecture exactly.
        self.assertEqual(
            [(layer.type, layer.params) for layer in self.app.spec.layers],
            [(layer.type, layer.params) for layer in spec.layers],
        )
        # The loaded module is stashed as the pending seed-weights source:
        # a fresh build from the imported spec takes every parameter
        # byte-identical from it.
        self.assertIsNotNone(self.app.seed_weights)
        fresh = btui.build_net(self.app.spec, seed=99)
        copied = btui.diff_keep_weights(self.app.seed_weights, fresh)
        self.assertEqual(sorted(copied), sorted(net_bytes.keys()))
        for name, param in fresh.named_parameters().items():
            self.assertEqual(param.to_numpy().tobytes(), net_bytes[name],
                             f"{name} not seeded byte-identical")
        self.assertTrue(self.app.shared.pending)
        import_lines = [line for line in self.app.log_lines if "imported" in line]
        self.assertEqual(len(import_lines), 1)
        self.assertNotIn("dropped:", import_lines[0],
                         "a clean import must not claim drops")

    def test_import_foreign_spec_warns_drops(self):
        self._make_checkpoint()
        real_export_spec = btui.bt.export_spec
        foreign = {
            "type": "Sequential", "hyperparameters": {},
            "children": [
                {"name": "0", "type": "Conv2d",
                 "hyperparameters": {"in_channels": "1", "out_channels": "4",
                                     "kernel": "3", "stride": "1",
                                     "padding": "0", "bias": "0"},
                 "children": []},
                {"name": "1", "type": "BatchNorm",
                 "hyperparameters": {"num_features": "4",
                                     "momentum": "0x1.999999999999ap-4",
                                     "eps": "0x1.0p-5"},
                 "children": []},
            ],
        }
        btui.bt.export_spec = lambda module: foreign
        try:
            self.app.handle_key("L")
            type_text(self.app, self.bt_path)
            self.app.handle_key("ENTER")
        finally:
            btui.bt.export_spec = real_export_spec
        # The import succeeds (warnings, not rejection) and the log line
        # names exactly what was dropped.
        self.assertEqual(
            [layer.type for layer in self.app.spec.layers],
            ["Conv2d", "BatchNorm"],
        )
        self.assertNotIn("bias", self.app.spec.layers[0].params)
        import_lines = [line for line in self.app.log_lines if "imported" in line]
        self.assertEqual(len(import_lines), 1)
        self.assertIn(
            f"(dropped: layer 0 Conv2d bias=0, layer 1 BatchNorm momentum/eps)",
            import_lines[0],
            f"log line: {import_lines[0]}",
        )

    def test_import_unknown_layer_type_rejected(self):
        self._make_checkpoint()
        spec_before = self.app.spec
        real_export_spec = btui.bt.export_spec
        bogus = {
            "type": "Sequential", "hyperparameters": {},
            "children": [
                {"name": "0", "type": "Linear",
                 "hyperparameters": {"in_features": "2", "out_features": "8",
                                     "bias": "1"},
                 "children": []},
                {"name": "1", "type": "Teleport",
                 "hyperparameters": {}, "children": []},
            ],
        }
        btui.bt.export_spec = lambda module: bogus
        try:
            self.app.handle_key("L")
            type_text(self.app, self.bt_path)
            self.app.handle_key("ENTER")
        finally:
            btui.bt.export_spec = real_export_spec
        # No partial import: the spec is untouched and the log names the error.
        self.assertEqual(self.app.spec, spec_before)
        self.assertIsNone(self.app.seed_weights)
        self.assertTrue(
            any("unknown layer type" in line for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )

    def test_import_missing_file_logs_error(self):
        self.app.handle_key("L")
        type_text(self.app, os.path.join(SCRATCH_BTUI, "__no_such__.bt"))
        self.app.handle_key("ENTER")
        self.assertEqual(self.app.spec.layers, [])
        self.assertTrue(
            any("invalid" in line for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )


class SaveLevelTest(AppTestCase):
    def setUp(self):
        super().setUp()
        self.app.last_net = btui.build_net(
            btui.ModelSpec(layers=[
                btui.LayerSpec("Linear", {"in_features": 2, "out_features": 2}),
            ]),
            seed=3,
        )

    def _save(self, path, level_text):
        # Hermetic: remove any stale artifact so file-existence assertions
        # can only pass when THIS drive actually saved.
        if os.path.isfile(path):
            os.unlink(path)
        self.app.handle_key("w")
        type_text(self.app, path)
        self.app.handle_key("ENTER")  # path commits; the strip form opens
        self.assertIsNotNone(self.app.form)
        self.app.handle_key("ENTER")  # strip: empty commit = default n
        self.assertIsNotNone(self.app.form)  # then the level form
        type_text(self.app, level_text)
        self.app.handle_key("ENTER")
        self.assertIsNone(self.app.form)

    def test_save_at_each_level(self):
        sizes = {}
        for level in ("Weights", "WeightsAndGrads", "Full"):
            path = os.path.join(SCRATCH_BTUI, f"level_{level}.bt")
            self._save(path, level)
            self.assertTrue(os.path.isfile(path), f"{level} save missing")
            loaded = bt.load(path)
            self.assertEqual(
                sorted(loaded.named_parameters().keys()),
                ["0.bias", "0.weight"],
            )
            sizes[level] = os.path.getsize(path)
        self.assertGreater(sizes["Full"], sizes["Weights"])
        self.assertTrue(
            any("(Full)" in line for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )

    def test_save_level_default_and_validation(self):
        # Empty level commit = the Weights default.
        default_path = os.path.join(SCRATCH_BTUI, "level_default.bt")
        self._save(default_path, "")
        self.assertTrue(os.path.isfile(default_path))
        # A bogus level is rejected; nothing is written.
        bogus_path = os.path.join(SCRATCH_BTUI, "level_bogus.bt")
        self._save(bogus_path, "Everything")
        self.assertFalse(os.path.isfile(bogus_path))
        self.assertTrue(
            any("invalid save level" in line for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )


class ResumeTest(AppTestCase):
    """L-import a Full checkpoint and run unedited: the loaded module IS the
    run net, so Adam moments and step counters continue. Any spec edit makes
    the spec dirty and the next run rebuilds + re-seeds instead (moments
    reset — the documented tradeoff)."""

    CSV_PATH = os.path.join(SCRATCH_BTUI, "resume_16row.csv")
    BT_PATH = os.path.join(SCRATCH_BTUI, "resume_full.bt")

    @classmethod
    def setUpClass(cls):
        os.makedirs(SCRATCH_BTUI, exist_ok=True)
        with open(cls.CSV_PATH, "w", encoding="utf-8") as fh:
            fh.write(CSV_16ROW)

    def _trained_checkpoint(self):
        spec = btui.ModelSpec(layers=[
            btui.LayerSpec("Linear", {"in_features": 2, "out_features": 8}),
            btui.LayerSpec("Relu", {}),
            btui.LayerSpec("Linear", {"in_features": 8, "out_features": 2}),
        ])
        net = btui.build_net(spec, seed=5)
        net.train()
        import numpy as np

        x = bt.Tensor.from_numpy(np.zeros((4, 2), dtype=np.float32))
        y = bt.Tensor.from_numpy(np.array([0, 1, 0, 1], dtype=np.int64))
        for _ in range(3):
            net.zero_grad()
            loss = bt.softmax_cross_entropy(net.forward(x), y)
            loss.backward(btui.F32)
            net.adam_step(1e-2, 0.9, 0.999, 1e-8)
        bt.save(net, self.BT_PATH, True, bt.CheckpointLevel.Full)
        return spec

    def _import_and_run(self, edit_after_import):
        self.app.handle_key("L")
        type_text(self.app, self.BT_PATH)
        self.app.handle_key("ENTER")
        self.assertIsNone(self.app.form)
        loaded = self.app.seed_weights
        self.assertIsNotNone(loaded)
        if edit_after_import:
            # Same-value param edit: dirty flag without a shape change. The
            # detected-input prefill may have filled the form — clear it
            # first so the typed value stands alone.
            feed(self.app, "ENTER")
            while self.app.form.text:
                self.app.handle_key("BACKSPACE")
            type_text(self.app, str(self.app.spec.layers[0].params["in_features"]))
            self.app.handle_key("ENTER")
        self.app.apply_setting("data_source", "csv")
        self.app.apply_setting("csv_path", self.CSV_PATH)
        self.app.apply_setting("batch", "8")
        self.app.apply_setting("epochs", "1")
        self.app.handle_key("r")
        deadline = time.perf_counter() + 30.0
        while self.app.busy and time.perf_counter() < deadline:
            self.app.redraw()
            time.sleep(0.01)
        self.assertFalse(self.app.busy, "run did not finish")
        self.assertIsNone(self.app.shared.error)
        self.assertIsNotNone(self.app.last_net)
        return loaded

    def test_clean_resume_preserves_moments(self):
        self._trained_checkpoint()
        loaded = self._import_and_run(edit_after_import=False)
        # True resume: the run net IS the loaded module, and its step count
        # continued (3 pre-save steps + 2 batches in the fixture epoch).
        self.assertIs(self.app.last_net, loaded)
        count = self.app.last_net.named_parameters()["0.weight"].adam_step_count()
        self.assertEqual(count, 5)
        self.assertTrue(
            any("resumed: optimizer state intact" in line
                for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )

    def test_dirty_edit_rebuilds_and_resets_moments(self):
        self._trained_checkpoint()
        loaded = self._import_and_run(edit_after_import=True)
        # Rebuild path: a fresh net (weights re-seeded byte-identically), so
        # the optimizer state starts over and the loaded module is untouched.
        self.assertIsNot(self.app.last_net, loaded)
        count = self.app.last_net.named_parameters()["0.weight"].adam_step_count()
        self.assertEqual(count, 2)
        self.assertEqual(
            loaded.named_parameters()["0.weight"].adam_step_count(), 3,
            "the rebuild path must not train the loaded module",
        )
        self.assertTrue(
            any("rebuilt: weights re-seeded" in line
                for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )


class GpuResumeRegressionTest(unittest.TestCase):
    """The device-mismatch resume bug: a CPU-saved Full checkpoint resumed
    onto GPU must REBUILD (moments honestly reset) — Adam moments live in an
    address-keyed map and do not move with Tensor.to, so the old ready path
    silently zeroed them while logging "resumed: optimizer state intact".
    Runs for real only on a GPU build (probe-gated)."""

    CSV_PATH = os.path.join(SCRATCH_BTUI, "resume_gpu_16row.csv")
    BT_PATH = os.path.join(SCRATCH_BTUI, "resume_gpu_full.bt")

    @classmethod
    def setUpClass(cls):
        if not btui.probe_gpu():
            raise unittest.SkipTest("SKIP: no gpu in this build")
        os.makedirs(SCRATCH_BTUI, exist_ok=True)
        with open(cls.CSV_PATH, "w", encoding="utf-8") as fh:
            fh.write(CSV_16ROW)
        spec = btui.ModelSpec(layers=[
            btui.LayerSpec("Linear", {"in_features": 2, "out_features": 8}),
            btui.LayerSpec("Relu", {}),
            btui.LayerSpec("Linear", {"in_features": 8, "out_features": 2}),
        ])
        net = btui.build_net(spec, seed=5)
        net.train()
        import numpy as np

        x = bt.Tensor.from_numpy(np.zeros((4, 2), dtype=np.float32))
        y = bt.Tensor.from_numpy(np.array([0, 1, 0, 1], dtype=np.int64))
        for _ in range(3):
            net.zero_grad()
            loss = bt.softmax_cross_entropy(net.forward(x), y)
            loss.backward(btui.F32)
            net.adam_step(1e-2, 0.9, 0.999, 1e-8)
        bt.save(net, cls.BT_PATH, True, bt.CheckpointLevel.Full)

    def test_cpu_checkpoint_onto_gpu_rebuilds_honestly(self):
        app = btui.BtuiApp(btui.BufferScreen(34, 110))  # probe: has_gpu True
        try:
            app.handle_key("L")
            for ch in self.BT_PATH:
                app.handle_key(ch)
            app.handle_key("ENTER")
            self.assertIsNone(app.form)
            loaded = app.seed_weights
            self.assertIsNotNone(loaded)
            app.apply_setting("device", "gpu")
            app.apply_setting("data_source", "csv")
            app.apply_setting("csv_path", self.CSV_PATH)
            app.apply_setting("batch", "8")
            app.apply_setting("epochs", "1")
            app.handle_key("r")
            deadline = time.perf_counter() + 60.0
            while app.busy and time.perf_counter() < deadline:
                app.redraw()
                time.sleep(0.01)
            self.assertFalse(app.busy, "run did not finish")
            self.assertIsNone(app.shared.error)
            self.assertTrue(
                any("device changed: rebuilt with weight re-seed" in line
                    and "optimizer state reset" in line
                    for line in app.log_lines),
                f"log lines: {app.log_lines}",
            )
            self.assertFalse(
                any("resumed: optimizer state intact" in line
                    for line in app.log_lines),
                "the dishonest resume claim must be gone",
            )
            # Honest rebuild: fresh net, count restarted from 0
            # (2 batches in the fixture epoch), loaded module untouched.
            self.assertIsNot(app.last_net, loaded)
            self.assertEqual(
                app.last_net.named_parameters()["0.weight"].adam_step_count(), 2)
            self.assertEqual(
                loaded.named_parameters()["0.weight"].adam_step_count(), 3)
        finally:
            app.shutdown()


class _FakeStdscr:
    """Minimal stdscr stand-in for CursesScreen construction tests: the
    methods the app actually calls, as no-ops with sane returns."""

    def keypad(self, flag):
        pass

    def timeout(self, ms):
        pass

    def getmaxyx(self):
        return (34, 110)

    def erase(self):
        pass

    def refresh(self):
        pass

    def addstr(self, y, x, text, attr=0):
        pass

    def getch(self):
        return -1


class CursesColorFallbackTest(unittest.TestCase):
    """CursesScreen must degrade cleanly when the terminal's terminfo
    lacks default-color support: init_pair falls back to COLOR_BLACK
    instead of -1, and with no color support at all every style is 0 —
    the app keeps running either way."""

    def _patched_curses(self, has_colors, use_default_raises):
        import curses

        calls = []
        real = {
            "has_colors": curses.has_colors,
            "start_color": curses.start_color,
            "use_default_colors": curses.use_default_colors,
            "init_pair": curses.init_pair,
            "color_pair": curses.color_pair,
        }
        curses.has_colors = lambda: has_colors
        curses.start_color = lambda: None
        # These scenarios exercise the color-init path itself, so the
        # environment's own NO_COLOR must not short-circuit it.
        self._saved_no_color = os.environ.pop("NO_COLOR", None)
        self._saved_term = os.environ.get("TERM")
        os.environ["TERM"] = "xterm-256color"

        def fake_use_default():
            if use_default_raises:
                raise curses.error("no default colors")

        def fake_init_pair(pair, fg, bg):
            calls.append((pair, fg, bg))

        curses.use_default_colors = fake_use_default
        curses.init_pair = fake_init_pair
        # The real color subsystem never sees our fake init_pair calls, so
        # color_pair must be faked too for style() to resolve.
        curses.color_pair = lambda n: n << 8
        return curses, real, calls

    def _restore_curses(self, curses, real):
        for name, fn in real.items():
            setattr(curses, name, fn)
        if self._saved_no_color is not None:
            os.environ["NO_COLOR"] = self._saved_no_color
        if self._saved_term is None:
            os.environ.pop("TERM", None)
        else:
            os.environ["TERM"] = self._saved_term

    def test_default_colors_failure_falls_back_to_black_bg(self):
        curses, real, calls = self._patched_curses(
            has_colors=True, use_default_raises=True)
        try:
            # The patches must stay live through the style/redraw calls:
            # CursesScreen resolves color_pair at call time.
            screen = btui.CursesScreen(_FakeStdscr())
            self.assertTrue(calls, "init_pair was never called")
            self.assertTrue(
                all(bg == curses.COLOR_BLACK for _p, _f, bg in calls),
                f"bg must fall back to COLOR_BLACK: {calls}")
            # The app still runs and color styles stay live (black bg).
            self.assertNotEqual(screen.style("error"), 0)
            app = btui.BtuiApp(screen)
            try:
                app.redraw()
            finally:
                app.shutdown()
        finally:
            self._restore_curses(curses, real)

    def test_no_color_support_renders_attr_zero(self):
        curses, real, _calls = self._patched_curses(
            has_colors=False, use_default_raises=False)
        try:
            screen = btui.CursesScreen(_FakeStdscr())
            for name in ("error", "st:RunningOwn", "state:running", "dim"):
                self.assertEqual(screen.style(name), 0, f"{name} must be attr 0")
            app = btui.BtuiApp(screen)
            try:
                app.redraw()
            finally:
                app.shutdown()
        finally:
            self._restore_curses(curses, real)


class PoolPanelHonestyTest(AppTestCase):
    def test_pool_box_labeled_cpu_pool(self):
        self.app.redraw()
        self.assertIn("CPU pool", self.screen.text())

    def test_gpu_hint_line_only_on_gpu(self):
        app = btui.BtuiApp(btui.BufferScreen(34, 110), has_gpu=True)
        try:
            app.redraw()
            self.assertNotIn("gpu run: pool enqueues kernels", app.screen.text())
            app.apply_setting("device", "gpu")
            app.redraw()
            self.assertIn("gpu run: pool enqueues kernels", app.screen.text())
            app.apply_setting("device", "cpu")
            app.redraw()
            self.assertNotIn("gpu run: pool enqueues kernels", app.screen.text())
        finally:
            app.shutdown()


class EvaluatingIndicatorTest(AppTestCase):
    def test_evaluating_indicator_renders_and_clears(self):
        self.assertNotIn("evaluating...", self._render_text())
        self.app.shared.set_evaluating(True)
        self.assertIn("evaluating...", self._render_text())
        self.app.shared.set_evaluating(False)
        self.assertNotIn("evaluating...", self._render_text())

    def test_snapshot_carries_evaluating_flag(self):
        self.assertFalse(self.app.shared.snapshot()["evaluating"])
        self.app.shared.set_evaluating(True)
        self.assertTrue(self.app.shared.snapshot()["evaluating"])
        self.app.shared.set_evaluating(False)

    def _render_text(self):
        self.app.redraw()
        return self.screen.text()


class ColorDiagTest(unittest.TestCase):
    """Startup color diagnostics: a monochrome session must say WHY in the
    log strip (TERM value, NO_COLOR honored), and a capable session says so
    positively. Diagnose-only — never forces colors past terminfo."""

    def _app_with(self, has_colors, env_updates, env_unset=()):
        import curses

        real = {
            "has_colors": curses.has_colors,
            "start_color": curses.start_color,
            "use_default_colors": curses.use_default_colors,
            "init_pair": curses.init_pair,
            "color_pair": curses.color_pair,
        }
        curses.has_colors = lambda: has_colors
        curses.start_color = lambda: None
        curses.use_default_colors = lambda: None
        curses.init_pair = lambda p, f, b: None
        curses.color_pair = lambda n: n << 8
        saved_env = {}
        for key, value in env_updates.items():
            saved_env[key] = os.environ.get(key)
            os.environ[key] = value
        for key in env_unset:
            saved_env[key] = os.environ.pop(key, None)
        app = None
        try:
            screen = btui.CursesScreen(_FakeStdscr())
            app = btui.BtuiApp(screen)
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            for name, fn in real.items():
                setattr(curses, name, fn)
        return app

    def setUp(self):
        self._saved_term = os.environ.get("TERM")
        self._saved_no_color = os.environ.get("NO_COLOR")

    def tearDown(self):
        if self._saved_term is None:
            os.environ.pop("TERM", None)
        else:
            os.environ["TERM"] = self._saved_term
        if self._saved_no_color is None:
            os.environ.pop("NO_COLOR", None)
        else:
            os.environ["NO_COLOR"] = self._saved_no_color

    def test_colors_off_reports_term(self):
        app = self._app_with(has_colors=False,
                             env_updates={"TERM": "dumb"},
                             env_unset=("NO_COLOR",))
        try:
            self.assertTrue(
                any("colors off: TERM=dumb reports no color support" in line
                    and "xterm-256color" in line
                    for line in app.log_lines),
                f"log lines: {app.log_lines}",
            )
        finally:
            app.shutdown()

    def test_no_color_env_honored(self):
        app = self._app_with(has_colors=True,
                             env_updates={"NO_COLOR": "1", "TERM": "xterm-256color"})
        try:
            self.assertTrue(
                any("colors off: NO_COLOR set" in line for line in app.log_lines),
                f"log lines: {app.log_lines}",
            )
            self.assertEqual(app.screen.style("error"), 0)
        finally:
            app.shutdown()

    def test_colors_on_positive_line(self):
        app = self._app_with(has_colors=True,
                             env_updates={"TERM": "xterm-256color"},
                             env_unset=("NO_COLOR",))
        try:
            self.assertTrue(
                any(line.startswith("colors on (") and "pairs" in line
                    for line in app.log_lines),
                f"log lines: {app.log_lines}",
            )
        finally:
            app.shutdown()


class ShareStripTest(AppTestCase):
    """The save form's strip-for-sharing step: y writes a recipient-safe
    copy (Weights level, NO metadata), n keeps the level + embed flow."""

    def setUp(self):
        super().setUp()
        # A net with real Adam state, so the level difference is observable.
        net = btui.build_net(
            btui.ModelSpec(layers=[
                btui.LayerSpec("Linear", {"in_features": 2, "out_features": 2}),
            ]),
            seed=3,
        )
        net.train()
        import numpy as np

        x = bt.Tensor.from_numpy(np.zeros((4, 2), dtype=np.float32))
        y = bt.Tensor.from_numpy(np.array([0, 1, 0, 1], dtype=np.int64))
        for _ in range(2):
            net.zero_grad()
            loss = bt.softmax_cross_entropy(net.forward(x), y)
            loss.backward(btui.F32)
            net.adam_step(1e-2, 0.9, 0.999, 1e-8)
        self.app.last_net = net

    def _save_with_strip(self, path, strip_answer, level=""):
        # Hermetic: remove any stale artifact so file-existence assertions
        # can only pass when THIS drive actually saved.
        if os.path.isfile(path):
            os.unlink(path)
        self.app.handle_key("w")
        type_text(self.app, path)
        self.app.handle_key("ENTER")  # path commits; the strip form opens
        self.assertIsNotNone(self.app.form)
        type_text(self.app, strip_answer)
        self.app.handle_key("ENTER")
        if strip_answer in ("", "n"):
            # strip=n continues to the level prompt.
            self.assertIsNotNone(self.app.form)
            type_text(self.app, level)
            self.app.handle_key("ENTER")
        self.assertIsNone(self.app.form)

    def test_strip_yes_writes_weights_only_no_metadata(self):
        path = os.path.join(SCRATCH_BTUI, "share_stripped.bt")
        self._save_with_strip(path, "y")
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(bt.read_metadata(path), "")
        loaded = bt.load(path)
        self.assertIsNone(loaded.named_parameters()["0.weight"].adam_step_count(),
                          "a stripped copy must drop optimizer state")
        self.assertTrue(
            any("shared copy:" in line and "weights only, config stripped" in line
                and path in line for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )

    def test_strip_no_keeps_level_and_metadata(self):
        path = os.path.join(SCRATCH_BTUI, "share_full.bt")
        self._save_with_strip(path, "n", level="Full")
        self.assertTrue(os.path.isfile(path))
        self.assertIn("optimizer=", bt.read_metadata(path))
        loaded = bt.load(path)
        self.assertEqual(loaded.named_parameters()["0.weight"].adam_step_count(), 2)

    def test_strip_default_is_no(self):
        path = os.path.join(SCRATCH_BTUI, "share_default.bt")
        self._save_with_strip(path, "", level="Full")  # empty = default n
        self.assertIn("optimizer=", bt.read_metadata(path))

    def test_strip_rejects_garbage(self):
        path = os.path.join(SCRATCH_BTUI, "share_garbage.bt")
        if os.path.isfile(path):
            os.unlink(path)
        self.app.handle_key("w")
        type_text(self.app, path)
        self.app.handle_key("ENTER")
        type_text(self.app, "maybe")
        self.app.handle_key("ENTER")
        self.assertIsNone(self.app.form)
        self.assertFalse(os.path.isfile(path))
        self.assertTrue(
            any("invalid strip" in line for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )


class ConfigMetadataTest(AppTestCase):
    """The w save embeds the run config as key=value metadata; L restores
    what parses (unknown keys ignored, malformed values keep defaults)."""

    def setUp(self):
        super().setUp()
        self.app.last_net = btui.build_net(
            btui.ModelSpec(layers=[
                btui.LayerSpec("Linear", {"in_features": 2, "out_features": 2}),
            ]),
            seed=3,
        )

    def _save_with_config(self, path):
        # Hermetic: remove any stale artifact so the isfile check below can
        # only pass when THIS drive actually saved.
        if os.path.isfile(path):
            os.unlink(path)
        self.app.apply_setting("optimizer", "sgd")
        self.app.apply_setting("epochs", "7")
        self.app.apply_setting("batch", "8")
        self.app.apply_setting("seed", "99")
        self.app.apply_setting("data_source", "csv")
        self.app.apply_setting("csv_path", "some/data.csv")
        self.app.apply_setting("data_dir", "some/mnist")
        self.app.handle_key("w")
        type_text(self.app, path)
        self.app.handle_key("ENTER")  # path commits; the strip form opens
        self.app.handle_key("ENTER")  # strip: empty = default n
        self.app.handle_key("ENTER")  # level: empty commit = Weights
        self.assertTrue(os.path.isfile(path), "save did not write a file")

    def test_save_embeds_config_metadata(self):
        path = os.path.join(SCRATCH_BTUI, "cfg_embed.bt")
        self._save_with_config(path)
        text = bt.read_metadata(path)
        pairs = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
        self.assertEqual(pairs["optimizer"], "sgd")
        self.assertEqual(pairs["epochs"], "7")
        self.assertEqual(pairs["batch"], "8")
        self.assertEqual(pairs["seed"], "99")
        self.assertEqual(pairs["source"], "csv")
        self.assertEqual(pairs["csv_path"], "some/data.csv")
        self.assertEqual(pairs["data_dir"], "some/mnist")
        self.assertIn("lr=", text)

    def _dirty_fresh_app(self):
        """A non-fresh app with deliberately weird settings everywhere —
        every L import must reset these to defaults before applying
        metadata, never leave them stale."""
        app = btui.BtuiApp(btui.BufferScreen(34, 110))
        app.apply_setting("optimizer", "sgd")
        app.apply_setting("epochs", "42")
        app.apply_setting("lr", "0.5")
        app.apply_setting("seed", "7")
        app.apply_setting("data_source", "csv")
        app.apply_setting("csv_path", "stale.csv")
        app.apply_setting("data_dir", "stale/dir")
        return app

    def test_import_restores_config_and_logs(self):
        path = os.path.join(SCRATCH_BTUI, "cfg_roundtrip.bt")
        self._save_with_config(path)
        # Import into a NON-fresh app: the metadata values must override the
        # stale ones.
        fresh = self._dirty_fresh_app()
        try:
            fresh.handle_key("L")
            type_text(fresh, path)
            fresh.handle_key("ENTER")
            self.assertEqual(fresh.hp_cfg.optimizer, "sgd")
            self.assertEqual(fresh.hp_cfg.epochs, 7)
            self.assertEqual(fresh.hp_cfg.batch, 8)
            self.assertEqual(fresh.hp_cfg.seed, 99)
            self.assertEqual(fresh.data_cfg.source, "csv")
            self.assertEqual(fresh.data_cfg.csv_path, "some/data.csv")
            self.assertEqual(fresh.data_cfg.data_dir, "some/mnist")
            self.assertTrue(
                any("restored config:" in line and "epochs=7" in line
                    for line in fresh.log_lines),
                f"log lines: {fresh.log_lines}",
            )
        finally:
            fresh.shutdown()

    def test_import_without_metadata_resets_to_defaults(self):
        # An engine-written checkpoint (no metadata) imported into a
        # non-fresh app voids EVERY config to defaults.
        path = os.path.join(SCRATCH_BTUI, "cfg_none.bt")
        bt.save(self.app.last_net, path, True, bt.CheckpointLevel.Weights)
        fresh = self._dirty_fresh_app()
        try:
            fresh.handle_key("L")
            type_text(fresh, path)
            fresh.handle_key("ENTER")
            self.assertEqual(fresh.hp_cfg, btui.HyperConfig())
            self.assertEqual(fresh.data_cfg, btui.DataConfig(source=btui.MNIST_FLAT))
            self.assertFalse(
                any("restored config:" in line for line in fresh.log_lines),
                f"log lines: {fresh.log_lines}",
            )
        finally:
            fresh.shutdown()

    def test_malformed_metadata_values_fall_back_to_defaults(self):
        # Import into a non-fresh app: a malformed value resets its key to
        # the DEFAULT (not the stale value, not the garbage); valid keys land.
        path = os.path.join(SCRATCH_BTUI, "cfg_garbage.bt")
        bt.save(self.app.last_net, path, True, bt.CheckpointLevel.Weights,
                metadata="epochs=not_a_number\nlr=-1\nunknown_key=x\noptimizer=sgd")
        fresh = self._dirty_fresh_app()  # epochs 42, lr 0.5, optimizer sgd
        try:
            fresh.handle_key("L")
            type_text(fresh, path)
            fresh.handle_key("ENTER")
            self.assertEqual(fresh.hp_cfg.epochs, 3)    # default, not 42
            self.assertEqual(fresh.hp_cfg.lr, 1e-3)     # default, not 0.5
            self.assertEqual(fresh.hp_cfg.optimizer, "sgd")  # valid one landed
            self.assertEqual(fresh.data_cfg.csv_path, "")    # stale.csv reset
        finally:
            fresh.shutdown()


class VramTest(AppTestCase):
    def test_vram_line_on_gpu_device_only(self):
        app = btui.BtuiApp(btui.BufferScreen(34, 110), has_gpu=True)
        real = btui.bt.gpu_memory
        half_gib = 2 ** 29
        btui.bt.gpu_memory = lambda: {
            "total_bytes": 4 * half_gib,
            "free_bytes": half_gib,
            "used_bytes": 3 * half_gib,
        }
        try:
            app.apply_setting("device", "gpu")
            app.redraw()
            self.assertIn("vram: 1.5/2 GiB", app.screen.text(),
                          app.screen.text())
            app.apply_setting("device", "cpu")
            app.redraw()
            self.assertNotIn("vram:", app.screen.text())
        finally:
            btui.bt.gpu_memory = real
            app.shutdown()


class PauseKeyTest(AppTestCase):
    def test_p_toggle_and_header(self):
        self.app.handle_key("p")
        self.assertTrue(
            any("no run to pause" in line for line in self.app.log_lines),
            f"log lines: {self.app.log_lines}",
        )
        # Simulate a live run: a job double plus the running flag.
        self.app.job = SlowJob()
        self.app.shared.begin_run()
        self.app.handle_key("p")
        self.assertTrue(self.app.shared.paused)
        self.app.redraw()
        self.assertIn("run: paused", self.screen.text())
        self.app.handle_key("p")
        self.assertFalse(self.app.shared.paused)
        self.app.redraw()
        self.assertIn("run: running", self.screen.text())
        self.app.job.release()  # let tearDown's shutdown collect it


if __name__ == "__main__":
    unittest.main()

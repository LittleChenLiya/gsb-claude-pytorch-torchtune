# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Acceptance test for training profiling window control.

Simulates a 10-step CPU training run with a profiling window, cancels at step 6,
and verifies:
  1. Overlapping intervals are merged.
  2. The record interval matches the configured window.
  3. The profiler handle is closed promptly on cancellation.
"""

import os
import tempfile

import torch
from torchtune.training import (
    ProfilingWindow,
    setup_torch_profiler,
    WindowedProfiler,
)


def test_window_merge():
    print("=" * 70)
    print("TEST 1: Overlapping interval merging")
    print("=" * 70)

    window = ProfilingWindow([(2, 5), (4, 8)])
    print(f"  Input intervals:  [(2, 5), (4, 8)]")
    print(f"  Merged intervals: {window.intervals}")
    assert window.intervals == [(2, 8)], f"Expected [(2, 8)], got {window.intervals}"

    window2 = ProfilingWindow([(1, 3), (3, 6)])
    print(f"  Input intervals:  [(1, 3), (3, 6)]")
    print(f"  Merged intervals: {window2.intervals}")
    assert window2.intervals == [(1, 6)], f"Expected [(1, 6)], got {window2.intervals}"

    window3 = ProfilingWindow([(1, 3), (5, 7)])
    print(f"  Input intervals:  [(1, 3), (5, 7)]")
    print(f"  Merged intervals: {window3.intervals}")
    assert window3.intervals == [(1, 3), (5, 7)]

    print("  PASSED\n")


def test_window_schedule():
    print("=" * 70)
    print("TEST 2: Window schedule actions")
    print("=" * 70)

    window = ProfilingWindow([(2, 5)])
    print(f"  Window: {window.intervals}")

    actions = []
    for step in range(8):
        action = window.schedule(step)
        actions.append(action)
        print(f"    step {step}: {action.name}")

    for step in [0, 1, 5, 6, 7]:
        assert actions[step] == torch.profiler.ProfilerAction.NONE, (
            f"step {step} should be NONE, got {actions[step].name}"
        )
    for step in [2, 3]:
        assert actions[step] == torch.profiler.ProfilerAction.RECORD, (
            f"step {step} should be RECORD, got {actions[step].name}"
        )
    assert actions[4] == torch.profiler.ProfilerAction.RECORD_AND_SAVE, (
        f"step 4 should be RECORD_AND_SAVE, got {actions[4].name}"
    )

    print("  PASSED\n")


def test_simulation_with_cancellation():
    print("=" * 70)
    print("TEST 3: 10-step CPU simulation with cancellation at step 6")
    print("=" * 70)

    output_dir = tempfile.mkdtemp(prefix="profiling_window_test_")
    print(f"  Output dir: {output_dir}")

    profiler, profiler_cfg = setup_torch_profiler(
        enabled=True,
        cpu=True,
        cuda=False,
        xpu=False,
        profile_steps=[[2, 5]],
        output_dir=output_dir,
    )

    print(f"  Profiler type: {type(profiler).__name__}")
    print(f"  Profiler config: {profiler_cfg}")
    assert isinstance(profiler, WindowedProfiler), (
        f"Expected WindowedProfiler, got {type(profiler).__name__}"
    )

    recorded_steps = []
    original_step = profiler.step

    def tracked_step():
        if torch.autograd._profiler_enabled():
            recorded_steps.append(profiler._profiler.step_num)
        original_step()

    profiler.step = tracked_step

    cancelled = False
    profiler.start()
    try:
        for step in range(10):
            x = torch.randn(4, 4, requires_grad=True)
            y = x @ x
            loss = y.sum()
            loss.backward()

            profiler.step()

            if step == 6:
                print(f"  Step {step}: cancelling training (KeyboardInterrupt)")
                raise KeyboardInterrupt("Simulated cancellation at step 6")
    except KeyboardInterrupt:
        cancelled = True
        print(f"  Step 6: KeyboardInterrupt caught, entering finally block")
    finally:
        profiler.stop()

    print(f"  Cancelled: {cancelled}")
    assert cancelled, "Expected KeyboardInterrupt to be raised"

    print(f"  Recorded steps (profiler step_num when recording): {recorded_steps}")
    assert len(recorded_steps) > 0, "Expected at least one recorded step"

    print(f"  Profiler stopped flag: {profiler.stopped}")
    assert profiler.stopped, "Profiler should be marked as stopped"
    assert not torch.autograd._profiler_enabled(), (
        "Profiler handle should be closed after stop()"
    )

    profiler.stop()
    profiler.stop()
    print(f"  Idempotent stop() calls succeeded (no double-stop error)")

    trace_files = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            trace_files.append(os.path.join(root, f))
    print(f"  Trace files created: {len(trace_files)}")
    for tf in trace_files:
        print(f"    {tf}")
    assert len(trace_files) > 0, "Expected trace files to be created"

    print("  PASSED\n")


def test_window_outside_no_records():
    print("=" * 70)
    print("TEST 4: No records allocated outside window")
    print("=" * 70)

    output_dir = tempfile.mkdtemp(prefix="profiling_window_outside_")

    profiler, _ = setup_torch_profiler(
        enabled=True,
        cpu=True,
        cuda=False,
        xpu=False,
        profile_steps=[[3, 6]],
        output_dir=output_dir,
    )

    recording_states = []
    profiler.start()
    try:
        for step in range(8):
            x = torch.randn(4, 4, requires_grad=True)
            y = x @ x
            y.sum().backward()
            recording_states.append(torch.autograd._profiler_enabled())
            profiler.step()
    finally:
        profiler.stop()

    print(f"  Recording states (before each step): {recording_states}")
    for step in [0, 1, 2]:
        assert not recording_states[step], (
            f"step {step} should not be recording (outside window)"
        )
    for step in [3, 4, 5]:
        assert recording_states[step], (
            f"step {step} should be recording (inside window)"
        )
    for step in [6, 7]:
        assert not recording_states[step], (
            f"step {step} should not be recording (outside window after close)"
        )

    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TRAINING PROFILING WINDOW CONTROL - ACCEPTANCE TEST")
    print("=" * 70 + "\n")

    test_window_merge()
    test_window_schedule()
    test_simulation_with_cancellation()
    test_window_outside_no_records()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)

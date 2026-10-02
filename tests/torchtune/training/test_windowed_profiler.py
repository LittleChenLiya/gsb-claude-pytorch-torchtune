# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests and the local CPU acceptance simulation for step-windowed (lightweight)
profiling. Run directly for the human-readable acceptance report:

    python tests/torchtune/training/test_windowed_profiler.py
"""

from torchtune.training import ProfileWindow, WindowedProfiler, merge_windows


class FakeRecorder:
    """
    CPU stand-in for ``torch.profiler.profile``. It records the step interval over which
    it was open and tracks whether it was started / stopped. No torch or kineto machinery
    is involved, so windowing behavior is observable directly.
    """

    instances = []

    def __init__(self):
        self.started = False
        self.stopped = False
        self.start_step = None
        self.end_step = None
        FakeRecorder.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    @classmethod
    def reset(cls):
        cls.instances = []

    @classmethod
    def factory(cls):
        def _make():
            return cls()

        return _make


def make_profiler(windows):
    FakeRecorder.reset()
    return WindowedProfiler(windows, recorder_factory=FakeRecorder.factory())


def run_loop(profiler, num_steps=10, cancel_after=None):
    """
    Simulates ``num_steps`` CPU optimizer steps. ``step()`` is called at the top of every
    step; each open recorder stamps the step indices it is alive for. If ``cancel_after``
    is given (1-indexed user step), training is cancelled right after that step's work.
    """
    profiler.start()
    cancelled = False
    for user_step in range(1, num_steps + 1):
        step_idx = user_step - 1
        profiler.step()  # announce optimizer step ``step_idx``
        if profiler.is_recording:
            rec = profiler.active_handle._recorder
            if rec.start_step is None:
                rec.start_step = step_idx
            rec.end_step = step_idx + 1
        if cancel_after is not None and user_step == cancel_after:
            profiler.cancel()
            cancelled = True
            break
    if not cancelled:
        profiler.stop()
    return profiler


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_window_validation():
    import pytest

    with pytest.raises(ValueError):
        ProfileWindow(start_step=-1, end_step=2)
    with pytest.raises(ValueError):
        ProfileWindow(start_step=3, end_step=3)
    with pytest.raises(ValueError):
        ProfileWindow(start_step=4, end_step=2)


def test_merge_windows_overlapping_and_adjacent():
    # Overlapping windows are merged
    assert merge_windows([ProfileWindow(2, 5), ProfileWindow(4, 8)]) == [
        ProfileWindow(2, 8)
    ]

    # Unordered input is sorted before merging
    assert merge_windows([ProfileWindow(7, 9), ProfileWindow(0, 3)]) == [
        ProfileWindow(0, 3),
        ProfileWindow(7, 9),
    ]

    # Adjacent (touching) windows are merged since the behavior is identical
    assert merge_windows([ProfileWindow(2, 4), ProfileWindow(4, 6)]) == [
        ProfileWindow(2, 6)
    ]

    # Containment
    assert merge_windows([ProfileWindow(0, 10), ProfileWindow(2, 4)]) == [
        ProfileWindow(0, 10)
    ]

    assert merge_windows([]) == []
    assert merge_windows([ProfileWindow(1, 2)]) == [ProfileWindow(1, 2)]


def test_no_recorder_allocated_outside_windows():
    profiler = make_profiler([ProfileWindow(2, 4)])
    profiler.start()
    profiler.step()  # step 0
    profiler.step()  # step 1
    assert FakeRecorder.instances == []
    assert not profiler.is_recording
    profiler.step()  # step 2 -> window opens
    assert len(FakeRecorder.instances) == 1
    assert FakeRecorder.instances[0].started
    profiler.stop()
    assert FakeRecorder.instances[0].stopped


def test_full_run_records_each_window_once():
    # [1,3) overlaps [2,4) -> merged [1,4); separate window [7,9)
    profiler = make_profiler(
        [ProfileWindow(1, 3), ProfileWindow(2, 4), ProfileWindow(7, 9)]
    )
    run_loop(profiler, num_steps=10)

    assert profiler.recorded_windows == [ProfileWindow(1, 4), ProfileWindow(7, 9)]
    assert len(FakeRecorder.instances) == 2
    assert [(r.start_step, r.end_step) for r in FakeRecorder.instances] == [
        (1, 4),
        (7, 9),
    ]
    assert all(r.started and r.stopped for r in FakeRecorder.instances)
    assert all(h.closed for h in profiler.handles)
    assert profiler.active_handle is None
    assert not profiler.is_recording


def test_cancel_outside_window_keeps_completed_windows():
    # Acceptance scenario: 10 simulated steps, cancel after the 6th step.
    # Windows: [1,3), [2,4) overlap and merge to [1,4); [7,9) is never reached.
    profiler = make_profiler(
        [ProfileWindow(1, 3), ProfileWindow(2, 4), ProfileWindow(7, 9)]
    )
    run_loop(profiler, num_steps=10, cancel_after=6)

    assert profiler.recorded_windows == [ProfileWindow(1, 4)]
    assert len(FakeRecorder.instances) == 1  # no recorder for the un-reached window
    rec = FakeRecorder.instances[0]
    assert (rec.start_step, rec.end_step) == (1, 4)
    assert rec.stopped
    assert len(profiler.handles) == 1
    assert profiler.handles[0].closed
    assert profiler.active_handle is None
    assert not profiler.is_recording

    # Further steps after cancellation never start another recorder
    profiler.step()
    profiler.step()
    assert len(FakeRecorder.instances) == 1


def test_cancel_during_active_window_closes_recorder_promptly():
    # Window [4,7) is active when training is cancelled after step 6 (index 5):
    # steps 4 and 5 were recorded, step 6 would have been but never runs.
    profiler = make_profiler([ProfileWindow(4, 7)])
    run_loop(profiler, num_steps=10, cancel_after=6)

    assert profiler.recorded_windows == []  # window never completed
    assert len(FakeRecorder.instances) == 1
    rec = FakeRecorder.instances[0]
    assert rec.started and rec.stopped  # closed promptly on cancel
    assert (rec.start_step, rec.end_step) == (4, 6)
    assert profiler.handles[0].closed
    assert profiler.active_handle is None
    assert not profiler.is_recording


def test_context_manager_closes_on_exception():
    profiler = make_profiler([ProfileWindow(0, 10)])
    try:
        with profiler:
            profiler.step()
            raise RuntimeError("simulated training failure")
    except RuntimeError:
        pass
    assert FakeRecorder.instances[0].stopped
    assert profiler.handles[0].closed
    assert profiler.active_handle is None


# ---------------------------------------------------------------------------
# Local CPU acceptance simulation: 10 steps, cancellation after step 6
# ---------------------------------------------------------------------------


def _print_acceptance_report():
    print("=" * 72)
    print("Training profiling window control - CPU acceptance simulation")
    print("10 simulated optimizer steps, training cancelled after step 6")
    print("=" * 72)

    requested = [
        ProfileWindow(1, 3),
        ProfileWindow(2, 4),  # overlaps [1,3)
        ProfileWindow(7, 9),
    ]
    print(
        "\nRequested windows (half-open [start, end)): "
        + str([(w.start_step, w.end_step) for w in requested])
    )
    print(
        "Merged windows: "
        + str([(w.start_step, w.end_step) for w in merge_windows(requested)])
    )

    profiler = make_profiler(requested)
    profiler.start()
    print("\nstep | recorded this step? | recorder events")
    print("-" * 60)
    cancelled = False
    for user_step in range(1, 11):
        step_idx = user_step - 1
        before = len(FakeRecorder.instances)
        profiler.step()
        if profiler.is_recording:
            rec = profiler.active_handle._recorder
            if rec.start_step is None:
                rec.start_step = step_idx
            rec.end_step = step_idx + 1
            state = "RECORD"
        else:
            state = "idle"
        after = len(FakeRecorder.instances)
        event = ""
        if after > before:
            event = "<- recorder allocated at step index %d" % step_idx
        print("  %2d (idx %d) | %-19s | %s" % (user_step, step_idx, state, event))
        if user_step == 6:
            print("-" * 60)
            print("Training cancellation requested after step 6 ...")
            profiler.cancel()
            cancelled = True
            break
    assert cancelled

    recs = FakeRecorder.instances
    print("\n--- Results ---")
    print(
        "recorder handles allocated : %d (none outside windows; "
        "late window [7,9) is never allocated)" % len(recs)
    )
    for i, r in enumerate(recs):
        print(
            "  handle[%d]: recorded step interval %s, started=%s, stopped=%s, "
            "handle.closed=%s"
            % (
                i,
                (r.start_step, r.end_step),
                r.started,
                r.stopped,
                profiler.handles[i].closed,
            )
        )
    print(
        "recorded_windows (completed): "
        + str([(w.start_step, w.end_step) for w in profiler.recorded_windows])
    )
    print("active handle after cancel  : %s" % profiler.active_handle)
    print("is_recording after cancel   : %s" % profiler.is_recording)

    # Assertions that make the behavior difference explicit
    assert merge_windows(requested) == [ProfileWindow(1, 4), ProfileWindow(7, 9)]
    assert len(recs) == 1, "exactly one recorder: [7,9) must not allocate after cancel"
    assert (recs[0].start_step, recs[0].end_step) == (1, 4)
    assert recs[0].stopped and profiler.handles[0].closed
    assert profiler.recorded_windows == [ProfileWindow(1, 4)]
    assert profiler.active_handle is None and not profiler.is_recording
    print("\nAll assertions passed.")

    # Bonus: cancellation inside an active window closes the recorder promptly
    print("\nBonus: cancel during active window [4,7), after step 6")
    p2 = make_profiler([ProfileWindow(4, 7)])
    p2.start()
    for user_step in range(1, 7):
        step_idx = user_step - 1
        p2.step()
        if p2.is_recording:
            rec = p2.active_handle._recorder
            if rec.start_step is None:
                rec.start_step = step_idx
            rec.end_step = step_idx + 1
    p2.cancel()
    r2 = FakeRecorder.instances[0]
    print(
        "  partial record interval: %s, stopped=%s, handle.closed=%s, "
        "completed windows: %s"
        % (
            (r2.start_step, r2.end_step),
            r2.stopped,
            p2.handles[0].closed,
            p2.recorded_windows,
        )
    )
    assert (r2.start_step, r2.end_step) == (4, 6)
    assert r2.stopped and p2.handles[0].closed
    assert p2.recorded_windows == []
    print("Bonus assertions passed.")


if __name__ == "__main__":
    _print_acceptance_report()

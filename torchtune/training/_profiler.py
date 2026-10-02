# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import datetime
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch
import torch.distributed

from omegaconf import DictConfig
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import tensorboard_trace_handler

from torchtune.utils import get_logger, get_world_size_and_rank

log = get_logger("INFO")

PROFILER_KEY = "profiler"
DEFAULT_PROFILER_ACTIVITIES = {
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
    torch.profiler.ProfilerActivity.XPU,
}

DEFAULT_SCHEDULE: dict = {
    "wait_steps": 5,
    "warmup_steps": 5,
    "active_steps": 2,
    "num_cycles": 1,
}

DEFAULT_TRACE_OPTS: dict = {
    "profile_memory": False,
    "with_stack": False,
    "record_shapes": True,
    "with_flops": False,
}

DEFAULT_PROFILE_DIR: str = "profiler_output"


def _warn(msg: str):
    _, rank = get_world_size_and_rank()
    if rank == 0:
        log.warning(msg)


def trace_handler(
    prof: torch.profiler.profile,
    output_dir: str,
    metric: str = "self_cuda_time_total",
    row_limit: int = 25,
):
    """
    Handles export of artifacts from ``torch.profiler.profile``.

    The following artifacts are exported:
    - chrome / tensorboard trace - viewable through tensorboard or perfetto.dev / chrome::/tracing
    - trace event table
    - memory timeline and snapshot.pickle if ``profile_memory``
    - stacks if ``with_stack`` (note that ``profile_memory`` requires ``with_stack`` to be ``True``),
    viewable as a flamegraph see (https://pytorch.org/docs/stable/profiler.html#torch.profiler._KinetoProfile.export_stacks).

    Notes:
    - Each profiling cycle is exported as a sub-directory in output_dir
        - E.g., profiling in 5-step cycle (wait=2, warmup=2, active=1, repeat=0) will result in
        sub-directories iteration_5, iteration_10, etc.
    - If profiling in a distributed setting, each artifact will be prefixed with rank.
    - Memory timeline is only exported for rank 0 (error if exporting from multiple ranks on single node)

    See profiler documentation (https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile) for more details

    Args:
        prof (torch.profiler.profile): instance of torch profiler to use
        output_dir (str):  directory to store artifacts
        metric (str): metric to order trace event table by, see ``torch.profiler.profile.key_averages().table`` for
        row_limit (int): number of rows to display in trace event table

    """
    world_size, rank = get_world_size_and_rank()
    curr_trace_dir_name = "iteration_" + str(prof.step_num)
    curr_trace_dir = os.path.join(output_dir, curr_trace_dir_name)
    if not os.path.exists(curr_trace_dir):
        os.makedirs(curr_trace_dir, exist_ok=True)

    # Export chrome / tensorboard trace
    if rank == 0:
        log.info(f"Dumping traces at step {prof.step_num}")
    begin = time.monotonic()

    # Use tensorboard trace handler rather than directly exporting chrome traces since
    # tensorboard doesn't seem to be able to parse traces with prof.export_chrome_trace

    now = datetime.datetime.now()

    exporter = tensorboard_trace_handler(
        curr_trace_dir,
        worker_name=f"r{rank}-{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}",
        use_gzip=True,
    )
    exporter(prof)

    if rank == 0:
        log.info(f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds")

    # Memory timeline sometimes fails to export
    if prof.profile_memory and torch.cuda.is_available():
        if rank == 0:
            torch.cuda.memory._dump_snapshot(
                f"{curr_trace_dir}/rank{rank}_memory_snapshot.pickle"
            )

    # Dump stack traces
    if prof.with_stack:
        prof.export_stacks(f"{curr_trace_dir}/rank{rank}_stacks.txt", metric=metric)

    # Export event averages
    key_avgs = prof.key_averages(
        group_by_input_shape=prof.record_shapes, group_by_stack_n=5
    ).table(sort_by=metric, row_limit=row_limit)
    with open(f"{curr_trace_dir}/rank{rank}_key_averages.txt", "w") as f:
        print(key_avgs, file=f)
    if rank == 0:
        log.info(f"Saving profiling results to {curr_trace_dir}")

    # TODO: Is this necessary?
    # see https://github.com/pytorch/torchtitan/blob/3050098dcee4901d88c712f9e8e9703d1735a29b/torchtitan/profiling.py#L48
    if world_size > 1:
        torch.distributed.barrier()


class DummyProfiler:
    """
    Drop-in replacement for torch.profiler.profile that functions as a nullcontext / object
    with no-op methods for ``start``, ``stop``, and ``step``.

    This is helpful for instrumenting profiling in a recipe without requiring changes to the
    code independent of whether profiling is on / off.

    E.g.,
    ```
        profiler = DummyProfiler()
        #profiler = torch.profiler.profile()

        # Below is same regardless of profiler object type
        with profiler as prof:
            for epoch in epochs:
                for batch in batches:
                    train.step()
                    prof.step()

    """

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def step(self):
        pass

    def cancel(self):
        pass


@dataclass(frozen=True)
class ProfileWindow:
    """
    Half-open step interval ``[start_step, end_step)`` during which profiling is active.

    Steps are counted the same way as ``torch.profiler.profile.step_num``: ``step()`` is
    called at the top of an optimizer step and advances the counter from ``n`` to ``n + 1``,
    so the *upcoming* step is indexed ``n``. A window of ``ProfileWindow(start_step=2,
    end_step=4)`` records optimizer steps 2 and 3 only.

    Attributes:
        start_step (int): First (0-indexed) optimizer step that is recorded.
        end_step (int): Step index at which recording stops (exclusive).
    """

    start_step: int
    end_step: int

    def __post_init__(self):
        if self.start_step < 0:
            raise ValueError(
                f"ProfileWindow.start_step must be non-negative, got {self.start_step}"
            )
        if self.end_step <= self.start_step:
            raise ValueError(
                "ProfileWindow.end_step must be strictly greater than start_step, "
                f"got start_step={self.start_step}, end_step={self.end_step}"
            )

    def contains(self, step: int) -> bool:
        return self.start_step <= step < self.end_step

    def overlaps(self, other: "ProfileWindow") -> bool:
        # Intervals are half-open, so touching boundaries (a.end == b.start) do not
        # overlap but are still eligible for merging in ``merge_windows``.
        return self.start_step < other.end_step and other.start_step < self.end_step


def merge_windows(windows: Sequence[ProfileWindow]) -> list[ProfileWindow]:
    """
    Merges overlapping (and directly adjacent) profiling windows into a sorted list of
    disjoint windows.

    ``[2, 5)`` and ``[4, 8)`` merge into ``[2, 8)``; ``[2, 4)`` and ``[4, 6)`` merge into
    ``[2, 6)`` as well since the resulting recording behavior is identical. Empty input
    returns an empty list.
    """
    if len(windows) == 0:
        return []
    sorted_windows = sorted(windows, key=lambda w: w.start_step)
    merged = [sorted_windows[0]]
    for window in sorted_windows[1:]:
        current = merged[-1]
        if window.start_step <= current.end_step:
            merged[-1] = ProfileWindow(
                current.start_step, max(current.end_step, window.end_step)
            )
        else:
            merged.append(window)
    return merged


class _FakeRecordingHandle:
    """Handle returned by recorders; ``stop()`` is idempotent."""

    def __init__(self, recorder: Any):
        self._recorder = recorder
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def stop(self):
        if self._closed:
            return
        self._closed = True
        self._recorder.stop()


def torch_profiler_recorder_factory(
    activities: Optional[Sequence[torch.profiler.ProfilerActivity]] = None,
    profile_memory: bool = False,
    with_stack: bool = False,
    record_shapes: bool = False,
    with_flops: bool = False,
    on_trace_ready: Optional[Callable[..., Any]] = None,
) -> Callable[[], Any]:
    """
    Builds a zero-argument factory that lazily creates fresh ``torch.profiler.profile``
    instances. A new instance is created *only* when a profiling window opens, so no
    profiler / kineto records are allocated while running outside of the windows.

    Windows are short and flat (no wait/warmup/active cycle), so the created profilers do
    not take a ``schedule`` and must be explicitly stopped (``WindowedProfiler`` does this
    on window end or on cancellation). If ``on_trace_ready`` is provided it is invoked by
    torch profiler on ``stop()``.
    """

    if activities is None:
        activities = [torch.profiler.ProfilerActivity.CPU]

    def factory():
        return torch.profiler.profile(
            activities=list(activities),
            profile_memory=profile_memory,
            with_stack=with_stack,
            record_shapes=record_shapes,
            with_flops=with_flops,
            on_trace_ready=on_trace_ready,
        )

    return factory


class WindowedProfiler:
    """
    Lightweight, step-windowed wrapper around a recorder (by default ``torch.profiler.profile``).

    Recorders are created lazily and exist **only** for optimizer steps that fall inside a
    configured profiling window. Outside of every window, no recorder is constructed and
    nothing is recorded. Overlapping (or adjacent) windows are merged up front, so each
    step is recorded at most once.

    Windows are half-open ``[start_step, end_step)`` intervals, indexed identically to
    ``torch.profiler.profile.step_num``. Typical instrumentation mirrors
    :class:`DummyProfiler`::

        profiler = WindowedProfiler([ProfileWindow(2, 5)])
        with profiler:
            for batch_idx, batch in enumerate(dataloader):
                profiler.step()  # announce optimizer step `batch_idx`
                train.step(batch)
        # if the run is cancelled early instead:
        profiler.cancel()

    Args:
        windows (Sequence[ProfileWindow]): Step intervals to profile. Overlapping windows
            are merged.
        recorder_factory (Optional[Callable[[], Any]]): Zero-argument callable returning a
            recorder with ``start()`` / ``stop()`` methods. Defaults to a lightweight CPU
            ``torch.profiler.profile`` factory. Pass a custom factory (e.g. a fake
            recorder in tests) to observe windowing behavior without kineto overhead.
    """

    def __init__(
        self,
        windows: Sequence[ProfileWindow],
        recorder_factory: Optional[Callable[[], Any]] = None,
    ):
        self.windows: list[ProfileWindow] = merge_windows(windows)
        self._recorder_factory = (
            recorder_factory
            if recorder_factory is not None
            else torch_profiler_recorder_factory()
        )
        # Step that will run next; `step()` announces this step before it executes.
        self._step_num = 0
        # Index into self.windows of the earliest window not yet entered.
        self._cursor = 0
        self._recorder: Any = None
        self._handle: Optional[_FakeRecordingHandle] = None
        # Handles retained after their recorder stops, in window order.
        self.handles: list[_FakeRecordingHandle] = []
        # Windows whose entire interval ran while training was alive.
        self._completed_windows: list[ProfileWindow] = []
        self._cancelled = False
        self._started = False

    @property
    def step_num(self) -> int:
        return self._step_num

    @property
    def is_recording(self) -> bool:
        return self._recorder is not None

    @property
    def active_handle(self) -> Optional[_FakeRecordingHandle]:
        return self._handle

    @property
    def recorded_windows(self) -> list[ProfileWindow]:
        """Windows whose full ``[start_step, end_step)`` interval was recorded."""
        return list(self._completed_windows)

    def start(self) -> "WindowedProfiler":
        self._started = True
        return self

    def __enter__(self) -> "WindowedProfiler":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Ensure no recorder is left running, including on abnormal exits.
        self.cancel()
        return False

    def _open_window(self):
        self._recorder = self._recorder_factory()
        self._recorder.start()
        self._handle = _FakeRecordingHandle(self._recorder)

    def _close_window(self, completed: bool):
        window = self.windows[self._cursor]
        if self._handle is not None:
            self._handle.stop()
            self.handles.append(self._handle)
        if completed:
            self._completed_windows.append(window)
        self._handle = None
        self._recorder = None
        self._cursor += 1

    def step(self):
        """
        Announce that optimizer step :attr:`step_num` is about to run and toggle the
        recorder according to the merged windows. Must be called at the *top* of each
        optimizer step, before the work of that step executes.
        """
        if not self._started or self._cancelled:
            self._step_num += 1
            return

        current = self._step_num

        # Close the active window when the current step is past its end.
        if self._recorder is not None:
            active_window = self.windows[self._cursor]
            if current >= active_window.end_step:
                self._close_window(completed=True)

        # Advance past windows that were entirely skipped (possible if `step()` was not
        # called for some steps); such windows produce no recorder or handle.
        while self._cursor < len(self.windows):
            window = self.windows[self._cursor]
            if current >= window.end_step:
                self._cursor += 1
                continue
            break

        # Open the next window if the current step entered it.
        if self._recorder is None and self._cursor < len(self.windows):
            window = self.windows[self._cursor]
            if current >= window.start_step:
                self._open_window()

        self._step_num += 1

    def cancel(self):
        """
        Stop an in-flight recorder immediately when training is cancelled.

        The partially recorded window is not reported in :attr:`recorded_windows`; its
        closed handle is still retained in :attr:`handles`. Calling ``step()`` after
        ``cancel()`` is a no-op with respect to recording, and ``cancel()`` is safe to
        call repeatedly.
        """
        self._cancelled = True
        if self._recorder is not None:
            active_window = self.windows[self._cursor]
            # If every step of the window was already announced, the window effectively
            # completed even without a final `step()` to close it.
            completed = self._step_num >= active_window.end_step
            self._close_window(completed=completed)

    def stop(self):
        """Clean shutdown: closes any active window as if it completed normally."""
        if self._recorder is not None:
            self._close_window(completed=True)
        self._cancelled = True


def setup_torch_profiler(
    enabled: bool = False,
    cpu: bool = True,
    cuda: bool = True,
    xpu: bool = True,
    hpu: bool = False,
    profile_memory: bool = DEFAULT_TRACE_OPTS["profile_memory"],
    with_stack: bool = DEFAULT_TRACE_OPTS["with_stack"],
    record_shapes: bool = DEFAULT_TRACE_OPTS["record_shapes"],
    with_flops: bool = DEFAULT_TRACE_OPTS["with_flops"],
    # `torch.profiler.schedule` args - note we defer setting these to enable more fine-grained
    # warnings within this setup function
    wait_steps: Optional[int] = None,
    warmup_steps: Optional[int] = None,
    active_steps: Optional[int] = None,
    num_cycles: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> tuple[torch.profiler.profile, DictConfig]:
    """
    Sets up :class:`~torch.profiler.profile` and returns the profiler config with post-setup updates.

    The profiler config can be provided in configs under the ``profiler`` key with the following layout:

    .. code-block:: yaml

        profiler:
          _component_: torchtune.training.setup_torch_profiler
          enabled: bool
          # Output directory of trace artifacts
          output_dir: str

          # torch.profiler.ProfilerActivity types to trace
          cpu: bool
          cuda: bool

          # Trace options
          profile_memory: bool
          with_stack: bool
          record_shapes: bool
          with_flops: bool

          # torch.profiler.schedule args
          wait_steps: int
          warmup_steps: int
          active_steps: int
          num_cycles: int

    The profiler schedule updates with respect to an optimizer step (e.g., if
    ``gradient_accumulation = 2``, then the profiler will step every 2 batches).

    Sensible defaults will be chosen if the config is missing options:

    - If no activities are specified, profiler will default to CPU + CUDA
    - If no schedule is specified, profiler will default to ``DEFAULT_SCHEDULE``
    - Certain options will be overridden (``with_stack`` and ``record_shapes``) \
    depending on requirements of other options (e.g., ``profile_memory`` requires \
    ``with_stack`` and ``record_shapes``).


    Note:
        - Enabling the profiler will result in training speed reduction.
        - Setting ``profile_memory: True`` will generate large trace files.
        - The profiler schedule is context dependent. Calling ``profiler.step()`` \
        at each batch iteration but **outside** the gradient accumulation scope will \
        ``step`` the profiler each forward / backward step. Calling ``profiler.step()`` \
        each batch iteration but **within** the gradient accumulation scope  will ``step`` \
        the profiler each optimizer update step such that each ``step`` contains multiple \
        forward / backward passes.

    Args:
        enabled (bool): Enable pytorch profiler. Default is False.
        cpu (bool): Enable cpu profiling. Default is True.
        cuda (bool): Enable cuda profiling. Default is True.
        xpu (bool): Enable xpu profiling. Default is True.
        hpu (bool): Enable hpu profiling. Default is False.
        profile_memory (bool): Profile memory usage. Default is False.
        with_stack (bool): Profile stack. Default is False.
        record_shapes (bool): Record shapes. Default is True.
        with_flops (bool): Profile flops. Default is False.
        wait_steps (Optional[int]): Wait time in steps. Maps to ``wait`` kwarg of ``torch.profiler.schedule``.
        warmup_steps (Optional[int]): Warmup time in steps. Maps to ``warmup`` kwarg of ``torch.profiler.schedule``.
        active_steps (Optional[int]): Active time in steps. Maps to ``active`` kwarg of ``torch.profiler.schedule``.
        num_cycles (Optional[int]): Number of profiling cycles. Maps to ``repeat`` kwarg of ``torch.profiler.schedule``.
        output_dir (Optional[str]): Tracing file output path.

    Returns:
        tuple[torch.profiler.profile, DictConfig]
    """

    if not enabled:
        _warn(" Profiling disabled.")
        return DummyProfiler(), DictConfig({"enabled": False})

    # Set up profiler activities
    activities = []
    if cpu:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    if xpu:
        activities.append(torch.profiler.ProfilerActivity.XPU)
    if hpu:
        activities.append(torch.profiler.ProfilerActivity.HPU)
    if len(activities) == 0:
        _warn("No activities specified, defaulting to CPU + CUDA")
        activities = DEFAULT_PROFILER_ACTIVITIES
        cpu = cuda = xpu = True

    # Check for schedule
    # 1) If no schedule is provided, set to DEFAULT_SCHEDULE
    # 2) else check for missing keys and warn if any are missing, setting these to defaults
    # Note that this might result in code duplication if these checks are already done in the `recipe`
    # However, we retain this checks in the case that the _setup_profiler section of the `recipe` does not implement these checks

    # Set up profiler schedule
    use_default_schedule = not any(
        [
            wait_steps is not None,
            warmup_steps is not None,
            active_steps is not None,
            num_cycles is not None,
        ]
    )

    # Use default schedule if None, else validate that schedule is valid and can be passed to `instantiate`
    if use_default_schedule:
        schedule_args = DEFAULT_SCHEDULE
        _warn(
            " No schedule found in config, defaulting to {}".format(
                ", ".join(f"{k} = {schedule_args[k]}" for k in schedule_args.keys())
            )
        )
    else:
        schedule_args = {
            "wait_steps": wait_steps,
            "warmup_steps": warmup_steps,
            "active_steps": active_steps,
            "num_cycles": num_cycles,
        }
        missing_keys = [k for k in schedule_args.keys() if schedule_args[k] is None]
        if len(missing_keys) > 0:
            for k in missing_keys:
                schedule_args[k] = DEFAULT_SCHEDULE[k]
            _warn(
                " Missing keys in torch profiler schedule {}: defaulting to {}".format(
                    ", ".join(missing_keys),
                    ", ".join(f"{k} = {schedule_args[k]}" for k in missing_keys),
                )
            )
    schedule = torch.profiler.schedule(
        wait=schedule_args["wait_steps"],
        warmup=schedule_args["warmup_steps"],
        active=schedule_args["active_steps"],
        repeat=schedule_args["num_cycles"],
    )

    # profile_memory requires with_stack and record_shapes, hence we override these if profile_memory is True
    # See torch.profiler.profiler._memory_profile
    if profile_memory:
        _warn(
            "`profile_memory` requires `with_stack` and `record_shapes`, these will be enabled since `profile_memory` is True"
        )
    with_stack = with_stack or profile_memory
    record_shapes = record_shapes or profile_memory
    # experimental config is needed to export stacks: see https://github.com/pytorch/pytorch/issues/100253
    experimental_config = _ExperimentalConfig(verbose=True) if with_stack else None

    # Handle exporting of trace, memory timeline and other profiler artifacts
    if output_dir is None:
        _warn(
            f" No output directory found in profiler config, defaulting to {DEFAULT_PROFILE_DIR}"
        )
        output_dir = DEFAULT_PROFILE_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir)

    # trace_handler manages the export of profiler artifacts
    # this callback will be triggered after **each** profiling cycle
    callback = partial(trace_handler, output_dir=output_dir)

    profiler = torch.profiler.profile(
        activities=activities,
        profile_memory=profile_memory,
        with_stack=with_stack,
        record_shapes=record_shapes,
        with_flops=with_flops,
        schedule=schedule,
        experimental_config=experimental_config,
        on_trace_ready=callback,
    )

    profiler_cfg = DictConfig(
        {
            "enabled": enabled,
            "output_dir": output_dir,
            "cpu": cpu,
            "cuda": cuda,
            "xpu": xpu,
            "hpu": hpu,
            "profile_memory": profile_memory,
            "with_stack": with_stack,
            "record_shapes": record_shapes,
            "with_flops": with_flops,
            **schedule_args,
        }
    )

    return (profiler, profiler_cfg)

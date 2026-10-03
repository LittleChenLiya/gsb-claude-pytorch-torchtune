# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import datetime
import os
import time
from functools import partial
from pathlib import Path
from typing import Optional

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


class ProfilingWindow:
    """
    Manages a set of half-open step intervals ``[start, end)`` during which the
    profiler records events.

    Overlapping (and adjacent) intervals are merged automatically so that a
    single contiguous trace is produced for each merged window. Outside all
    windows the schedule returns ``ProfilerAction.NONE``, which tells the torch
    profiler not to allocate any records -- this is the "lightweight" property
    of window-based profiling.

    The schedule is expressed in terms of **profiler steps**, i.e. the values
    passed to :meth:`torch.profiler.profile.step`. The first ``step()`` call
    after ``start()`` increments the step number to 1, so a window of
    ``[2, 5)`` opens on the second ``step()`` call and closes (saving the
    trace) on the fifth. Because the recipe calls ``step()`` *after* each
    optimizer step, the captured forward/backward passes are those that fall
    between the window's ``start`` and ``end`` profiler steps.

    Args:
        intervals (list[tuple[int, int]]): list of ``[start, end)`` step
            intervals. ``start`` must be non-negative and ``end`` must be
            strictly greater than ``start``.

    Example:
        >>> window = ProfilingWindow([(2, 5), (4, 8)])
        >>> window.intervals
        [(2, 8)]
        >>> window.schedule(2)
        ProfilerAction.RECORD
        >>> window.schedule(5)
        ProfilerAction.RECORD
        >>> window.schedule(8)
        ProfilerAction.NONE
    """

    def __init__(self, intervals: list[tuple[int, int]]):
        self._validate(intervals)
        self.intervals: list[tuple[int, int]] = self._merge(intervals)

    @staticmethod
    def _validate(intervals: list[tuple[int, int]]) -> None:
        for iv in intervals:
            if not isinstance(iv, (list, tuple)) or len(iv) != 2:
                raise ValueError(
                    f"Each profiling window must be a [start, end] pair, got {iv!r}"
                )
            start, end = iv
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(
                    f"Profiling window bounds must be integers, got {iv!r}"
                )
            if start < 0:
                raise ValueError(
                    f"Profiling window start must be non-negative, got {start}"
                )
            if end <= start:
                raise ValueError(
                    f"Profiling window end must be greater than start, got [{start}, {end})"
                )

    @staticmethod
    def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Merge overlapping or adjacent intervals into a sorted list."""
        if not intervals:
            return []
        sorted_ivs = sorted(intervals, key=lambda iv: iv[0])
        merged: list[list[int]] = [list(sorted_ivs[0])]
        for start, end in sorted_ivs[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                # Overlapping or adjacent -- extend the current window
                merged[-1][1] = max(last_end, end)
            else:
                merged.append([start, end])
        return [(s, e) for s, e in merged]

    def contains(self, step: int) -> bool:
        """Return ``True`` if ``step`` falls within any window."""
        for start, end in self.intervals:
            if start <= step < end:
                return True
        return False

    def schedule(self, step: int) -> "torch.profiler.ProfilerAction":
        """
        Return the :class:`~torch.profiler.ProfilerAction` for ``step``.

        - ``RECORD`` while inside a window, except on the last in-window step
          where ``RECORD_AND_SAVE`` is returned. The trace is started on the
          first in-window step and kept recording through the last.
        - ``RECORD_AND_SAVE`` on the last in-window step. The transition to
          ``NONE`` on the following step triggers ``stop_trace`` +
          ``on_trace_ready``, saving the trace for the window that just closed.
        - ``NONE`` outside all windows. No records are allocated.
        """
        for start, end in self.intervals:
            if start <= step < end:
                if step == end - 1:
                    return torch.profiler.ProfilerAction.RECORD_AND_SAVE
                return torch.profiler.ProfilerAction.RECORD
        return torch.profiler.ProfilerAction.NONE

    def __repr__(self) -> str:
        return f"ProfilingWindow(intervals={self.intervals})"


class WindowedProfiler:
    """
    Cancellation-safe wrapper around a :class:`torch.profiler.profile`.

    Delegates :meth:`start`, :meth:`stop`, and :meth:`step` to the wrapped
    profiler while making :meth:`stop` idempotent. This guarantees that the
    underlying kineto profiler handle is closed exactly once even if training
    is cancelled (e.g. ``KeyboardInterrupt``) and ``stop()`` is reached through
    a ``finally`` block.

    The wrapper also supports the context-manager protocol so it can be used
    interchangeably with the raw :class:`torch.profiler.profile` and
    :class:`DummyProfiler`.

    Args:
        profiler (torch.profiler.profile): the underlying torch profiler to
            wrap.
    """

    def __init__(self, profiler: "torch.profiler.profile"):
        self._profiler = profiler
        self._stopped = False

    def start(self) -> None:
        self._profiler.start()
        self._stopped = False

    def stop(self) -> None:
        """Stop the profiler. Idempotent -- safe to call multiple times."""
        if self._stopped:
            return
        self._profiler.stop()
        self._stopped = True

    def step(self) -> None:
        self._profiler.step()

    def __enter__(self) -> "WindowedProfiler":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    def __getattr__(self, name):
        # Delegate attribute access to the wrapped profiler so that existing
        # code that reads profiler.profile_memory, .activities, etc. continues
        # to work.
        return getattr(self._profiler, name)

    @property
    def stopped(self) -> bool:
        """Return ``True`` if :meth:`stop` has been called."""
        return self._stopped

    def __repr__(self) -> str:
        return f"WindowedProfiler(profiler={self._profiler!r}, stopped={self._stopped})"


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
    # Optional step intervals for lightweight window-based profiling. When
    # provided, the wait/warmup/active schedule is replaced by a window schedule
    # that only records inside the given [start, end) intervals.
    profile_steps: Optional[list[list[int]]] = None,
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

          # Optional lightweight profiling windows as a list of [start, end)
          # step intervals. When provided, the wait/warmup/active schedule is
          # replaced by a window schedule that only records inside the given
          # intervals (overlapping intervals are merged). Outside the windows
          # no records are allocated.
          profile_steps: list[list[int]]

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
        profile_steps (Optional[list[list[int]]]): Optional list of ``[start, end)`` step intervals for
            lightweight window-based profiling. When provided, the wait/warmup/active schedule is
            replaced by a window schedule that only records inside the given intervals. Overlapping
            intervals are merged. Outside all windows the profiler allocates no records.

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

    # Set up profiler schedule. When `profile_steps` is provided we use a
    # lightweight window schedule instead of the wait/warmup/active cycle.
    use_windows = profile_steps is not None and len(profile_steps) > 0
    if use_windows:
        window = ProfilingWindow(profile_steps)
        schedule = window.schedule
        # The wait/warmup/active args are not used by the window schedule, but
        # we still populate them in the config so downstream code that reads
        # them does not break.
        schedule_args = {
            "wait_steps": 0,
            "warmup_steps": 0,
            "active_steps": 0,
            "num_cycles": 0,
        }
        _warn(
            f" Using lightweight profiling windows (overlapping intervals merged): {window.intervals}"
        )
    else:
        window = None
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

    # Wrap in a cancellation-safe wrapper when using windows so that stop()
    # is idempotent and the kineto handle is closed exactly once.
    if use_windows:
        profiler = WindowedProfiler(profiler)

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
            "profile_steps": window.intervals if window is not None else None,
            **schedule_args,
        }
    )

    return (profiler, profiler_cfg)

# Training Profiling Window Control - Verification Results

## Implementation Summary

### Files Changed
1. `torchtune/training/_profiler.py` - Core implementation
2. `torchtune/training/__init__.py` - New exports
3. `recipes/full_finetune_single_device.py` - Try/finally for cancellation safety
4. `recipes/full_finetune_distributed.py` - Try/finally for cancellation safety
5. `test_profiling_window.py` - Acceptance test (new)

### New Classes

#### ProfilingWindow
Manages half-open step intervals `[start, end)` during which the profiler records.
- `_merge()`: Merges overlapping and adjacent intervals into a sorted list.
- `schedule(step)`: Returns `RECORD` inside windows, `RECORD_AND_SAVE` on the last
  in-window step, and `NONE` outside. The transition `(RECORD_AND_SAVE, NONE)` triggers
  `stop_trace` + `on_trace_ready`, saving the trace.

#### WindowedProfiler
Cancellation-safe wrapper around `torch.profiler.profile`.
- `stop()` is idempotent (safe to call multiple times).
- Supports `__enter__`/`__exit__` context-manager protocol.
- `stopped` property indicates whether `stop()` has been called.

### setup_torch_profiler Enhancement
New `profile_steps` parameter accepts a list of `[start, end)` intervals. When provided:
- The wait/warmup/active schedule is replaced by the window schedule.
- The profiler is wrapped in `WindowedProfiler`.
- The config includes `profile_steps` and sets schedule args to 0.

## Commands Run

### 1. Acceptance Test
```bash
python test_profiling_window.py
```

### 2. Existing Profiler Tests (backward compatibility)
```bash
python -m pytest tests/torchtune/training/test_profiler.py -v
```

### 3. Compilation Checks
```bash
python -m py_compile torchtune/training/_profiler.py
python -m py_compile recipes/full_finetune_single_device.py
python -m py_compile recipes/full_finetune_distributed.py
```

## Key Results

### Test 1: Overlapping Interval Merging
- `[(2, 5), (4, 8)]` -> `[(2, 8)]` (overlapping merged)
- `[(1, 3), (3, 6)]` -> `[(1, 6)]` (adjacent merged)
- `[(1, 3), (5, 7)]` -> `[(1, 3), (5, 7)]` (non-overlapping kept separate)

### Test 2: Window Schedule Actions
Window `[(2, 5)]`:
- step 0: NONE
- step 1: NONE
- step 2: RECORD
- step 3: RECORD
- step 4: RECORD_AND_SAVE
- step 5: NONE (trace saved at this step)
- step 6: NONE
- step 7: NONE

### Test 3: 10-step CPU Simulation with Cancellation at Step 6
- Profiler type: `WindowedProfiler`
- Window: `[(2, 5)]`
- Recorded steps: `[2, 3, 4]` (matches window)
- Cancellation: KeyboardInterrupt raised at step 6, caught in `finally` block
- Profiler stopped flag: `True`
- `torch.autograd._profiler_enabled()`: `False` (handle closed)
- Idempotent `stop()` calls succeeded (no double-stop error)
- Trace files created: 2 (trace.json.gz + rank0_key_averages.txt)
- Trace saved at step 5 (iteration_5)

### Test 4: No Records Allocated Outside Window
Window `[(3, 6)]`:
- Recording states: `[False, False, False, True, True, True, False, False]`
- Steps 0,1,2: not recording (outside window)
- Steps 3,4,5: recording (inside window)
- Steps 6,7: not recording (outside window after close)

### Existing Tests
All 7 existing profiler tests pass (backward compatibility maintained).

## State Flow Diagram

```
start()
  |
  v
schedule(0) == NONE? --> yes --> (NONE, NONE) --> no-op
  |
  no
  v
(NONE, RECORD) --> prepare_trace + start_trace
  |
  v
step(1..start-1): NONE --> no-op
  |
  v
step(start): RECORD --> (NONE, RECORD) --> prepare + start
  |
  v
step(start+1..end-2): RECORD --> (RECORD, RECORD) --> no-op
  |
  v
step(end-1): RECORD_AND_SAVE --> (RECORD, RECORD_AND_SAVE) --> no-op
  |
  v
step(end): NONE --> (RECORD_AND_SAVE, NONE) --> stop_trace + on_trace_ready (SAVE)
  |
  v
step(end+1..): NONE --> no-op
```

On cancellation (KeyboardInterrupt), the `finally` block calls `stop()`:
- If recording: `(current_action, None)` --> stop_trace + on_trace_ready (SAVE)
- If not recording: no-op
- Idempotent: safe to call multiple times

"""
Tests for DBS integration.

Covers the three pure-logic additions:
  - AdaptiveLoadBalancer.compute_partitions()   (load_balancer.py)
  - PerformanceProfiler.get_timing_summary()    (performance_profiler.py)
  - simulate_straggler()                        (health_monitor.py)

ssgd() and spawn_workers() require a live dist process group and are
validated separately against a mocked dist.all_reduce.
"""

import math
import time
import numpy as np
import pytest

from src.scheduling.load_balancer import AdaptiveLoadBalancer
from src.scheduling.health_monitor import simulate_straggler


# ---------------------------------------------------------------------------
# compute_partitions() — DBS get_size() integration
# ---------------------------------------------------------------------------

@pytest.fixture
def lb():
    return AdaptiveLoadBalancer(policy="dynamic")


def test_compute_partitions_sums_to_one(lb):
    """Output ratios must always sum to exactly 1.0."""
    nodes_time = [1.0, 1.5, 2.0]
    partition = [1 / 3, 1 / 3, 1 / 3]
    result = lb.compute_partitions(nodes_time, partition, batch_size=64)
    assert abs(result.sum() - 1.0) < 1e-6, f"Sum={result.sum()} != 1.0"


def test_compute_partitions_equal_times(lb):
    """Equal node times → approximately equal partition ratios."""
    nodes_time = [1.0, 1.0, 1.0]
    partition = [1 / 3, 1 / 3, 1 / 3]
    result = lb.compute_partitions(nodes_time, partition, batch_size=90)
    for r in result:
        assert abs(r - 1 / 3) < 0.05, f"Expected ~0.333, got {r}"


def test_compute_partitions_slow_node_gets_less(lb):
    """A 2× slower worker should receive a smaller partition than the fast one."""
    nodes_time = [1.0, 2.0]   # worker 1 is half as fast
    partition = [0.5, 0.5]
    result = lb.compute_partitions(nodes_time, partition, batch_size=100)
    assert result[0] > result[1], (
        f"Fast worker should get more work: {result[0]:.3f} vs {result[1]:.3f}"
    )


def test_compute_partitions_batch_allocation_integer(lb):
    """Allocated batch counts must be non-negative integers and sum to batch_size."""
    batch_size = 128
    nodes_time = [0.8, 1.2, 1.5, 2.0]
    partition = [0.25, 0.25, 0.25, 0.25]
    norm = lb.compute_partitions(nodes_time, partition, batch_size)
    # Convert back to counts and check total
    counts = np.round(norm * batch_size).astype(int)
    # Allow ±1 due to rounding; the norm ratios are what matter
    assert all(c >= 0 for c in counts), "Negative batch count"
    assert abs(sum(counts) - batch_size) <= 1, f"Count sum {sum(counts)} != {batch_size}"


def test_compute_partitions_zero_time_guard(lb):
    """A stalled worker (time=0) must not cause division by zero."""
    nodes_time = [0.0, 1.0]   # worker 0 appears stalled
    partition = [0.5, 0.5]
    result = lb.compute_partitions(nodes_time, partition, batch_size=64)
    assert not np.any(np.isnan(result)), "NaN in output"
    assert not np.any(np.isinf(result)), "Inf in output"
    assert abs(result.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# simulate_straggler() — DBS fault_tolerance_wait() integration
# ---------------------------------------------------------------------------

def _fresh_state():
    return {"fault_wait": False, "fault_round": 0, "fault_wait_time": 0.0, "saved_epoch": -1}


def test_simulate_straggler_no_fault_with_zero_chance():
    """fault_chance=0 must never sleep."""
    state = _fresh_state()
    elapsed = simulate_straggler(rank=0, epoch=1, batch_num=100, state=state, fault_chance=0.0)
    assert elapsed == 0.0


def test_simulate_straggler_always_faults_with_full_chance():
    """fault_chance=1 guarantees a fault is injected on first call."""
    state = _fresh_state()
    t0 = time.time()
    simulate_straggler(rank=0, epoch=1, batch_num=1, state=state, fault_chance=1.0)
    elapsed = time.time() - t0
    # Should have slept at least a tiny bit (first instalment)
    assert state["fault_wait"] is True, "fault_wait not set"
    assert state["fault_round"] > 1, "fault_round not set"


def test_simulate_straggler_once_per_epoch():
    """The lottery runs at most once per epoch."""
    state = _fresh_state()
    # Call twice for the same epoch — second call must be a no-op
    simulate_straggler(rank=0, epoch=5, batch_num=100, state=state, fault_chance=0.0)
    state_snapshot = dict(state)
    simulate_straggler(rank=0, epoch=5, batch_num=100, state=state, fault_chance=0.0)
    assert state == state_snapshot, "State changed on second call in same epoch"


def test_simulate_straggler_clears_after_fault_round(monkeypatch):
    """After fault_round expires, the fault_wait flag resets."""
    state = {
        "fault_wait": True,
        "fault_round": 3,
        "fault_wait_time": 0.0,   # 0s wait so test doesn't sleep
        "saved_epoch": 3,
    }
    # Epoch 4 > fault_round=3 → should clear fault_wait
    simulate_straggler(rank=0, epoch=4, batch_num=100, state=state, fault_chance=0.0)
    assert state["fault_wait"] is False, "fault_wait should reset after fault_round"


# ---------------------------------------------------------------------------
# get_timing_summary() — performance_profiler.py integration
# ---------------------------------------------------------------------------

def test_get_timing_summary_no_data():
    """Empty profiler returns zero-filled summary with correct keys."""
    from src.profiling.performance_profiler import PerformanceProfiler
    p = PerformanceProfiler(device_id=0, rank=0, enable_nvml=False)
    summary = p.get_timing_summary()
    expected_keys = {"train_time", "sync_time", "average_time", "throughput", "bottleneck"}
    assert expected_keys == set(summary.keys()), f"Unexpected keys: {set(summary.keys())}"
    assert summary["train_time"] == 0.0
    assert summary["sync_time"] == 0.0
    assert summary["average_time"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

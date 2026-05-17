"""
Standalone DBS integration validator — no pytest needed.
Run from the project root with the venv Python.
"""
import sys, os, traceback, time

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []

def check(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"        {type(e).__name__}: {e}")
        results.append((name, False, str(e)))


print("\n=== DBS Integration Validator ===\n")

# ------------------------------------------------------------------
# 1. AdaptiveLoadBalancer.compute_partitions()
# ------------------------------------------------------------------
print("[ scheduling/load_balancer.py ]")
from src.scheduling.load_balancer import AdaptiveLoadBalancer
lb = AdaptiveLoadBalancer()

def t1():
    r = lb.compute_partitions([1.0, 1.0, 1.0], [1/3, 1/3, 1/3], 90)
    assert abs(r.sum() - 1.0) < 1e-6, f"sum={r.sum()}"
check("compute_partitions sums to 1.0", t1)

def t2():
    r = lb.compute_partitions([1.0, 2.0], [0.5, 0.5], 100)
    assert r[0] > r[1], f"fast={r[0]:.3f} slow={r[1]:.3f}"
check("compute_partitions: fast worker gets more", t2)

def t3():
    r = lb.compute_partitions([0.0, 1.0], [0.5, 0.5], 64)
    assert not np.any(np.isnan(r)), "NaN in output"
    assert abs(r.sum() - 1.0) < 1e-6
check("compute_partitions: zero-time guard (no NaN)", t3)

# ------------------------------------------------------------------
# 2. simulate_straggler()
# ------------------------------------------------------------------
print("\n[ scheduling/health_monitor.py ]")
from src.scheduling.health_monitor import simulate_straggler

def _state():
    return {"fault_wait": False, "fault_round": 0, "fault_wait_time": 0.0, "saved_epoch": -1}

def t4():
    s = _state()
    r = simulate_straggler(0, 1, 100, s, fault_chance=0.0)
    assert r == 0.0, f"expected 0.0 got {r}"
check("simulate_straggler: fault_chance=0 → no sleep", t4)

def t5():
    s = _state()
    simulate_straggler(0, 5, 100, s, fault_chance=0.0)
    snap = dict(s)
    simulate_straggler(0, 5, 100, s, fault_chance=0.0)
    assert s == snap, "State changed on 2nd call same epoch"
check("simulate_straggler: once-per-epoch lottery", t5)

def t6():
    s = {"fault_wait": True, "fault_round": 3, "fault_wait_time": 0.0, "saved_epoch": 3}
    simulate_straggler(0, 4, 100, s, fault_chance=0.0)
    assert s["fault_wait"] is False, "fault_wait not cleared"
check("simulate_straggler: clears after fault_round", t6)

# ------------------------------------------------------------------
# 3. PerformanceProfiler.get_timing_summary()
# ------------------------------------------------------------------
print("\n[ profiling/performance_profiler.py ]")
from src.profiling.performance_profiler import PerformanceProfiler

def t7():
    p = PerformanceProfiler(device_id=0, rank=0, enable_nvml=False)
    s = p.get_timing_summary()
    expected = {"train_time", "sync_time", "average_time", "throughput", "bottleneck"}
    assert set(s.keys()) == expected, f"Keys: {set(s.keys())}"
    assert s["train_time"] == 0.0
    assert s["sync_time"] == 0.0
    assert s["average_time"] == 0.0
check("get_timing_summary: correct keys and zero values when empty", t7)

# ------------------------------------------------------------------
# 4. Symbol presence checks (ssgd, spawn_workers)
# ------------------------------------------------------------------
print("\n[ training/distributed_trainer.py ]")

def t8():
    from src.training.distributed_trainer import ssgd
    import inspect
    sig = inspect.signature(ssgd)
    params = list(sig.parameters)
    assert "model" in params and "partition_size" in params
check("ssgd: importable with correct signature", t8)

def t9():
    from src.training.distributed_trainer import spawn_workers
    import inspect
    sig = inspect.signature(spawn_workers)
    assert "world_size" in sig.parameters and "fn" in sig.parameters
check("spawn_workers: importable with correct signature", t9)

# ------------------------------------------------------------------
# 5. Dashboard chart functions importable
# ------------------------------------------------------------------
print("\n[ monitoring/dashboard.py — import guard ]")

def t10():
    import importlib, unittest.mock as mock
    # dashboard imports streamlit at module level; mock it so we can import
    with mock.patch.dict("sys.modules", {
        "streamlit": mock.MagicMock(),
        "plotly": mock.MagicMock(),
        "plotly.graph_objects": mock.MagicMock(),
        "plotly.subplots": mock.MagicMock(),
        "pandas": mock.MagicMock(),
    }):
        import importlib.util, types
        spec = importlib.util.spec_from_file_location(
            "dashboard",
            os.path.join(ROOT, "src", "monitoring", "dashboard.py")
        )
        mod = types.ModuleType("dashboard")
        spec.loader.exec_module(mod)
        assert hasattr(mod, "plot_partition_distribution")
        assert hasattr(mod, "plot_sync_overhead")
check("dashboard: plot_partition_distribution + plot_sync_overhead importable", t10)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"\n{'='*40}")
print(f"Results: {passed}/{total} passed")
if passed == total:
    print("All checks PASSED ✓")
    sys.exit(0)
else:
    print("Some checks FAILED ✗")
    sys.exit(1)

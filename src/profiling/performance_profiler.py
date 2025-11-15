"""
Performance Profiling Engine
Real-time monitoring of training performance metrics
"""

import torch
import pynvml
import time
import psutil
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
import threading
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single iteration"""
    timestamp: float
    rank: int
    epoch: int
    iteration: int

    # Training metrics
    batch_size: int
    loss: float
    iteration_time: float
    data_loading_time: float
    forward_time: float
    backward_time: float
    optimizer_time: float
    throughput: float  # samples/sec

    # GPU metrics
    gpu_utilization: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_memory_percent: float
    gpu_temperature: float
    gpu_power_draw: float

    # System metrics
    cpu_percent: float
    ram_percent: float


class PerformanceProfiler:
    """Real-time performance profiler for distributed training"""

    def __init__(
        self,
        device_id: int = 0,
        rank: int = 0,
        window_size: int = 100,
        enable_nvml: bool = True
    ):
        """
        Initialize performance profiler

        Args:
            device_id: GPU device ID
            rank: Distributed training rank
            window_size: Number of recent metrics to keep
            enable_nvml: Enable NVML for detailed GPU metrics
        """
        self.device_id = device_id
        self.rank = rank
        self.window_size = window_size

        # Metrics history
        self.metrics_history: deque = deque(maxlen=window_size)

        # Timing
        self.iteration_start = None
        self.data_loading_start = None
        self.forward_start = None
        self.backward_start = None
        self.optimizer_start = None

        # NVML setup
        self.nvml_enabled = False
        if enable_nvml:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self.nvml_enabled = True
                logger.info(f"NVML enabled for GPU {device_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")

        # Current metrics
        self.current_metrics = {
            'epoch': 0,
            'iteration': 0,
            'batch_size': 0,
            'loss': 0.0,
        }

        # Timing measurements
        self.timings = {
            'data_loading': 0.0,
            'forward': 0.0,
            'backward': 0.0,
            'optimizer': 0.0,
            'iteration': 0.0,
        }

    def __del__(self):
        """Cleanup NVML"""
        if self.nvml_enabled:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

    def start_iteration(self):
        """Mark start of iteration"""
        self.iteration_start = time.time()

    def start_data_loading(self):
        """Mark start of data loading"""
        self.data_loading_start = time.time()

    def end_data_loading(self):
        """Mark end of data loading"""
        if self.data_loading_start:
            self.timings['data_loading'] = time.time() - self.data_loading_start

    def start_forward(self):
        """Mark start of forward pass"""
        self.forward_start = time.time()

    def end_forward(self):
        """Mark end of forward pass"""
        if self.forward_start:
            self.timings['forward'] = time.time() - self.forward_start

    def start_backward(self):
        """Mark start of backward pass"""
        self.backward_start = time.time()

    def end_backward(self):
        """Mark end of backward pass"""
        if self.backward_start:
            self.timings['backward'] = time.time() - self.backward_start

    def start_optimizer(self):
        """Mark start of optimizer step"""
        self.optimizer_start = time.time()

    def end_optimizer(self):
        """Mark end of optimizer step"""
        if self.optimizer_start:
            self.timings['optimizer'] = time.time() - self.optimizer_start

    def end_iteration(self, epoch: int, iteration: int, batch_size: int, loss: float):
        """
        Mark end of iteration and record metrics

        Args:
            epoch: Current epoch
            iteration: Current iteration
            batch_size: Batch size
            loss: Training loss
        """
        if self.iteration_start is None:
            return

        # Calculate iteration time
        iteration_time = time.time() - self.iteration_start
        self.timings['iteration'] = iteration_time

        # Calculate throughput
        throughput = batch_size / iteration_time if iteration_time > 0 else 0.0

        # Get GPU metrics
        gpu_metrics = self._get_gpu_metrics()

        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram_percent = psutil.virtual_memory().percent

        # Create metrics object
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            rank=self.rank,
            epoch=epoch,
            iteration=iteration,
            batch_size=batch_size,
            loss=loss,
            iteration_time=iteration_time,
            data_loading_time=self.timings.get('data_loading', 0.0),
            forward_time=self.timings.get('forward', 0.0),
            backward_time=self.timings.get('backward', 0.0),
            optimizer_time=self.timings.get('optimizer', 0.0),
            throughput=throughput,
            gpu_utilization=gpu_metrics['utilization'],
            gpu_memory_used_mb=gpu_metrics['memory_used_mb'],
            gpu_memory_total_mb=gpu_metrics['memory_total_mb'],
            gpu_memory_percent=gpu_metrics['memory_percent'],
            gpu_temperature=gpu_metrics['temperature'],
            gpu_power_draw=gpu_metrics['power_draw'],
            cpu_percent=cpu_percent,
            ram_percent=ram_percent,
        )

        # Store metrics
        self.metrics_history.append(metrics)

        # Reset timings
        self.iteration_start = None
        self.timings = {k: 0.0 for k in self.timings}

        return metrics

    def _get_gpu_metrics(self) -> Dict:
        """Get current GPU metrics"""
        if self.nvml_enabled:
            try:
                # Utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_util = utilization.gpu

                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                memory_used_mb = mem_info.used / (1024 ** 2)
                memory_total_mb = mem_info.total / (1024 ** 2)
                memory_percent = (mem_info.used / mem_info.total) * 100

                # Temperature
                temperature = pynvml.nvmlDeviceGetTemperature(
                    self.gpu_handle,
                    pynvml.NVML_TEMPERATURE_GPU
                )

                # Power
                try:
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # mW to W
                except:
                    power_draw = 0.0

                return {
                    'utilization': gpu_util,
                    'memory_used_mb': memory_used_mb,
                    'memory_total_mb': memory_total_mb,
                    'memory_percent': memory_percent,
                    'temperature': temperature,
                    'power_draw': power_draw,
                }

            except Exception as e:
                logger.warning(f"Failed to get NVML metrics: {e}")

        # Fallback to PyTorch
        try:
            memory_allocated = torch.cuda.memory_allocated(self.device_id) / (1024 ** 2)
            memory_reserved = torch.cuda.memory_reserved(self.device_id) / (1024 ** 2)
            memory_total = torch.cuda.get_device_properties(self.device_id).total_memory / (1024 ** 2)

            return {
                'utilization': 0.0,
                'memory_used_mb': memory_allocated,
                'memory_total_mb': memory_total,
                'memory_percent': (memory_allocated / memory_total) * 100 if memory_total > 0 else 0.0,
                'temperature': 0.0,
                'power_draw': 0.0,
            }
        except:
            return {
                'utilization': 0.0,
                'memory_used_mb': 0.0,
                'memory_total_mb': 0.0,
                'memory_percent': 0.0,
                'temperature': 0.0,
                'power_draw': 0.0,
            }

    def get_recent_metrics(self, n: Optional[int] = None) -> List[PerformanceMetrics]:
        """Get recent metrics"""
        if n is None:
            return list(self.metrics_history)
        return list(self.metrics_history)[-n:]

    def get_average_metrics(self, n: Optional[int] = None) -> Dict:
        """Calculate average metrics over recent iterations"""
        recent = self.get_recent_metrics(n)
        if not recent:
            return {}

        avg_metrics = {
            'iteration_time': sum(m.iteration_time for m in recent) / len(recent),
            'throughput': sum(m.throughput for m in recent) / len(recent),
            'gpu_utilization': sum(m.gpu_utilization for m in recent) / len(recent),
            'gpu_memory_percent': sum(m.gpu_memory_percent for m in recent) / len(recent),
            'data_loading_time': sum(m.data_loading_time for m in recent) / len(recent),
            'forward_time': sum(m.forward_time for m in recent) / len(recent),
            'backward_time': sum(m.backward_time for m in recent) / len(recent),
            'optimizer_time': sum(m.optimizer_time for m in recent) / len(recent),
        }

        return avg_metrics

    def identify_bottleneck(self) -> str:
        """Identify performance bottleneck"""
        avg = self.get_average_metrics(n=20)
        if not avg:
            return "No data"

        # Find which operation takes most time
        operations = {
            'Data Loading': avg.get('data_loading_time', 0),
            'Forward Pass': avg.get('forward_time', 0),
            'Backward Pass': avg.get('backward_time', 0),
            'Optimizer': avg.get('optimizer_time', 0),
        }

        bottleneck = max(operations.items(), key=lambda x: x[1])
        return f"{bottleneck[0]} ({bottleneck[1]:.3f}s)"

    def save_metrics(self, filepath: str):
        """Save metrics to JSON file"""
        data = [asdict(m) for m in self.metrics_history]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} metrics to {filepath}")

    def load_metrics(self, filepath: str):
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.metrics_history = deque(
            [PerformanceMetrics(**m) for m in data],
            maxlen=self.window_size
        )
        logger.info(f"Loaded {len(data)} metrics from {filepath}")

    def print_summary(self):
        """Print performance summary"""
        avg = self.get_average_metrics()
        if not avg:
            print("No metrics available")
            return

        print("\n" + "="*80)
        print(f"PERFORMANCE SUMMARY (Rank {self.rank})")
        print("="*80)
        print(f"Average Iteration Time: {avg['iteration_time']:.3f}s")
        print(f"Average Throughput: {avg['throughput']:.2f} samples/s")
        print(f"Average GPU Utilization: {avg['gpu_utilization']:.1f}%")
        print(f"Average GPU Memory: {avg['gpu_memory_percent']:.1f}%")
        print(f"\nTime Breakdown:")
        print(f"  Data Loading: {avg['data_loading_time']:.3f}s ({avg['data_loading_time']/avg['iteration_time']*100:.1f}%)")
        print(f"  Forward Pass: {avg['forward_time']:.3f}s ({avg['forward_time']/avg['iteration_time']*100:.1f}%)")
        print(f"  Backward Pass: {avg['backward_time']:.3f}s ({avg['backward_time']/avg['iteration_time']*100:.1f}%)")
        print(f"  Optimizer: {avg['optimizer_time']:.3f}s ({avg['optimizer_time']/avg['iteration_time']*100:.1f}%)")
        print(f"\nBottleneck: {self.identify_bottleneck()}")
        print("="*80 + "\n")

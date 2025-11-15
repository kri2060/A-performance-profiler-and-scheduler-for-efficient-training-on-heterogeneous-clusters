"""
GPU Profiling Module
Detects and benchmarks GPU hardware characteristics
"""

import torch
import pynvml
import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GPUProfile:
    """GPU hardware profile"""
    device_id: int
    name: str
    compute_capability: str
    total_memory_mb: float
    memory_bandwidth_gbps: float
    cuda_cores: Optional[int]
    sm_count: int
    clock_rate_mhz: float
    memory_clock_rate_mhz: float
    pcie_link_gen: int
    pcie_link_width: int
    compute_score: float  # Relative performance score


class GPUProfiler:
    """Hardware profiling for GPUs"""

    def __init__(self):
        """Initialize NVML"""
        try:
            pynvml.nvmlInit()
            self.initialized = True
            logger.info("NVML initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self.initialized = False

    def __del__(self):
        """Cleanup NVML"""
        if self.initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

    def get_gpu_count(self) -> int:
        """Get number of available GPUs"""
        if not self.initialized:
            return torch.cuda.device_count()
        try:
            return pynvml.nvmlDeviceGetCount()
        except:
            return torch.cuda.device_count()

    def get_gpu_info(self, device_id: int) -> Dict:
        """Get detailed GPU information"""
        if not self.initialized:
            return self._get_torch_gpu_info(device_id)

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            # Basic info
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory_mb = mem_info.total / (1024 ** 2)

            # Compute capability
            major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
            minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
            compute_capability = f"{major}.{minor}"

            # Clock rates
            clock_rate_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            memory_clock_rate_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)

            # SM count
            sm_count = pynvml.nvmlDeviceGetNumGpuCores(handle) if hasattr(pynvml, 'nvmlDeviceGetNumGpuCores') else 0

            # PCIe info
            try:
                pcie_link_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
                pcie_link_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
            except:
                pcie_link_gen = 3
                pcie_link_width = 16

            # Estimate memory bandwidth (GB/s)
            memory_bandwidth_gbps = (memory_clock_rate_mhz * 1e6 * 2 * (pcie_link_width / 8)) / 1e9

            return {
                'device_id': device_id,
                'name': name,
                'compute_capability': compute_capability,
                'total_memory_mb': total_memory_mb,
                'memory_bandwidth_gbps': memory_bandwidth_gbps,
                'sm_count': sm_count,
                'clock_rate_mhz': clock_rate_mhz,
                'memory_clock_rate_mhz': memory_clock_rate_mhz,
                'pcie_link_gen': pcie_link_gen,
                'pcie_link_width': pcie_link_width,
            }
        except Exception as e:
            logger.warning(f"NVML failed for GPU {device_id}, falling back to PyTorch: {e}")
            return self._get_torch_gpu_info(device_id)

    def _get_torch_gpu_info(self, device_id: int) -> Dict:
        """Fallback GPU info using PyTorch"""
        props = torch.cuda.get_device_properties(device_id)
        return {
            'device_id': device_id,
            'name': props.name,
            'compute_capability': f"{props.major}.{props.minor}",
            'total_memory_mb': props.total_memory / (1024 ** 2),
            'memory_bandwidth_gbps': 100.0,  # Default estimate
            'sm_count': props.multi_processor_count,
            'clock_rate_mhz': props.clock_rate / 1000,
            'memory_clock_rate_mhz': 0,
            'pcie_link_gen': 3,
            'pcie_link_width': 16,
        }

    def estimate_cuda_cores(self, sm_count: int, compute_capability: str) -> Optional[int]:
        """Estimate CUDA cores based on SM count and architecture"""
        major = int(compute_capability.split('.')[0])

        # CUDA cores per SM for different architectures
        cores_per_sm = {
            3: 192,  # Kepler
            5: 128,  # Maxwell
            6: 64 if int(compute_capability.split('.')[1]) == 0 else 128,  # Pascal
            7: 64,   # Volta/Turing
            8: 64 if int(compute_capability.split('.')[1]) >= 6 else 128,  # Ampere
            9: 128,  # Hopper
        }

        multiplier = cores_per_sm.get(major, 64)
        return sm_count * multiplier if sm_count > 0 else None

    def benchmark_gpu_compute(self, device_id: int, matrix_size: int = 4096, iterations: int = 100) -> float:
        """Benchmark GPU compute performance (TFLOPS)"""
        device = torch.device(f'cuda:{device_id}')

        # Warm-up
        a = torch.randn(matrix_size, matrix_size, device=device)
        b = torch.randn(matrix_size, matrix_size, device=device)
        torch.matmul(a, b)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(iterations):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Calculate TFLOPS
        flops = 2 * matrix_size ** 3 * iterations  # Matrix multiplication FLOPs
        tflops = flops / elapsed / 1e12

        logger.info(f"GPU {device_id} benchmark: {tflops:.2f} TFLOPS")
        return tflops

    def benchmark_memory_bandwidth(self, device_id: int, size_mb: int = 1024, iterations: int = 50) -> float:
        """Benchmark GPU memory bandwidth (GB/s)"""
        device = torch.device(f'cuda:{device_id}')
        size_bytes = size_mb * 1024 * 1024
        elements = size_bytes // 4  # float32

        # Allocate memory
        data = torch.randn(elements, device=device)

        # Warm-up
        data.copy_(data)
        torch.cuda.synchronize()

        # Benchmark copy
        start = time.time()
        for _ in range(iterations):
            data.copy_(data)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        bandwidth_gbps = (size_bytes * iterations) / elapsed / 1e9
        logger.info(f"GPU {device_id} memory bandwidth: {bandwidth_gbps:.2f} GB/s")
        return bandwidth_gbps

    def calculate_compute_score(self, gpu_info: Dict, tflops: float) -> float:
        """Calculate relative compute score for GPU"""
        # Weighted score based on multiple factors
        memory_score = gpu_info['total_memory_mb'] / 1024  # Normalize to GB
        bandwidth_score = gpu_info['memory_bandwidth_gbps'] / 100  # Normalize
        clock_score = gpu_info['clock_rate_mhz'] / 1000  # Normalize to GHz
        compute_score = tflops

        # Weighted combination
        score = (
            0.4 * compute_score +
            0.3 * memory_score +
            0.2 * bandwidth_score +
            0.1 * clock_score
        )

        return round(score, 2)

    def profile_all_gpus(self, benchmark: bool = True) -> List[GPUProfile]:
        """Profile all available GPUs"""
        gpu_count = self.get_gpu_count()
        logger.info(f"Found {gpu_count} GPU(s)")

        profiles = []
        for device_id in range(gpu_count):
            logger.info(f"Profiling GPU {device_id}...")

            # Get basic info
            gpu_info = self.get_gpu_info(device_id)

            # Run benchmarks if requested
            if benchmark and torch.cuda.is_available():
                try:
                    tflops = self.benchmark_gpu_compute(device_id, matrix_size=2048, iterations=50)
                    bandwidth = self.benchmark_memory_bandwidth(device_id, size_mb=512, iterations=20)
                    gpu_info['memory_bandwidth_gbps'] = bandwidth
                except Exception as e:
                    logger.warning(f"Benchmark failed for GPU {device_id}: {e}")
                    tflops = 1.0
            else:
                tflops = 1.0

            # Estimate CUDA cores
            cuda_cores = self.estimate_cuda_cores(
                gpu_info['sm_count'],
                gpu_info['compute_capability']
            )

            # Calculate compute score
            compute_score = self.calculate_compute_score(gpu_info, tflops)

            profile = GPUProfile(
                device_id=device_id,
                name=gpu_info['name'],
                compute_capability=gpu_info['compute_capability'],
                total_memory_mb=gpu_info['total_memory_mb'],
                memory_bandwidth_gbps=gpu_info['memory_bandwidth_gbps'],
                cuda_cores=cuda_cores,
                sm_count=gpu_info['sm_count'],
                clock_rate_mhz=gpu_info['clock_rate_mhz'],
                memory_clock_rate_mhz=gpu_info['memory_clock_rate_mhz'],
                pcie_link_gen=gpu_info['pcie_link_gen'],
                pcie_link_width=gpu_info['pcie_link_width'],
                compute_score=compute_score
            )

            profiles.append(profile)
            logger.info(f"GPU {device_id}: {profile.name} (Score: {compute_score})")

        return profiles

    def save_profiles(self, profiles: List[GPUProfile], output_path: str):
        """Save GPU profiles to JSON"""
        data = [asdict(profile) for profile in profiles]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved GPU profiles to {output_path}")

    def load_profiles(self, input_path: str) -> List[GPUProfile]:
        """Load GPU profiles from JSON"""
        with open(input_path, 'r') as f:
            data = json.load(f)
        profiles = [GPUProfile(**item) for item in data]
        logger.info(f"Loaded {len(profiles)} GPU profiles from {input_path}")
        return profiles


def main():
    """Main profiling function"""
    profiler = GPUProfiler()
    profiles = profiler.profile_all_gpus(benchmark=True)

    # Print summary
    print("\n" + "="*80)
    print("GPU PROFILING SUMMARY")
    print("="*80)
    for profile in profiles:
        print(f"\nGPU {profile.device_id}: {profile.name}")
        print(f"  Compute Capability: {profile.compute_capability}")
        print(f"  Memory: {profile.total_memory_mb:.0f} MB")
        print(f"  Bandwidth: {profile.memory_bandwidth_gbps:.2f} GB/s")
        print(f"  CUDA Cores: {profile.cuda_cores or 'N/A'}")
        print(f"  SM Count: {profile.sm_count}")
        print(f"  Clock Rate: {profile.clock_rate_mhz:.0f} MHz")
        print(f"  Compute Score: {profile.compute_score}")

    # Save to file
    profiler.save_profiles(profiles, "experiments/configs/gpu_profiles.json")


if __name__ == "__main__":
    main()

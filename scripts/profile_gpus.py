#!/usr/bin/env python3
"""
Standalone GPU profiling script
Usage: python profile_gpus.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from profiling.gpu_profiler import GPUProfiler
from profiling.system_profiler import SystemProfiler
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Profile hardware for heterogeneous cluster")
    parser.add_argument("--output-dir", type=str, default="experiments/configs",
                       help="Output directory for profiles")
    parser.add_argument("--no-benchmark", action="store_true",
                       help="Skip GPU benchmarking")

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Profile GPUs
    print("\n" + "="*80)
    print("PROFILING GPUs...")
    print("="*80)
    gpu_profiler = GPUProfiler()
    gpu_profiles = gpu_profiler.profile_all_gpus(benchmark=not args.no_benchmark)

    # Profile System
    print("\n" + "="*80)
    print("PROFILING SYSTEM...")
    print("="*80)
    system_profiler = SystemProfiler()
    system_profile = system_profiler.profile_system()

    # Save profiles
    gpu_path = os.path.join(output_dir, "gpu_profiles.json")
    system_path = os.path.join(output_dir, "system_profile.json")

    gpu_profiler.save_profiles(gpu_profiles, gpu_path)
    system_profiler.save_profile(system_profile, system_path)

    # Create combined profile
    combined = {
        'hostname': system_profile.hostname,
        'ip_address': system_profile.ip_address,
        'system': system_profile.__dict__,
        'gpus': [gpu.__dict__ for gpu in gpu_profiles]
    }

    combined_path = os.path.join(output_dir, f"node_profile_{system_profile.hostname}.json")
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2)

    print("\n" + "="*80)
    print(f"PROFILING COMPLETE")
    print(f"Profiles saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

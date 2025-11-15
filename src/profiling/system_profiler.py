"""
System Profiling Module
Profiles CPU, RAM, Network, and Storage
"""

import psutil
import platform
import socket
import time
import json
from typing import Dict, List
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SystemProfile:
    """System hardware profile"""
    hostname: str
    platform: str
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_frequency_mhz: float
    ram_total_gb: float
    ram_available_gb: float
    network_interfaces: List[str]
    ip_address: str


class SystemProfiler:
    """System hardware profiler"""

    def __init__(self):
        self.hostname = socket.gethostname()

    def get_cpu_info(self) -> Dict:
        """Get CPU information"""
        cpu_freq = psutil.cpu_freq()

        return {
            'cpu_model': platform.processor() or 'Unknown',
            'cpu_cores_physical': psutil.cpu_count(logical=False),
            'cpu_cores_logical': psutil.cpu_count(logical=True),
            'cpu_frequency_mhz': cpu_freq.max if cpu_freq else 0,
        }

    def get_memory_info(self) -> Dict:
        """Get RAM information"""
        mem = psutil.virtual_memory()

        return {
            'ram_total_gb': mem.total / (1024 ** 3),
            'ram_available_gb': mem.available / (1024 ** 3),
            'ram_used_gb': mem.used / (1024 ** 3),
            'ram_percent': mem.percent,
        }

    def get_network_info(self) -> Dict:
        """Get network information"""
        # Get network interfaces
        interfaces = []
        addrs = psutil.net_if_addrs()
        for interface_name in addrs:
            interfaces.append(interface_name)

        # Get primary IP address
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            s.close()
        except:
            ip_address = "127.0.0.1"

        return {
            'network_interfaces': interfaces,
            'ip_address': ip_address,
        }

    def benchmark_network_bandwidth(self, target_host: str = None, port: int = 5001,
                                   size_mb: int = 100) -> float:
        """
        Benchmark network bandwidth (MB/s)
        Note: Requires a receiver on target_host or use iperf
        """
        if target_host is None:
            logger.warning("No target host specified for network benchmark")
            return 0.0

        try:
            # Simple bandwidth test using socket
            data = b'x' * (1024 * 1024)  # 1 MB chunk
            iterations = size_mb

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((target_host, port))

            start = time.time()
            for _ in range(iterations):
                sock.sendall(data)
            elapsed = time.time() - start

            sock.close()

            bandwidth_mbps = (size_mb) / elapsed
            logger.info(f"Network bandwidth to {target_host}: {bandwidth_mbps:.2f} MB/s")
            return bandwidth_mbps

        except Exception as e:
            logger.warning(f"Network benchmark failed: {e}")
            return 0.0

    def benchmark_disk_io(self, test_file: str = "/tmp/io_test", size_mb: int = 100) -> Dict:
        """Benchmark disk I/O (MB/s)"""
        try:
            data = b'x' * (1024 * 1024)  # 1 MB

            # Write benchmark
            start = time.time()
            with open(test_file, 'wb') as f:
                for _ in range(size_mb):
                    f.write(data)
            write_time = time.time() - start
            write_speed = size_mb / write_time

            # Read benchmark
            start = time.time()
            with open(test_file, 'rb') as f:
                while f.read(1024 * 1024):
                    pass
            read_time = time.time() - start
            read_speed = size_mb / read_time

            # Cleanup
            import os
            os.remove(test_file)

            logger.info(f"Disk I/O - Write: {write_speed:.2f} MB/s, Read: {read_speed:.2f} MB/s")

            return {
                'write_speed_mbps': write_speed,
                'read_speed_mbps': read_speed,
            }

        except Exception as e:
            logger.warning(f"Disk I/O benchmark failed: {e}")
            return {'write_speed_mbps': 0.0, 'read_speed_mbps': 0.0}

    def profile_system(self) -> SystemProfile:
        """Profile entire system"""
        logger.info(f"Profiling system: {self.hostname}")

        cpu_info = self.get_cpu_info()
        mem_info = self.get_memory_info()
        net_info = self.get_network_info()

        profile = SystemProfile(
            hostname=self.hostname,
            platform=platform.system(),
            cpu_model=cpu_info['cpu_model'],
            cpu_cores_physical=cpu_info['cpu_cores_physical'],
            cpu_cores_logical=cpu_info['cpu_cores_logical'],
            cpu_frequency_mhz=cpu_info['cpu_frequency_mhz'],
            ram_total_gb=mem_info['ram_total_gb'],
            ram_available_gb=mem_info['ram_available_gb'],
            network_interfaces=net_info['network_interfaces'],
            ip_address=net_info['ip_address'],
        )

        return profile

    def save_profile(self, profile: SystemProfile, output_path: str):
        """Save system profile to JSON"""
        with open(output_path, 'w') as f:
            json.dump(asdict(profile), f, indent=2)
        logger.info(f"Saved system profile to {output_path}")

    def load_profile(self, input_path: str) -> SystemProfile:
        """Load system profile from JSON"""
        with open(input_path, 'r') as f:
            data = json.load(f)
        profile = SystemProfile(**data)
        logger.info(f"Loaded system profile from {input_path}")
        return profile


def main():
    """Main profiling function"""
    profiler = SystemProfiler()
    profile = profiler.profile_system()

    # Run benchmarks
    disk_io = profiler.benchmark_disk_io()

    # Print summary
    print("\n" + "="*80)
    print("SYSTEM PROFILING SUMMARY")
    print("="*80)
    print(f"\nHostname: {profile.hostname}")
    print(f"Platform: {profile.platform}")
    print(f"IP Address: {profile.ip_address}")
    print(f"\nCPU: {profile.cpu_model}")
    print(f"  Physical Cores: {profile.cpu_cores_physical}")
    print(f"  Logical Cores: {profile.cpu_cores_logical}")
    print(f"  Frequency: {profile.cpu_frequency_mhz:.0f} MHz")
    print(f"\nRAM:")
    print(f"  Total: {profile.ram_total_gb:.2f} GB")
    print(f"  Available: {profile.ram_available_gb:.2f} GB")
    print(f"\nNetwork Interfaces: {', '.join(profile.network_interfaces)}")
    print(f"\nDisk I/O:")
    print(f"  Write: {disk_io['write_speed_mbps']:.2f} MB/s")
    print(f"  Read: {disk_io['read_speed_mbps']:.2f} MB/s")

    # Save to file
    profiler.save_profile(profile, "experiments/configs/system_profile.json")


if __name__ == "__main__":
    main()

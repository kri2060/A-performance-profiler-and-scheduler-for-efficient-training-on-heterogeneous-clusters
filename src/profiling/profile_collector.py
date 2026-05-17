#!/usr/bin/env python3
"""
Distributed GPU Profile Collector
Master collects profiles from all workers before training
"""
import socket
import json
import time
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProfileCollector:
    """Collects GPU profiles from distributed workers"""

    def __init__(self, port: int = 29501):
        self.port = port
        self.profiles = []

    def start_server(self, world_size: int, timeout: int = 60):
        """
        Start server to collect profiles from workers

        Args:
            world_size: Total number of nodes (master + workers)
            timeout: Timeout in seconds
        """
        expected_workers = world_size - 1  # Exclude master
        logger.info(f"Starting profile collector on port {self.port}")
        logger.info(f"Waiting for {expected_workers} worker(s) to send profiles...")

        server_socket = socket.socket(socket.AF_INET, socket.AF_INET6)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('', self.port))
        server_socket.listen(expected_workers)
        server_socket.settimeout(timeout)

        profiles_received = 0
        start_time = time.time()

        try:
            while profiles_received < expected_workers:
                elapsed = time.time() - start_time
                remaining = timeout - elapsed

                if remaining <= 0:
                    logger.warning(f"Timeout! Only received {profiles_received}/{expected_workers} profiles")
                    break

                logger.info(f"Waiting for profiles... ({profiles_received}/{expected_workers}) - {remaining:.0f}s remaining")

                try:
                    client_socket, addr = server_socket.accept()
                    logger.info(f"Connection from {addr}")

                    # Receive profile data
                    data = b""
                    while True:
                        chunk = client_socket.recv(4096)
                        if not chunk:
                            break
                        data += chunk

                    if data:
                        profile = json.loads(data.decode('utf-8'))
                        self.profiles.append(profile)
                        profiles_received += 1
                        logger.info(f"✓ Received profile from {profile.get('hostname', 'unknown')} "
                                  f"({profiles_received}/{expected_workers})")

                    client_socket.close()

                except socket.timeout:
                    continue

        except Exception as e:
            logger.error(f"Error in profile collector: {e}")
        finally:
            server_socket.close()

        logger.info(f"Profile collection complete: {profiles_received}/{expected_workers} received")
        return self.profiles

    def merge_profiles(self, master_profile: Dict, worker_profiles: List[Dict], output_path: str):
        """
        Merge master and worker profiles into single file

        Args:
            master_profile: Master node GPU profile
            worker_profiles: List of worker GPU profiles
            output_path: Path to save merged profiles
        """
        all_profiles = []

        # Add master GPUs with rank 0
        if 'gpus' in master_profile:
            for i, gpu in enumerate(master_profile['gpus']):
                gpu['rank'] = 0
                gpu['node'] = master_profile.get('hostname', 'master')
                all_profiles.append(gpu)

        # Add worker GPUs with their ranks
        for rank, worker in enumerate(worker_profiles, start=1):
            if 'gpus' in worker:
                for i, gpu in enumerate(worker['gpus']):
                    gpu['rank'] = rank
                    gpu['node'] = worker.get('hostname', f'worker-{rank}')
                    all_profiles.append(gpu)

        # Save merged profiles
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(all_profiles, f, indent=2)

        logger.info(f"✓ Merged {len(all_profiles)} GPU profiles from {len(worker_profiles) + 1} nodes")
        logger.info(f"✓ Saved to: {output_path}")

        return all_profiles


class ProfileSender:
    """Sends GPU profile to master"""

    def __init__(self, master_addr: str, port: int = 29501):
        self.master_addr = master_addr
        self.port = port

    def send_profile(self, profile_path: str, max_retries: int = 3):
        """
        Send GPU profile to master

        Args:
            profile_path: Path to GPU profile JSON
            max_retries: Maximum number of retry attempts
        """
        # Load profile
        with open(profile_path, 'r') as f:
            profile = json.load(f)

        logger.info(f"Sending profile to master at {self.master_addr}:{self.port}")

        for attempt in range(max_retries):
            try:
                # Connect to master
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(10)
                client_socket.connect((self.master_addr, self.port))

                # Send profile
                data = json.dumps(profile).encode('utf-8')
                client_socket.sendall(data)
                client_socket.close()

                logger.info(f"✓ Profile sent successfully")
                return True

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    logger.error(f"Failed to send profile after {max_retries} attempts")
                    return False

        return False


def collect_profiles_cli():
    """CLI for collecting profiles on master"""
    import argparse

    parser = argparse.ArgumentParser(description="Collect GPU profiles from workers")
    parser.add_argument('--world-size', type=int, required=True,
                       help='Total number of nodes (master + workers)')
    parser.add_argument('--port', type=int, default=29501,
                       help='Port for profile collection')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Timeout in seconds')
    parser.add_argument('--master-profile', type=str, required=True,
                       help='Path to master GPU profile JSON')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for merged profiles')

    args = parser.parse_args()

    # Load master profile
    with open(args.master_profile, 'r') as f:
        master_profile = json.load(f)

    # Collect worker profiles
    collector = ProfileCollector(port=args.port)
    worker_profiles = collector.start_server(args.world_size, args.timeout)

    # Merge profiles
    collector.merge_profiles(master_profile, worker_profiles, args.output)


def send_profile_cli():
    """CLI for sending profile from worker"""
    import argparse

    parser = argparse.ArgumentParser(description="Send GPU profile to master")
    parser.add_argument('--master-addr', type=str, required=True,
                       help='Master node IP address')
    parser.add_argument('--port', type=int, default=29501,
                       help='Port for profile collection')
    parser.add_argument('--profile', type=str, required=True,
                       help='Path to GPU profile JSON')

    args = parser.parse_args()

    sender = ProfileSender(args.master_addr, args.port)
    sender.send_profile(args.profile)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'collect':
        sys.argv.pop(1)
        collect_profiles_cli()
    elif len(sys.argv) > 1 and sys.argv[1] == 'send':
        sys.argv.pop(1)
        send_profile_cli()
    else:
        print("Usage:")
        print("  python profile_collector.py collect --world-size N --master-profile PATH --output PATH")
        print("  python profile_collector.py send --master-addr IP --profile PATH")

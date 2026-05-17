import time
import random
import logging
import argparse
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

# Import existing modules (assuming they are in PYTHONPATH or relative)
import sys
import os
from types import ModuleType

# Mock torch if not present
try:
    import torch
except ImportError:
    torch = ModuleType('torch')
    sys.modules['torch'] = torch
    
# Mock numpy if not present (simpler to mock)
try:
    import numpy as np
except ImportError:
    np = ModuleType('numpy')
    np.median = lambda x: sorted(x)[len(x)//2] if x else 0
    sys.modules['numpy'] = np

sys.path.append(os.path.join(os.getcwd(), 'src'))

from scheduling.load_balancer import AdaptiveLoadBalancer, NodeCapability

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MockNode:
    """Simulates a physical GPU node"""
    rank: int
    device_name: str
    compute_capability: float  # Relative speed (1.0 = baseline)
    memory_capacity_mb: int
    bandwidth_gbps: int
    network_mbps: int
    
    # State
    current_batch_size: int = 0
    utilization: float = 0.0
    memory_used: float = 0.0
    
    def simulate_step(self, base_step_time: float) -> float:
        """
        Simulate one training step
        Returns: Time taken for the step in seconds
        """
        if self.current_batch_size == 0:
            return 0.0
            
        # Time = (Base Time * Batch Size) / Compute Capability
        # Add some random noise for realism
        noise = random.uniform(0.95, 1.05)
        step_time = (base_step_time * self.current_batch_size) / self.compute_capability
        step_time *= noise
        
        # Simulate network delay if straggler
        if self.network_mbps < 500: # Slow network
             step_time += random.uniform(0.05, 0.1)
             
        self.utilization = min(100.0, 80.0 + random.uniform(-5, 10))
        self.memory_used = min(100.0, (self.current_batch_size * 100) / (self.memory_capacity_mb / 128)) # Approx
        
        return step_time

class ClusterSimulator:
    def __init__(self, num_nodes: int = 4, profile_type: str = "heterogeneous", policy: str = "dynamic"):
        self.nodes: List[MockNode] = []
        self.load_balancer = AdaptiveLoadBalancer(policy=policy, rebalance_interval=5)
        self.setup_cluster(num_nodes, profile_type)
        
    def setup_cluster(self, num_nodes: int, profile_type: str):
        """Create mock nodes based on profile"""
        logger.info(f"Setting up {num_nodes} nodes with profile: {profile_type}")
        
        profiles = []
        if profile_type == "heterogeneous":
            # Gaming PC Mix: High-end, Mid-range, Budget, Legacy
            types = [
                ("NVIDIA RTX 4090", 3.0, 24000, 1000, 2500), # 2.5GbE
                ("NVIDIA RTX 3080", 1.5, 10000, 760, 1000),  # 1GbE
                ("NVIDIA RTX 3060", 0.6, 12000, 360, 1000),  # 1GbE, High VRAM but slow compute
                ("NVIDIA GTX 1080 Ti", 0.4, 11000, 484, 1000) # Legacy Straggler
            ]
            
            for i in range(num_nodes):
                # Cycle through types
                name, cap, mem, bw, net = types[i % len(types)]
                node = MockNode(i, name, cap, mem, bw, net)
                self.nodes.append(node)
                
                profiles.append({
                    'device_id': i,
                    'compute_score': cap,
                    'total_memory_mb': mem,
                    'memory_bandwidth_gbps': bw,
                    'network_mbps': net,
                    'hostname': f"node-{i}"
                })
        else:
            # Homogeneous
            for i in range(num_nodes):
                node = MockNode(i, "Generic GPU", 1.0, 16000, 300, 10000)
                self.nodes.append(node)
                profiles.append({
                    'device_id': i,
                    'compute_score': 1.0,
                    'total_memory_mb': 16000,
                    'memory_bandwidth_gbps': 300,
                    'network_mbps': 10000
                })
                
        # Register with Load Balancer
        self.load_balancer.register_nodes(profiles)

    def run_simulation(self, total_epochs: int = 5, steps_per_epoch: int = 10, total_global_batch: int = 128):
        """Run the simulation loop"""
        logger.info(f"Starting simulation: {total_epochs} epochs, {steps_per_epoch} steps/epoch, Batch={total_global_batch}")
        
        history = []
        
        for epoch in range(total_epochs):
            logger.info(f"--- Epoch {epoch+1}/{total_epochs} ---")
            
            epoch_times = []
            
            for step in range(steps_per_epoch):
                # 1. Get Batch Sizes from Load Balancer
                if self.load_balancer.should_rebalance() or step == 0:
                    logger.info("Rebalancing triggered...")
                    batch_sizes = self.load_balancer.calculate_batch_sizes(total_global_batch)
                    
                    # Apply to nodes
                    for node in self.nodes:
                        node.current_batch_size = batch_sizes.get(node.rank, 1)
                
                # 2. Simulate Training Step (Parallel)
                step_times = []
                for node in self.nodes:
                    # Base time for 1 sample on baseline GPU = 0.01s
                    t = node.simulate_step(base_step_time=0.01)
                    step_times.append(t)
                    
                    # Update LB stats
                    self.load_balancer.update_node_stats(node.rank, {
                        'utilization': node.utilization,
                        'memory_percent': node.memory_used,
                        'iteration_time': t
                    })
                
                # Synchronous training waits for the slowest
                max_step_time = max(step_times)
                epoch_times.append(max_step_time)
                
                # Check for updates to communication config
                if step % 10 == 0:
                    comp, accum = self.load_balancer.calculate_communication_config()
                
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            total_time = sum(epoch_times)
            throughput = (total_global_batch * steps_per_epoch) / total_time
            
            logger.info(f"Epoch {epoch+1} Complete: Avg Step={avg_epoch_time:.3f}s, Throughput={throughput:.2f} samples/s")
            
            history.append({
                'epoch': epoch,
                'avg_step_time': avg_epoch_time,
                'throughput': throughput,
                'efficiency': self.load_balancer.get_scaling_efficiency()
            })
            
            self.load_balancer.print_status()

        return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--profile", type=str, default="heterogeneous", choices=["heterogeneous", "homogeneous"])
    args = parser.parse_args()
    
    sim = ClusterSimulator(num_nodes=args.nodes, profile_type=args.profile)
    sim.run_simulation()

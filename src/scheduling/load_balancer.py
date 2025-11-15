"""
Adaptive Load Balancer
Core innovation: Dynamic workload distribution for heterogeneous GPUs
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NodeCapability:
    """Node capability profile"""
    rank: int
    device_id: int
    compute_score: float
    memory_mb: float
    bandwidth_gbps: float
    current_utilization: float = 0.0
    current_memory_percent: float = 0.0
    avg_iteration_time: float = 0.0
    is_straggler: bool = False


class LoadBalancingPolicy:
    """Base class for load balancing policies"""

    def calculate_batch_sizes(
        self,
        nodes: List[NodeCapability],
        total_batch_size: int
    ) -> Dict[int, int]:
        """Calculate batch size for each node"""
        raise NotImplementedError


class ProportionalPolicy(LoadBalancingPolicy):
    """Proportional batch sizing based on compute capability"""

    def calculate_batch_sizes(
        self,
        nodes: List[NodeCapability],
        total_batch_size: int
    ) -> Dict[int, int]:
        """
        Distribute batches proportional to compute scores

        Args:
            nodes: List of node capabilities
            total_batch_size: Total batch size to distribute

        Returns:
            Dictionary mapping rank to batch size
        """
        # Calculate total compute capacity
        total_compute = sum(node.compute_score for node in nodes)

        if total_compute == 0:
            # Equal distribution fallback
            batch_per_node = total_batch_size // len(nodes)
            return {node.rank: batch_per_node for node in nodes}

        # Proportional distribution
        batch_sizes = {}
        allocated = 0

        for i, node in enumerate(nodes):
            if i == len(nodes) - 1:
                # Last node gets remainder
                batch_sizes[node.rank] = total_batch_size - allocated
            else:
                # Proportional allocation
                proportion = node.compute_score / total_compute
                batch_size = int(total_batch_size * proportion)
                batch_size = max(1, batch_size)  # At least 1
                batch_sizes[node.rank] = batch_size
                allocated += batch_size

        logger.info(f"Proportional batch sizes: {batch_sizes}")
        return batch_sizes


class DynamicPolicy(LoadBalancingPolicy):
    """Dynamic batch sizing based on real-time performance"""

    def __init__(self, alpha: float = 0.7):
        """
        Args:
            alpha: Weight for compute score vs. runtime performance (0-1)
                  Higher alpha = more weight on compute score
                  Lower alpha = more weight on actual runtime
        """
        self.alpha = alpha

    def calculate_batch_sizes(
        self,
        nodes: List[NodeCapability],
        total_batch_size: int
    ) -> Dict[int, int]:
        """
        Dynamic distribution based on compute score and runtime performance

        Args:
            nodes: List of node capabilities with runtime stats
            total_batch_size: Total batch size to distribute

        Returns:
            Dictionary mapping rank to batch size
        """
        # Calculate performance scores
        performance_scores = []

        for node in nodes:
            # Combine compute score and runtime performance
            compute_factor = node.compute_score

            # Inverse of iteration time (faster = higher score)
            if node.avg_iteration_time > 0:
                runtime_factor = 1.0 / node.avg_iteration_time
            else:
                runtime_factor = 1.0

            # Normalize runtime factor
            runtime_factor = runtime_factor * compute_factor  # Scale to similar range

            # Penalize high utilization or memory usage
            utilization_penalty = 1.0 - (node.current_utilization / 100.0) * 0.2
            memory_penalty = 1.0 - (node.current_memory_percent / 100.0) * 0.2

            # Combined score
            score = (
                self.alpha * compute_factor +
                (1 - self.alpha) * runtime_factor
            ) * utilization_penalty * memory_penalty

            # Penalize stragglers heavily
            if node.is_straggler:
                score *= 0.5

            performance_scores.append(score)

        total_score = sum(performance_scores)

        if total_score == 0:
            batch_per_node = total_batch_size // len(nodes)
            return {node.rank: batch_per_node for node in nodes}

        # Allocate based on performance scores
        batch_sizes = {}
        allocated = 0

        for i, (node, score) in enumerate(zip(nodes, performance_scores)):
            if i == len(nodes) - 1:
                batch_sizes[node.rank] = total_batch_size - allocated
            else:
                proportion = score / total_score
                batch_size = int(total_batch_size * proportion)
                batch_size = max(1, batch_size)
                batch_sizes[node.rank] = batch_size
                allocated += batch_size

        logger.info(f"Dynamic batch sizes: {batch_sizes}")
        return batch_sizes


class AdaptiveLoadBalancer:
    """
    Adaptive load balancer for heterogeneous distributed training
    """

    def __init__(
        self,
        policy: str = "dynamic",
        straggler_threshold: float = 1.5,
        rebalance_interval: int = 10,
        min_batch_size: int = 1,
    ):
        """
        Initialize adaptive load balancer

        Args:
            policy: Load balancing policy ('proportional', 'dynamic', 'hybrid')
            straggler_threshold: Multiplier to detect stragglers
            rebalance_interval: Iterations between rebalancing
            min_batch_size: Minimum batch size per node
        """
        self.straggler_threshold = straggler_threshold
        self.rebalance_interval = rebalance_interval
        self.min_batch_size = min_batch_size

        # Select policy
        if policy == "proportional":
            self.policy = ProportionalPolicy()
        elif policy == "dynamic":
            self.policy = DynamicPolicy(alpha=0.7)
        elif policy == "hybrid":
            self.policy = DynamicPolicy(alpha=0.5)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        logger.info(f"Initialized AdaptiveLoadBalancer with policy={policy}")

        # State
        self.nodes: List[NodeCapability] = []
        self.iteration_count = 0
        self.batch_sizes: Dict[int, int] = {}

    def register_nodes(self, gpu_profiles: List[Dict]):
        """
        Register nodes from GPU profiles

        Args:
            gpu_profiles: List of GPU profile dictionaries
        """
        self.nodes = []

        for profile in gpu_profiles:
            node = NodeCapability(
                rank=profile.get('device_id', 0),
                device_id=profile.get('device_id', 0),
                compute_score=profile.get('compute_score', 1.0),
                memory_mb=profile.get('total_memory_mb', 0),
                bandwidth_gbps=profile.get('memory_bandwidth_gbps', 0),
            )
            self.nodes.append(node)

        logger.info(f"Registered {len(self.nodes)} nodes")

        # Print node summary
        for node in self.nodes:
            logger.info(
                f"Node {node.rank}: "
                f"Score={node.compute_score:.2f}, "
                f"Memory={node.memory_mb:.0f}MB"
            )

    def update_node_stats(self, rank: int, stats: Dict):
        """
        Update node runtime statistics

        Args:
            rank: Node rank
            stats: Statistics dictionary with keys:
                  - utilization
                  - memory_percent
                  - iteration_time
        """
        for node in self.nodes:
            if node.rank == rank:
                node.current_utilization = stats.get('utilization', 0.0)
                node.current_memory_percent = stats.get('memory_percent', 0.0)
                node.avg_iteration_time = stats.get('iteration_time', 0.0)
                break

    def detect_stragglers(self):
        """Detect straggler nodes based on iteration time"""
        if not self.nodes:
            return

        # Get valid iteration times
        iteration_times = [
            node.avg_iteration_time
            for node in self.nodes
            if node.avg_iteration_time > 0
        ]

        if not iteration_times:
            return

        # Calculate median time
        median_time = np.median(iteration_times)

        # Mark stragglers
        for node in self.nodes:
            if node.avg_iteration_time > median_time * self.straggler_threshold:
                node.is_straggler = True
                logger.warning(
                    f"Node {node.rank} detected as straggler "
                    f"(time={node.avg_iteration_time:.3f}s, median={median_time:.3f}s)"
                )
            else:
                node.is_straggler = False

    def calculate_batch_sizes(self, total_batch_size: int) -> Dict[int, int]:
        """
        Calculate optimal batch size distribution

        Args:
            total_batch_size: Total batch size across all nodes

        Returns:
            Dictionary mapping rank to batch size
        """
        if not self.nodes:
            raise ValueError("No nodes registered")

        # Detect stragglers
        self.detect_stragglers()

        # Calculate using policy
        batch_sizes = self.policy.calculate_batch_sizes(self.nodes, total_batch_size)

        # Enforce minimum batch size
        for rank in batch_sizes:
            batch_sizes[rank] = max(self.min_batch_size, batch_sizes[rank])

        # Store current batch sizes
        self.batch_sizes = batch_sizes

        return batch_sizes

    def should_rebalance(self) -> bool:
        """Check if rebalancing should occur"""
        self.iteration_count += 1
        return self.iteration_count % self.rebalance_interval == 0

    def get_batch_size(self, rank: int) -> int:
        """Get current batch size for a rank"""
        return self.batch_sizes.get(rank, 32)

    def get_scaling_efficiency(self) -> float:
        """
        Calculate scaling efficiency
        Perfect scaling = 1.0, poor scaling < 0.5
        """
        if not self.nodes:
            return 0.0

        # Calculate ideal speedup (linear scaling)
        total_compute = sum(node.compute_score for node in self.nodes)

        # Calculate actual speedup (inverse of slowest node)
        slowest_time = max(
            node.avg_iteration_time for node in self.nodes
            if node.avg_iteration_time > 0
        ) if any(node.avg_iteration_time > 0 for node in self.nodes) else 1.0

        fastest_time = min(
            node.avg_iteration_time for node in self.nodes
            if node.avg_iteration_time > 0
        ) if any(node.avg_iteration_time > 0 for node in self.nodes) else 1.0

        if slowest_time == 0:
            return 0.0

        # Efficiency = actual speedup / ideal speedup
        actual_speedup = len(self.nodes) * (fastest_time / slowest_time)
        ideal_speedup = len(self.nodes)

        efficiency = actual_speedup / ideal_speedup
        return min(1.0, efficiency)

    def get_load_imbalance(self) -> float:
        """
        Calculate load imbalance metric
        0.0 = perfect balance, 1.0 = maximum imbalance
        """
        if not self.nodes or not any(node.avg_iteration_time > 0 for node in self.nodes):
            return 0.0

        times = [node.avg_iteration_time for node in self.nodes if node.avg_iteration_time > 0]

        if not times:
            return 0.0

        max_time = max(times)
        min_time = min(times)

        if max_time == 0:
            return 0.0

        imbalance = (max_time - min_time) / max_time
        return imbalance

    def print_status(self):
        """Print current load balancer status"""
        print("\n" + "="*80)
        print("LOAD BALANCER STATUS")
        print("="*80)

        for node in self.nodes:
            straggler_mark = " [STRAGGLER]" if node.is_straggler else ""
            print(
                f"Node {node.rank}{straggler_mark}: "
                f"Batch={self.batch_sizes.get(node.rank, 'N/A')}, "
                f"Time={node.avg_iteration_time:.3f}s, "
                f"GPU={node.current_utilization:.1f}%, "
                f"Mem={node.current_memory_percent:.1f}%"
            )

        print(f"\nScaling Efficiency: {self.get_scaling_efficiency():.2%}")
        print(f"Load Imbalance: {self.get_load_imbalance():.2%}")
        print("="*80 + "\n")

    def save_state(self, filepath: str):
        """Save load balancer state"""
        state = {
            'nodes': [node.__dict__ for node in self.nodes],
            'batch_sizes': self.batch_sizes,
            'iteration_count': self.iteration_count,
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved load balancer state to {filepath}")

    def load_state(self, filepath: str):
        """Load load balancer state"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.nodes = [NodeCapability(**n) for n in state['nodes']]
        self.batch_sizes = {int(k): v for k, v in state['batch_sizes'].items()}
        self.iteration_count = state['iteration_count']

        logger.info(f"Loaded load balancer state from {filepath}")


def test_load_balancer():
    """Test load balancer with sample GPU profiles"""

    # Simulated heterogeneous cluster
    gpu_profiles = [
        {  # Fast GPU
            'device_id': 0,
            'compute_score': 10.0,
            'total_memory_mb': 8192,
            'memory_bandwidth_gbps': 320,
        },
        {  # Medium GPU
            'device_id': 1,
            'compute_score': 6.0,
            'total_memory_mb': 6144,
            'memory_bandwidth_gbps': 192,
        },
        {  # Slow GPU
            'device_id': 2,
            'compute_score': 3.0,
            'total_memory_mb': 4096,
            'memory_bandwidth_gbps': 128,
        },
    ]

    # Test proportional policy
    print("\nTesting PROPORTIONAL Policy:")
    lb_prop = AdaptiveLoadBalancer(policy="proportional")
    lb_prop.register_nodes(gpu_profiles)
    batch_sizes = lb_prop.calculate_batch_sizes(total_batch_size=128)
    print(f"Batch sizes: {batch_sizes}")

    # Simulate runtime stats
    lb_prop.update_node_stats(0, {'utilization': 80, 'memory_percent': 70, 'iteration_time': 0.1})
    lb_prop.update_node_stats(1, {'utilization': 75, 'memory_percent': 65, 'iteration_time': 0.15})
    lb_prop.update_node_stats(2, {'utilization': 90, 'memory_percent': 85, 'iteration_time': 0.25})

    # Test dynamic policy
    print("\nTesting DYNAMIC Policy:")
    lb_dyn = AdaptiveLoadBalancer(policy="dynamic")
    lb_dyn.register_nodes(gpu_profiles)
    lb_dyn.update_node_stats(0, {'utilization': 80, 'memory_percent': 70, 'iteration_time': 0.1})
    lb_dyn.update_node_stats(1, {'utilization': 75, 'memory_percent': 65, 'iteration_time': 0.15})
    lb_dyn.update_node_stats(2, {'utilization': 90, 'memory_percent': 85, 'iteration_time': 0.25})
    batch_sizes = lb_dyn.calculate_batch_sizes(total_batch_size=128)
    print(f"Batch sizes: {batch_sizes}")

    lb_dyn.print_status()


if __name__ == "__main__":
    test_load_balancer()

"""
Main Training Script
Orchestrates distributed training with load balancing
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path
import json
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.distributed_trainer import DistributedTrainer, setup_distributed_training
from training.models import get_model
from utils.datasets import get_dataset
from profiling.performance_profiler import PerformanceProfiler
from scheduling.load_balancer import AdaptiveLoadBalancer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Heterogeneous distributed training")

    # Model args
    parser.add_argument('--model', type=str, default='simple_cnn',
                       choices=['simple_cnn', 'resnet50', 'bert', 'gpt2'],
                       help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Number of classes')

    # Dataset args
    parser.add_argument('--dataset', type=str, default='synthetic_image',
                       choices=['synthetic_image', 'synthetic_text', 'cifar10', 'cifar100'],
                       help='Dataset')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of samples (for synthetic datasets)')
    parser.add_argument('--image-size', type=int, default=32,
                       help='Image size')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory')

    # Training args
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Base batch size per GPU')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')

    # Distributed args
    parser.add_argument('--backend', type=str, default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Distributed backend')
    parser.add_argument('--master-addr', type=str, default='localhost',
                       help='Master node address')
    parser.add_argument('--master-port', type=str, default='29500',
                       help='Master node port')
    parser.add_argument('--rank', type=int, default=None,
                       help='Rank (auto-detected if None)')
    parser.add_argument('--world-size', type=int, default=None,
                       help='World size (auto-detected if None)')

    # Load balancing args
    parser.add_argument('--enable-load-balancing', action='store_true',
                       help='Enable adaptive load balancing')
    parser.add_argument('--load-balance-policy', type=str, default='dynamic',
                       choices=['proportional', 'dynamic', 'hybrid'],
                       help='Load balancing policy')
    parser.add_argument('--rebalance-interval', type=int, default=10,
                       help='Rebalance interval (iterations)')

    # Profiling args
    parser.add_argument('--enable-profiling', action='store_true',
                       help='Enable performance profiling')
    parser.add_argument('--profile-interval', type=int, default=1,
                       help='Profiling interval (iterations)')

    # Output args
    parser.add_argument('--output-dir', type=str, default='experiments',
                       help='Output directory')
    parser.add_argument('--experiment-name', type=str, default='default',
                       help='Experiment name')

    # GPU profiling
    parser.add_argument('--gpu-profiles', type=str, default=None,
                       help='Path to GPU profiles JSON')

    args = parser.parse_args()
    return args


def load_gpu_profiles(gpu_profiles_path: str):
    """Load GPU profiles from file"""
    if not os.path.exists(gpu_profiles_path):
        logger.warning(f"GPU profiles not found: {gpu_profiles_path}")
        return []

    with open(gpu_profiles_path, 'r') as f:
        profiles = json.load(f)

    logger.info(f"Loaded {len(profiles)} GPU profiles")
    return profiles


def train_worker(rank, args):
    """Training worker function"""

    # Setup distributed
    if args.rank is None:
        rank = int(os.environ.get('RANK', rank))
    else:
        rank = args.rank

    if args.world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    else:
        world_size = args.world_size

    logger.info(f"Starting worker: rank={rank}, world_size={world_size}")

    # Create output directories
    output_dir = Path(args.output_dir) / args.experiment_name
    logs_dir = output_dir / "logs"
    configs_dir = output_dir / "configs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Load GPU profiles for load balancing
    load_balancer = None
    batch_size_multiplier = 1.0

    if args.enable_load_balancing:
        gpu_profiles_path = args.gpu_profiles or (configs_dir / "gpu_profiles.json")

        if os.path.exists(gpu_profiles_path):
            gpu_profiles = load_gpu_profiles(str(gpu_profiles_path))

            if gpu_profiles:
                load_balancer = AdaptiveLoadBalancer(
                    policy=args.load_balance_policy,
                    rebalance_interval=args.rebalance_interval
                )
                load_balancer.register_nodes(gpu_profiles)

                # Calculate initial batch sizes
                batch_sizes = load_balancer.calculate_batch_sizes(
                    total_batch_size=args.batch_size * world_size
                )

                # Get batch size for this rank
                my_batch_size = batch_sizes.get(rank, args.batch_size)
                batch_size_multiplier = my_batch_size / args.batch_size

                logger.info(f"Rank {rank}: Adjusted batch size = {my_batch_size}")
        else:
            logger.warning(f"GPU profiles not found, using equal batch sizes")

    # Create model
    model = get_model(args.model, num_classes=args.num_classes, pretrained=False)

    # Create dataset
    train_dataset = get_dataset(
        args.dataset,
        num_samples=args.num_samples,
        image_size=args.image_size,
        num_classes=args.num_classes,
        data_dir=args.data_dir
    )

    # Create profiler
    profiler = None
    if args.enable_profiling:
        profiler = PerformanceProfiler(
            device_id=rank,
            rank=rank,
            window_size=100
        )

    # Create trainer
    trainer = DistributedTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=None,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        backend=args.backend,
        master_addr=args.master_addr,
        master_port=args.master_port,
        device_id=rank,
        heterogeneous_batch=args.enable_load_balancing,
        batch_size_multiplier=batch_size_multiplier
    )

    # Training loop with profiling
    metrics_history = []

    for epoch in range(args.epochs):
        trainer.current_epoch = epoch

        if hasattr(trainer.train_loader.sampler, 'set_epoch'):
            trainer.train_loader.sampler.set_epoch(epoch)

        # Train epoch
        trainer.model.train()

        for batch_idx, (data, target) in enumerate(trainer.train_loader):
            # Start profiling
            if profiler:
                profiler.start_iteration()
                profiler.start_data_loading()

            # Move to device
            data, target = data.to(trainer.device), target.to(trainer.device)

            if profiler:
                profiler.end_data_loading()
                profiler.start_forward()

            # Forward
            trainer.optimizer.zero_grad()
            output = trainer.model(data)
            loss = trainer.criterion(output, target)

            if profiler:
                profiler.end_forward()
                profiler.start_backward()

            # Backward
            loss.backward()

            if profiler:
                profiler.end_backward()
                profiler.start_optimizer()

            # Optimizer
            trainer.optimizer.step()

            if profiler:
                profiler.end_optimizer()

            # End profiling
            if profiler:
                metrics = profiler.end_iteration(
                    epoch=epoch,
                    iteration=batch_idx,
                    batch_size=data.size(0),
                    loss=loss.item()
                )

                if metrics:
                    metrics_history.append({
                        'epoch': epoch,
                        'iteration': batch_idx,
                        'loss': metrics.loss,
                        'throughput': metrics.throughput,
                        'iteration_time': metrics.iteration_time,
                        'data_loading_time': metrics.data_loading_time,
                        'forward_time': metrics.forward_time,
                        'backward_time': metrics.backward_time,
                        'optimizer_time': metrics.optimizer_time,
                        'gpu_utilization': metrics.gpu_utilization,
                        'gpu_memory_percent': metrics.gpu_memory_percent,
                    })

            # Update load balancer
            if load_balancer and batch_idx % args.rebalance_interval == 0:
                avg_metrics = profiler.get_average_metrics(n=10) if profiler else {}

                load_balancer.update_node_stats(rank, {
                    'utilization': avg_metrics.get('gpu_utilization', 0),
                    'memory_percent': avg_metrics.get('gpu_memory_percent', 0),
                    'iteration_time': avg_metrics.get('iteration_time', 0),
                })

                if rank == 0 and load_balancer.should_rebalance():
                    load_balancer.print_status()

            # Log
            if batch_idx % 10 == 0:
                logger.info(
                    f"Rank {rank} | Epoch {epoch} | Batch {batch_idx}/{len(trainer.train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

    # Save metrics
    if profiler and rank == 0:
        profiler.print_summary()

    if metrics_history:
        metrics_file = logs_dir / f"rank_{rank}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        logger.info(f"Saved metrics to {metrics_file}")

    # Cleanup
    trainer.cleanup()


def main():
    """Main function"""
    args = parse_args()

    logger.info("="*80)
    logger.info("Heterogeneous Distributed Training")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Load Balancing: {args.enable_load_balancing}")
    logger.info(f"Profiling: {args.enable_profiling}")
    logger.info("="*80)

    # Get rank
    rank = args.rank if args.rank is not None else int(os.environ.get('RANK', 0))

    # Train
    train_worker(rank, args)


if __name__ == "__main__":
    main()

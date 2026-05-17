"""
Distributed Training Framework
Supports PyTorch DDP with heterogeneous batch sizes
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Dict, Any, Callable
import logging
import time
import json
import numpy as np
from src.communication.compression import GradientCompressor, FP16Compressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributedTrainer:
    """
    Distributed trainer with support for heterogeneous GPUs
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset=None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        batch_size: int = 32,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        backend: str = "nccl",
        master_addr: str = "localhost",
        master_port: str = "29500",
        device_id: Optional[int] = None,
        heterogeneous_batch: bool = True,
        batch_size_multiplier: float = 1.0,
        accumulation_steps: int = 1,
        use_compression: bool = False,
        compression_type: str = 'fp16',
    ):
        """
        Initialize distributed trainer

        Args:
            model: Neural network model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            optimizer: Optimizer (auto-created if None)
            criterion: Loss function
            batch_size: Base batch size per GPU
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            backend: Distributed backend (nccl, gloo)
            master_addr: Master node address
            master_port: Master node port
            device_id: GPU device ID
            heterogeneous_batch: Enable heterogeneous batch sizing
            batch_size_multiplier: Batch size multiplier for this GPU
        """
        self.base_batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.backend = backend
        self.heterogeneous_batch = heterogeneous_batch
        self.batch_size_multiplier = batch_size_multiplier
        self.accumulation_steps = accumulation_steps
        self.use_compression = use_compression
        self.compression_type = compression_type

        # Setup distributed environment
        self._setup_distributed(master_addr, master_port, backend, device_id)

        # Adjust batch size for heterogeneous GPUs
        self.batch_size = self._get_adjusted_batch_size()

        # Setup device - handle both CUDA and CPU
        # Gloo backend supports both CPU and GPU, NCCL requires GPU
        if torch.cuda.is_available():
            try:
                self.device = torch.device(f"cuda:{self.local_rank}")
                torch.cuda.set_device(self.device)
                self.use_cuda = True
                logger.info(f"Rank {self.rank}: Using CUDA device {self.local_rank}")
            except Exception as e:
                logger.warning(f"Rank {self.rank}: Failed to set CUDA device, falling back to CPU: {e}")
                self.device = torch.device("cpu")
                self.use_cuda = False
                logger.info(f"Rank {self.rank}: Using CPU device")
        else:
            self.device = torch.device("cpu")
            self.use_cuda = False
            logger.info(f"Rank {self.rank}: Using CPU device")

        # Move model to device
        self.model = model.to(self.device)

        # Wrap model with DDP
        if self.use_cuda:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        else:
            # CPU-only DDP (no device_ids)
            self.model = DDP(
                self.model,
                find_unused_parameters=False
            )

        # Register communication hook for compression
        if self.use_compression and dist.is_initialized():
            self._register_comm_hook()

        # Setup criterion
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)

        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )
        else:
            self.optimizer = optimizer

        # Setup data loaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(val_dataset, shuffle=False) if val_dataset else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        logger.info(f"Rank {self.rank}: Initialized with batch_size={self.batch_size}")

    def _setup_distributed(self, master_addr: str, master_port: str,
                          backend: str, device_id: Optional[int]):
        """Setup distributed training environment"""

        # Set environment variables
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port

        # Get rank from environment or use device_id
        if 'RANK' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ.get('LOCAL_RANK', device_id or 0))
        else:
            self.rank = device_id or 0
            self.world_size = torch.cuda.device_count()
            self.local_rank = device_id or 0

        # Initialize process group
        if not dist.is_initialized():
            # Fallback to gloo if NCCL not available
            try:
                dist.init_process_group(
                    backend=backend,
                    init_method=f'tcp://{master_addr}:{master_port}',
                    world_size=self.world_size,
                    rank=self.rank
                )
                logger.info(f"Initialized process group: rank={self.rank}, world_size={self.world_size}, backend={backend}")
            except Exception as e:
                logger.warning(f"Failed to initialize with {backend}, falling back to gloo: {e}")
                dist.init_process_group(
                    backend='gloo',
                    init_method=f'tcp://{master_addr}:{master_port}',
                    world_size=self.world_size,
                    rank=self.rank
                )
                self.backend = 'gloo'

    def _register_comm_hook(self):
        """Register DDP communication hook"""
        if self.compression_type == 'fp16':
            hook = GradientCompressor.get_comm_hook('fp16')
            self.model.register_comm_hook(state=None, hook=hook)
            logger.info(f"Rank {self.rank}: Registered FP16 communication hook")
        elif self.compression_type == 'none':
            # No hook (default) or identify hook
            pass

    def _get_adjusted_batch_size(self) -> int:
        """Calculate adjusted batch size for heterogeneous GPUs"""
        if not self.heterogeneous_batch:
            return self.base_batch_size

        # Adjust based on multiplier
        adjusted = int(self.base_batch_size * self.batch_size_multiplier)
        adjusted = max(1, adjusted)  # At least 1

        logger.info(f"Rank {self.rank}: Adjusted batch size from {self.base_batch_size} to {adjusted} (multiplier={self.batch_size_multiplier:.2f})")
        return adjusted

    def _create_dataloader(self, dataset, shuffle: bool = True) -> Optional[DataLoader]:
        """Create distributed data loader"""
        if dataset is None:
            return None

        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        return loader

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        epoch_start = time.time()

        # Set epoch for sampler
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.current_epoch)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            batch_start = time.time()

            # Move to device
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
        # Zero gradients at start of epoch
        self.optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            batch_start = time.time()

            # Move to device
            data, target = data.to(self.device), target.to(self.device)

            # Determine if we should sync gradients
            is_sync_step = ((batch_idx + 1) % self.accumulation_steps == 0) or ((batch_idx + 1) == len(self.train_loader))

            # Forward + Backward context
            # If not syncing, use no_sync() to prevent all_reduce
            context = self.model.no_sync() if not is_sync_step else torch.enable_grad()  # enable_grad is dummy context
            
            # Only use no_sync if NOT syncing
            if not is_sync_step:
                context = self.model.no_sync()
            else:
                context = torch.enable_grad() # Dummy context

            with context:
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps
                loss.backward()

            # Step optimizer only on sync steps
            if is_sync_step:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Update metrics
            # Note: loss.item() is scaled, so we multiply back for logging
            batch_size = data.size(0)
            current_loss = loss.item() * self.accumulation_steps
            total_loss += current_loss * batch_size
            total_samples += batch_size
            self.global_step += 1

            batch_time = time.time() - batch_start

            if batch_idx % 10 == 0:
                logger.info(
                    f"Rank {self.rank} | Epoch {self.current_epoch} | "
                    f"Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {current_loss:.4f} | "
                    f"Time: {batch_time:.3f}s"
                )

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # Synchronize metrics across all processes
        avg_loss_tensor = torch.tensor([avg_loss], device=self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()

        metrics = {
            'loss': avg_loss,
            'epoch_time': epoch_time,
            'samples': total_samples,
            'throughput': total_samples / epoch_time if epoch_time > 0 else 0.0
        }

        return metrics

    def validate(self) -> Dict[str, float]:
        """Validate model"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                # Calculate accuracy
                pred = output.argmax(dim=1)
                correct = pred.eq(target).sum().item()

                batch_size = data.size(0)
                total_loss += loss.item() * batch_size
                total_correct += correct
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # Synchronize metrics
        metrics_tensor = torch.tensor([avg_loss, accuracy, total_samples], device=self.device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        # Calculate global averages
        global_samples = metrics_tensor[2].item()
        global_loss = metrics_tensor[0].item() / self.world_size
        global_accuracy = metrics_tensor[1].item() / self.world_size

        metrics = {
            'val_loss': global_loss,
            'val_accuracy': global_accuracy,
            'samples': global_samples
        }

        return metrics

    def train(self, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Train model for all epochs

        Args:
            callback: Optional callback function called after each epoch
                     with signature: callback(epoch, train_metrics, val_metrics)

        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'train_throughput': [],
            'val_loss': [],
            'val_accuracy': [],
        }

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['loss'])
            history['train_throughput'].append(train_metrics['throughput'])

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_accuracy'].append(val_metrics['val_accuracy'])

            # Log
            if self.rank == 0:
                logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss={train_metrics['loss']:.4f}, "
                    f"Val Loss={val_metrics.get('val_loss', 0):.4f}, "
                    f"Val Acc={val_metrics.get('val_accuracy', 0):.4f}, "
                    f"Throughput={train_metrics['throughput']:.2f} samples/s"
                )

            # Callback
            if callback:
                callback(epoch, train_metrics, val_metrics)

        return history

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        if self.rank == 0:
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
            }
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Loaded checkpoint from {path}")

    def cleanup(self):
        """Cleanup distributed training"""
        if dist.is_initialized():
            dist.destroy_process_group()
        logger.info(f"Rank {self.rank}: Cleanup complete")

    def update_communication_config(self, compression_type: str, accumulation_steps: int):
        """
        Update communication configuration dynamically
        """
        changed = False
        
        # Update accumulation steps
        if self.accumulation_steps != accumulation_steps:
            logger.info(f"Rank {self.rank}: Updating accumulation_steps from {self.accumulation_steps} to {accumulation_steps}")
            self.accumulation_steps = accumulation_steps
            changed = True
            
        # Update connection hook if compression type changed
        if self.compression_type != compression_type:
            logger.info(f"Rank {self.rank}: Updating compression_type from {self.compression_type} to {compression_type}")
            self.compression_type = compression_type
            
            # Since hooks are registered on the model, we can try to re-register or clear
            # In PyTorch DDP, register_comm_hook replaces the previous one.
            # But we only do this if dist is initialized
            if dist.is_initialized():
                self._register_comm_hook()
            changed = True
            
        return changed



def setup_distributed_training(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: str = "29500",
    backend: str = "nccl"
):
    """
    Helper function to setup distributed training
    Can be called from multiprocessing or Ray
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=f'tcp://{master_addr}:{master_port}',
            world_size=world_size,
            rank=rank
        )

    return rank


# ---------------------------------------------------------------------------
# DBS: Synchronous SGD with weighted gradient averaging (from SSGD())
# ---------------------------------------------------------------------------

def ssgd(
    model: "torch.nn.Module",
    rank: int,
    world_size: int,
    partition_size: "np.ndarray",
) -> float:
    """Weighted gradient all-reduce implementing the DBS SSGD algorithm.

    Each worker scales its gradients by its data partition ratio before
    all_reduce, so the global update is correctly weighted even when
    workers process different numbers of samples.

    Args:
        model:          The model whose ``.grad`` buffers to synchronise.
        rank:           This worker's rank.
        world_size:     Total number of workers.
        partition_size: Per-worker partition ratios (numpy array, sums to 1).

    Returns:
        Total synchronisation wall-time in seconds.
    """
    import time as _time

    wait_time = 0.0
    partition_size = np.asarray(partition_size, dtype=float)
    total = partition_size.sum()
    weight = float(partition_size[rank] / total) if total > 0 else 1.0 / world_size

    for param in model.parameters():
        if param.grad is None:
            continue
        scaled_grad = weight * param.grad.data
        req = dist.all_reduce(scaled_grad, op=dist.ReduceOp.SUM, async_op=True)
        t0 = _time.time()
        req.wait()
        wait_time += _time.time() - t0
        param.grad.data = scaled_grad

    return wait_time


# ---------------------------------------------------------------------------
# DBS: Worker process spawning (from init_processes())
# ---------------------------------------------------------------------------

def spawn_workers(
    world_size: int,
    fn,
    backend: str = "gloo",
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
) -> None:
    """Spawn *world_size* worker processes running *fn(rank, world_size)*.

    Each spawned process calls ``dist.init_process_group`` before *fn*.
    Mirrors the DBS ``init_processes`` / ``__main__`` spawning pattern
    without globals.

    Args:
        world_size:   Number of parallel workers to spawn.
        fn:           Callable with signature ``fn(rank, world_size)``.
        backend:      Distributed backend (``'gloo'`` or ``'nccl'``).
        master_addr:  Rendezvous IP.
        master_port:  Rendezvous port (string).
    """
    from torch.multiprocessing import Process, set_start_method

    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    def _init_and_run(rank, size, _fn, _backend, _addr, _port):
        os.environ["MASTER_ADDR"] = _addr
        os.environ["MASTER_PORT"] = _port
        dist.init_process_group(_backend, rank=rank, world_size=size)
        _fn(rank, size)

    processes = []
    for rank in range(world_size):
        p = Process(
            target=_init_and_run,
            args=(rank, world_size, fn, backend, master_addr, master_port),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

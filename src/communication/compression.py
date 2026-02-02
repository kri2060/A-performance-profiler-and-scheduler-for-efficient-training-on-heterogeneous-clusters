"""
Gradient Compression Module
Implements different compression strategies for Distributed Data Parallel communication.
"""

import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from typing import Any, Tuple

class GradientCompressor(ABC):
    """Abstract base class for gradient compressors"""
    
    @abstractmethod
    def compress(self, tensor: torch.Tensor) -> Any:
        """Compress a tensor"""
        pass

    @abstractmethod
    def decompress(self, compressed: Any) -> torch.Tensor:
        """Decompress a tensor"""
        pass

    @staticmethod
    def get_comm_hook(compressor_type: str = 'fp16'):
        """Get DDP communication hook"""
        if compressor_type == 'fp16':
            return fp16_compress_hook
        elif compressor_type == 'none':
            return default_hook
        else:
            raise ValueError(f"Unknown compressor type: {compressor_type}")


class FP16Compressor(GradientCompressor):
    """FP16 Compression (half precision)"""
    
    def compress(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.half()

    def decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float()


def default_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """Default DDP hook (no compression)"""
    # This is basically what DDP does by default: all_reduce the bucket buffer
    
    # We need to return a future. 
    # Since we can't easily access the default C++ implementation, 
    # we implement a simple all_reduce here.
    
    tensor = bucket.buffer()
    fut = dist.all_reduce(tensor, op=dist.ReduceOp.AVG, async_op=True).get_future()
    
    def return_tensor(fut):
        return fut.value()[0]
        
    return fut.then(return_tensor)


def fp16_compress_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    DDP Communication Hook for FP16 Compression.
    
    1. Casts gradients to FP16.
    2. Performs AllReduce.
    3. Casts back to FP32.
    """
    process_group = state if state is not None else dist.group.WORLD
    
    # Compress (FP32 -> FP16)
    tensor = bucket.buffer()
    compressed_tensor = tensor.half()
    
    # AllReduce
    # Note: We use SUM and then divide by world_size manually if needed, 
    # or just use AVG if backend supports it for FP16. NCCL usually does.
    fut = dist.all_reduce(
        compressed_tensor, 
        op=dist.ReduceOp.AVG, 
        group=process_group, 
        async_op=True
    ).get_future()
    
    def decompress(fut):
        # Decompress (FP16 -> FP32)
        synced_compressed = fut.value()[0]
        return synced_compressed.float()
        
    return fut.then(decompress)

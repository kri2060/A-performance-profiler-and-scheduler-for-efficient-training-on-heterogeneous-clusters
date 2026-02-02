#!/usr/bin/env python3
"""
Redis Metrics Writer
Helper module to write GPU and training metrics to Redis
"""

import redis
import socket
import time
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class RedisMetricsWriter:
    """Write performance metrics to Redis for Grafana visualization"""

    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        ttl: int = 3600  # 1 hour default TTL
    ):
        """
        Initialize Redis metrics writer

        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            ttl: Time-to-live for metrics in seconds
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.ttl = ttl
        self.hostname = socket.gethostname()

        try:
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def write_gpu_metrics(
        self,
        gpu_id: int,
        utilization: float,
        memory_used: float,
        memory_total: float,
        temperature: float,
        power_draw: float,
        power_limit: float,
        hostname: Optional[str] = None
    ):
        """
        Write GPU metrics to Redis

        Args:
            gpu_id: GPU device ID
            utilization: GPU utilization percentage (0-100)
            memory_used: Memory used in MB
            memory_total: Total memory in MB
            temperature: Temperature in Celsius
            power_draw: Power consumption in Watts
            power_limit: Power limit in Watts
            hostname: Optional hostname override
        """
        host = hostname or self.hostname
        timestamp = int(time.time())

        metrics = {
            f'gpu:{host}:{gpu_id}:utilization': utilization,
            f'gpu:{host}:{gpu_id}:memory_used': memory_used,
            f'gpu:{host}:{gpu_id}:memory_total': memory_total,
            f'gpu:{host}:{gpu_id}:temperature': temperature,
            f'gpu:{host}:{gpu_id}:power_draw': power_draw,
            f'gpu:{host}:{gpu_id}:power_limit': power_limit,
            f'gpu:{host}:{gpu_id}:timestamp': timestamp,
        }

        try:
            pipe = self.redis_client.pipeline()
            for key, value in metrics.items():
                pipe.set(key, value, ex=self.ttl)
            pipe.execute()
        except Exception as e:
            logger.error(f"Error writing GPU metrics: {e}")

    def write_training_metrics(
        self,
        rank: int,
        epoch: int,
        iteration: int,
        loss: float,
        iteration_time: float,
        throughput: Optional[float] = None,
        hostname: Optional[str] = None
    ):
        """
        Write training metrics to Redis

        Args:
            rank: Process rank in distributed training
            epoch: Current epoch number
            iteration: Current iteration number
            loss: Training loss value
            iteration_time: Time per iteration in seconds
            throughput: Training throughput in samples/sec
            hostname: Optional hostname override
        """
        host = hostname or self.hostname
        timestamp = int(time.time())

        metrics = {
            f'training:{host}:{rank}:epoch': epoch,
            f'training:{host}:{rank}:iteration': iteration,
            f'training:{host}:{rank}:loss': loss,
            f'training:{host}:{rank}:iteration_time': iteration_time,
            f'training:{host}:{rank}:timestamp': timestamp,
        }

        if throughput is not None:
            metrics['training:throughput'] = throughput

        try:
            pipe = self.redis_client.pipeline()
            for key, value in metrics.items():
                pipe.set(key, value, ex=self.ttl)
            pipe.execute()
        except Exception as e:
            logger.error(f"Error writing training metrics: {e}")

    def write_cluster_metrics(
        self,
        avg_gpu_utilization: float,
        active_gpus: int,
        training_status: int
    ):
        """
        Write cluster-level metrics to Redis

        Args:
            avg_gpu_utilization: Average GPU utilization across cluster
            active_gpus: Number of active GPUs
            training_status: Training status (0=stopped, 1=running)
        """
        timestamp = int(time.time())

        metrics = {
            'cluster:avg_gpu_utilization': avg_gpu_utilization,
            'cluster:active_gpus': active_gpus,
            'training:status': training_status,
            'cluster:timestamp': timestamp,
        }

        try:
            pipe = self.redis_client.pipeline()
            for key, value in metrics.items():
                pipe.set(key, value, ex=self.ttl)
            pipe.execute()
        except Exception as e:
            logger.error(f"Error writing cluster metrics: {e}")

    def write_custom_metric(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        Write a custom metric to Redis

        Args:
            key: Metric key
            value: Metric value
            ttl: Optional TTL override
        """
        try:
            self.redis_client.set(key, value, ex=ttl or self.ttl)
        except Exception as e:
            logger.error(f"Error writing custom metric {key}: {e}")

    def get_metric(self, key: str) -> Optional[str]:
        """
        Get a metric value from Redis

        Args:
            key: Metric key

        Returns:
            Metric value or None
        """
        try:
            return self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Error reading metric {key}: {e}")
            return None

    def close(self):
        """Close Redis connection"""
        try:
            self.redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")


# Example usage
if __name__ == '__main__':
    import pynvml

    # Initialize Redis writer
    writer = RedisMetricsWriter(redis_host='localhost')

    # Example: Write GPU metrics
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    for gpu_id in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

        # Get GPU metrics
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0

        # Write to Redis
        writer.write_gpu_metrics(
            gpu_id=gpu_id,
            utilization=utilization,
            memory_used=memory_info.used / (1024 ** 2),  # Bytes to MB
            memory_total=memory_info.total / (1024 ** 2),
            temperature=temperature,
            power_draw=power_draw,
            power_limit=power_limit
        )

    pynvml.nvmlShutdown()
    writer.close()

    print("Metrics written to Redis successfully!")

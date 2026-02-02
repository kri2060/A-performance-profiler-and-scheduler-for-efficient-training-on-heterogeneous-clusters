#!/usr/bin/env python3
"""
Redis Exporter for Grafana
Exports GPU metrics from Redis to Prometheus-compatible format
"""

import os
import time
import redis
from prometheus_client import start_http_server, Gauge, Info
from typing import Dict, List
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedisMetricsExporter:
    """Export GPU and training metrics from Redis to Prometheus format"""

    def __init__(self, redis_host: str = 'redis', redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )

        # GPU Metrics
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id', 'hostname']
        )
        self.gpu_memory_used = Gauge(
            'gpu_memory_used_mb',
            'GPU memory used in MB',
            ['gpu_id', 'hostname']
        )
        self.gpu_memory_total = Gauge(
            'gpu_memory_total_mb',
            'GPU total memory in MB',
            ['gpu_id', 'hostname']
        )
        self.gpu_temperature = Gauge(
            'gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['gpu_id', 'hostname']
        )
        self.gpu_power_draw = Gauge(
            'gpu_power_draw_watts',
            'GPU power draw in Watts',
            ['gpu_id', 'hostname']
        )
        self.gpu_power_limit = Gauge(
            'gpu_power_limit_watts',
            'GPU power limit in Watts',
            ['gpu_id', 'hostname']
        )

        # Training Metrics
        self.training_loss = Gauge(
            'training_loss',
            'Current training loss',
            ['rank', 'hostname']
        )
        self.training_iteration_time = Gauge(
            'training_iteration_time_seconds',
            'Training iteration time in seconds',
            ['rank', 'hostname']
        )
        self.training_throughput = Gauge(
            'training_throughput_samples_per_sec',
            'Training throughput in samples per second',
            ['rank', 'hostname']
        )
        self.training_epoch = Gauge(
            'training_epoch',
            'Current training epoch',
            ['rank', 'hostname']
        )

        # Cluster Metrics
        self.cluster_avg_gpu_utilization = Gauge(
            'cluster_avg_gpu_utilization_percent',
            'Cluster average GPU utilization'
        )
        self.cluster_active_gpus = Gauge(
            'cluster_active_gpus',
            'Number of active GPUs in cluster'
        )
        self.cluster_training_status = Gauge(
            'cluster_training_status',
            'Training status (0=stopped, 1=running)'
        )

        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")

    def collect_gpu_metrics(self):
        """Collect GPU metrics from Redis"""
        try:
            # Get all GPU keys
            gpu_keys = self.redis_client.keys('gpu:*:utilization')

            for key in gpu_keys:
                # Parse key: gpu:hostname:gpu_id:utilization
                parts = key.split(':')
                if len(parts) >= 3:
                    hostname = parts[1]
                    gpu_id = parts[2] if len(parts) > 3 else parts[1]

                    # Get utilization
                    util = self.redis_client.get(key)
                    if util:
                        self.gpu_utilization.labels(
                            gpu_id=gpu_id,
                            hostname=hostname
                        ).set(float(util))

                    # Get memory usage
                    mem_key = f"gpu:{hostname}:{gpu_id}:memory_used"
                    mem_used = self.redis_client.get(mem_key)
                    if mem_used:
                        self.gpu_memory_used.labels(
                            gpu_id=gpu_id,
                            hostname=hostname
                        ).set(float(mem_used))

                    # Get total memory
                    mem_total_key = f"gpu:{hostname}:{gpu_id}:memory_total"
                    mem_total = self.redis_client.get(mem_total_key)
                    if mem_total:
                        self.gpu_memory_total.labels(
                            gpu_id=gpu_id,
                            hostname=hostname
                        ).set(float(mem_total))

                    # Get temperature
                    temp_key = f"gpu:{hostname}:{gpu_id}:temperature"
                    temp = self.redis_client.get(temp_key)
                    if temp:
                        self.gpu_temperature.labels(
                            gpu_id=gpu_id,
                            hostname=hostname
                        ).set(float(temp))

                    # Get power draw
                    power_key = f"gpu:{hostname}:{gpu_id}:power_draw"
                    power = self.redis_client.get(power_key)
                    if power:
                        self.gpu_power_draw.labels(
                            gpu_id=gpu_id,
                            hostname=hostname
                        ).set(float(power))

                    # Get power limit
                    power_limit_key = f"gpu:{hostname}:{gpu_id}:power_limit"
                    power_limit = self.redis_client.get(power_limit_key)
                    if power_limit:
                        self.gpu_power_limit.labels(
                            gpu_id=gpu_id,
                            hostname=hostname
                        ).set(float(power_limit))

        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")

    def collect_training_metrics(self):
        """Collect training metrics from Redis"""
        try:
            # Get all training keys
            training_keys = self.redis_client.keys('training:*:loss')

            for key in training_keys:
                # Parse key: training:hostname:rank:loss
                parts = key.split(':')
                if len(parts) >= 3:
                    hostname = parts[1]
                    rank = parts[2] if len(parts) > 3 else parts[1]

                    # Get loss
                    loss = self.redis_client.get(key)
                    if loss:
                        self.training_loss.labels(
                            rank=rank,
                            hostname=hostname
                        ).set(float(loss))

                    # Get iteration time
                    iter_key = f"training:{hostname}:{rank}:iteration_time"
                    iter_time = self.redis_client.get(iter_key)
                    if iter_time:
                        self.training_iteration_time.labels(
                            rank=rank,
                            hostname=hostname
                        ).set(float(iter_time))

                    # Get epoch
                    epoch_key = f"training:{hostname}:{rank}:epoch"
                    epoch = self.redis_client.get(epoch_key)
                    if epoch:
                        self.training_epoch.labels(
                            rank=rank,
                            hostname=hostname
                        ).set(float(epoch))

            # Get throughput
            throughput = self.redis_client.get('training:throughput')
            if throughput:
                self.training_throughput.labels(
                    rank='global',
                    hostname='cluster'
                ).set(float(throughput))

        except Exception as e:
            logger.error(f"Error collecting training metrics: {e}")

    def collect_cluster_metrics(self):
        """Collect cluster-level metrics from Redis"""
        try:
            # Average GPU utilization
            avg_util = self.redis_client.get('cluster:avg_gpu_utilization')
            if avg_util:
                self.cluster_avg_gpu_utilization.set(float(avg_util))

            # Active GPUs
            active_gpus = self.redis_client.get('cluster:active_gpus')
            if active_gpus:
                self.cluster_active_gpus.set(float(active_gpus))

            # Training status
            status = self.redis_client.get('training:status')
            if status:
                self.cluster_training_status.set(float(status))

        except Exception as e:
            logger.error(f"Error collecting cluster metrics: {e}")

    def collect_all_metrics(self):
        """Collect all metrics from Redis"""
        self.collect_gpu_metrics()
        self.collect_training_metrics()
        self.collect_cluster_metrics()

    def run(self, port: int = 9121, interval: int = 5):
        """Run the exporter"""
        # Start Prometheus HTTP server
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")

        # Continuously collect metrics
        while True:
            try:
                self.collect_all_metrics()
                time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("Exporter stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(interval)


def main():
    """Main entry point"""
    redis_host = os.getenv('REDIS_HOST', 'redis')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    metrics_port = int(os.getenv('METRICS_PORT', 9121))
    collection_interval = int(os.getenv('COLLECTION_INTERVAL', 5))

    exporter = RedisMetricsExporter(redis_host, redis_port)
    exporter.run(port=metrics_port, interval=collection_interval)


if __name__ == '__main__':
    main()

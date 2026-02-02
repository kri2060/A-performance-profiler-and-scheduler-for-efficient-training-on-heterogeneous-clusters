#!/usr/bin/env python3
"""
Test script to populate Redis with sample GPU metrics for Grafana dashboard
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from monitoring.redis_metrics import RedisMetricsWriter
import time
import random

def generate_sample_metrics():
    """Generate and write sample GPU metrics to Redis"""

    print("Connecting to Redis...")
    writer = RedisMetricsWriter(redis_host='localhost', redis_port=6379)

    print("Writing sample GPU metrics...")

    # Simulate 4 GPUs
    num_gpus = 4

    for i in range(10):  # Write 10 samples with 2-second intervals
        print(f"\nIteration {i+1}/10")

        # Write GPU metrics for each GPU
        for gpu_id in range(num_gpus):
            utilization = random.uniform(60, 95)
            memory_used = random.uniform(8000, 15000)
            memory_total = 16384
            temperature = random.uniform(65, 80)
            power_draw = random.uniform(200, 280)
            power_limit = 300

            writer.write_gpu_metrics(
                gpu_id=gpu_id,
                utilization=utilization,
                memory_used=memory_used,
                memory_total=memory_total,
                temperature=temperature,
                power_draw=power_draw,
                power_limit=power_limit,
                hostname='test-node'
            )

            print(f"  GPU {gpu_id}: {utilization:.1f}% util, {memory_used:.0f}MB mem, {temperature:.1f}°C")

        # Write training metrics
        for rank in range(num_gpus):
            epoch = 5
            iteration = 1000 + i * 10
            loss = 0.5 * (1 - i * 0.05) + random.uniform(-0.02, 0.02)
            iteration_time = random.uniform(0.14, 0.18)

            writer.write_training_metrics(
                rank=rank,
                epoch=epoch,
                iteration=iteration,
                loss=loss,
                iteration_time=iteration_time,
                throughput=256.0,
                hostname='test-node'
            )

        # Write cluster metrics
        avg_util = sum([random.uniform(60, 95) for _ in range(num_gpus)]) / num_gpus
        writer.write_cluster_metrics(
            avg_gpu_utilization=avg_util,
            active_gpus=num_gpus,
            training_status=1  # Running
        )

        print(f"  Cluster: {avg_util:.1f}% avg utilization, {num_gpus} active GPUs")

        if i < 9:  # Don't sleep on last iteration
            time.sleep(2)

    print("\n✅ Successfully wrote sample metrics to Redis!")
    print(f"\n📊 Grafana Dashboard: http://localhost:3000")
    print("   Username: admin")
    print("   Password: admin")
    print("\nThe dashboard should now display the sample data!")

    writer.close()

if __name__ == '__main__':
    try:
        generate_sample_metrics()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

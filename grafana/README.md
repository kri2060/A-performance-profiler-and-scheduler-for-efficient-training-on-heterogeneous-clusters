# Grafana Dashboard Setup

This directory contains the configuration for Grafana-based monitoring and visualization of GPU performance metrics and distributed training statistics.

## Architecture

```
┌─────────────────┐      ┌─────────────┐      ┌──────────────────┐
│   GPU Profiler  │─────▶│    Redis    │◀────▶│ Redis Exporter   │
│  Training Loop  │      │  (Metrics)  │      │  (Prometheus)    │
└─────────────────┘      └─────────────┘      └──────────────────┘
                                │                       │
                                │                       │
                                ▼                       ▼
                         ┌──────────────────────────────────┐
                         │          Grafana                 │
                         │  (Visualization & Dashboards)    │
                         └──────────────────────────────────┘
```

## Components

### 1. Redis (Time-Series Database)
- **Port**: 6379
- **Purpose**: Stores GPU metrics, training statistics, and cluster information
- **Persistence**: Data persisted to disk with AOF (Append-Only File)
- **TTL**: Metrics expire after 1 hour by default

### 2. Grafana (Dashboard & Visualization)
- **Port**: 3000
- **Default Credentials**: admin / admin
- **Features**:
  - Real-time GPU utilization monitoring
  - Memory usage tracking
  - Temperature and power consumption graphs
  - Training metrics (loss, throughput, iteration time)
  - Cluster-level statistics

### 3. Redis Exporter
- **Port**: 9121
- **Purpose**: Exports Redis metrics in Prometheus format for Grafana
- **Update Interval**: 5 seconds

## Directory Structure

```
grafana/
├── README.md                          # This file
├── provisioning/
│   ├── datasources/
│   │   └── redis.yml                  # Redis datasource configuration
│   └── dashboards/
│       └── dashboard.yml              # Dashboard provisioning config
└── dashboards/
    └── gpu-performance.json           # GPU Performance Dashboard
```

## Quick Start

### 1. Start Services

```bash
# Start all services (Redis, Grafana, Redis Exporter)
docker-compose up -d redis grafana redis-exporter

# Check service status
docker-compose ps
```

### 2. Access Grafana

1. Open browser: http://localhost:3000
2. Login with credentials:
   - Username: `admin`
   - Password: `admin`
3. The GPU Performance dashboard will be auto-loaded

### 3. Write Metrics to Redis

Use the `RedisMetricsWriter` class in your Python code:

```python
from src.monitoring.redis_metrics import RedisMetricsWriter

# Initialize writer
writer = RedisMetricsWriter(redis_host='localhost', redis_port=6379)

# Write GPU metrics
writer.write_gpu_metrics(
    gpu_id=0,
    utilization=85.5,
    memory_used=8192,
    memory_total=16384,
    temperature=72.0,
    power_draw=250.0,
    power_limit=300.0
)

# Write training metrics
writer.write_training_metrics(
    rank=0,
    epoch=5,
    iteration=1000,
    loss=0.234,
    iteration_time=0.152,
    throughput=256.0
)

# Write cluster metrics
writer.write_cluster_metrics(
    avg_gpu_utilization=75.3,
    active_gpus=4,
    training_status=1  # 1 = running
)
```

## Dashboard Panels

The GPU Performance dashboard includes:

### Time-Series Graphs
1. **GPU Utilization (%)** - Per-GPU utilization over time
2. **GPU Memory Usage (MB)** - Memory consumption per GPU
3. **GPU Temperature (°C)** - Temperature monitoring with thresholds
4. **GPU Power Draw (W)** - Power consumption tracking
5. **Training Iteration Time** - Performance per iteration
6. **Training Loss** - Loss curve over time

### Real-Time Stats
7. **Cluster Avg GPU Utilization** - Gauge showing cluster average
8. **Active GPUs** - Number of GPUs currently in use
9. **Training Throughput** - Samples processed per second
10. **Training Status** - Current training state (Running/Stopped)

## Redis Key Schema

### GPU Metrics
```
gpu:{hostname}:{gpu_id}:utilization       # GPU utilization %
gpu:{hostname}:{gpu_id}:memory_used       # Memory used (MB)
gpu:{hostname}:{gpu_id}:memory_total      # Total memory (MB)
gpu:{hostname}:{gpu_id}:temperature       # Temperature (°C)
gpu:{hostname}:{gpu_id}:power_draw        # Power draw (W)
gpu:{hostname}:{gpu_id}:power_limit       # Power limit (W)
```

### Training Metrics
```
training:{hostname}:{rank}:epoch          # Current epoch
training:{hostname}:{rank}:iteration      # Current iteration
training:{hostname}:{rank}:loss           # Training loss
training:{hostname}:{rank}:iteration_time # Iteration time (seconds)
training:throughput                       # Samples/sec
training:status                           # 0=stopped, 1=running
```

### Cluster Metrics
```
cluster:avg_gpu_utilization               # Cluster average GPU util %
cluster:active_gpus                       # Number of active GPUs
cluster:timestamp                         # Last update timestamp
```

## Configuration

### Change Update Interval

Edit `docker-compose.yml`:

```yaml
redis-exporter:
  environment:
    - COLLECTION_INTERVAL=5  # Change to desired interval (seconds)
```

### Change Redis TTL

In your Python code:

```python
writer = RedisMetricsWriter(
    redis_host='localhost',
    ttl=7200  # 2 hours
)
```

### Add Custom Metrics

```python
# Write custom metric
writer.write_custom_metric(
    key='custom:metric:name',
    value=42.0,
    ttl=3600
)
```

## Troubleshooting

### Grafana shows "No Data"
1. Check Redis is running: `docker-compose ps redis`
2. Verify metrics are being written:
   ```bash
   docker exec -it hetero-redis redis-cli
   > KEYS gpu:*
   > GET gpu:hostname:0:utilization
   ```
3. Check Redis exporter logs:
   ```bash
   docker-compose logs redis-exporter
   ```

### Redis Connection Error
1. Check Redis container: `docker-compose logs redis`
2. Verify network connectivity: `docker network ls`
3. Check Redis port mapping: `docker-compose ps redis`

### Dashboard Not Auto-Loading
1. Check provisioning volume mount in `docker-compose.yml`
2. Verify files exist in `grafana/provisioning/`
3. Restart Grafana: `docker-compose restart grafana`

## Performance Tips

1. **Reduce Update Frequency**: Increase `COLLECTION_INTERVAL` to reduce overhead
2. **Adjust TTL**: Set appropriate TTL based on monitoring needs
3. **Use Pipeline**: The `RedisMetricsWriter` uses Redis pipelines for efficiency
4. **Monitor Redis Memory**: Check Redis memory usage with `INFO memory`

## Customization

### Add New Dashboard Panel

1. Edit `grafana/dashboards/gpu-performance.json`
2. Add new panel configuration in the `panels` array
3. Restart Grafana or wait for auto-reload (10 seconds)

### Add New Data Source

1. Create new YAML in `grafana/provisioning/datasources/`
2. Follow the same format as `redis.yml`
3. Restart Grafana

## Production Deployment

For production use:

1. **Change Default Password**:
   ```yaml
   environment:
     - GF_SECURITY_ADMIN_PASSWORD=your_secure_password
   ```

2. **Enable HTTPS**:
   - Configure reverse proxy (nginx, Caddy)
   - Or use Grafana's built-in HTTPS

3. **Set Up Alerts**:
   - Configure alert rules in Grafana
   - Add notification channels (Slack, Email, etc.)

4. **Enable Authentication**:
   - Configure OAuth, LDAP, or other auth methods
   - Disable anonymous access

5. **Persistent Volumes**:
   - Already configured in `docker-compose.yml`
   - Backup volumes regularly

## References

- [Grafana Documentation](https://grafana.com/docs/grafana/latest/)
- [Redis Datasource Plugin](https://grafana.com/grafana/plugins/redis-datasource/)
- [Prometheus Client Python](https://github.com/prometheus/client_python)

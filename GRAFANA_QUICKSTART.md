# Grafana Dashboard Quick Start

## 🚀 Services Running

✅ **Redis** - Running on port 6379
✅ **Grafana** - Running on port 3000
✅ **Sample Data** - Populated in Redis

## 📊 Access the Dashboard

**URL:** http://localhost:3000

**Credentials:**
- Username: `admin`
- Password: `admin`

## 🎯 Current Setup

The dashboard is now running with:
- **4 simulated GPUs** with metrics
- **Real-time data** from Redis
- **Pre-configured dashboard** with 10 panels

## 📈 Dashboard Panels

1. **GPU Utilization (%)** - Per-GPU usage over time
2. **GPU Memory Usage (MB)** - Memory consumption
3. **GPU Temperature (°C)** - Temperature monitoring
4. **GPU Power Draw (W)** - Power consumption
5. **Training Iteration Time** - Performance metrics
6. **Training Loss** - Training progress
7. **Cluster Avg GPU Utilization** - Gauge view
8. **Active GPUs** - Number of GPUs
9. **Training Throughput** - Samples/sec
10. **Training Status** - Running/Stopped

## 🔧 Running Services

Check service status:
```bash
docker ps --filter name=hetero
```

View logs:
```bash
docker logs hetero-grafana
docker logs hetero-redis
```

Stop services:
```bash
docker stop hetero-grafana hetero-redis
```

Start services:
```bash
docker start hetero-redis hetero-grafana
```

## 📝 Generate Test Data

Run the test script to populate metrics:
```bash
source venv/bin/activate
python test_dashboard.py
```

## 🔌 Using in Your Code

```python
from src.monitoring.redis_metrics import RedisMetricsWriter

# Initialize writer
writer = RedisMetricsWriter(redis_host='localhost')

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
```

## 🐛 Troubleshooting

### Grafana shows "No Data"
```bash
# Check if Redis has data
docker exec hetero-redis redis-cli KEYS "*"

# Re-run test script
python test_dashboard.py
```

### Can't connect to Grafana
```bash
# Check if Grafana is running
docker ps | grep grafana

# Check Grafana logs
docker logs hetero-grafana
```

### Redis connection error
```bash
# Check if Redis is running
docker ps | grep redis

# Test Redis
docker exec hetero-redis redis-cli ping
```

## 📚 Documentation

- Full setup guide: [grafana/README.md](grafana/README.md)
- Dashboard JSON: [grafana/dashboards/gpu-performance.json](grafana/dashboards/gpu-performance.json)
- Python API: [src/monitoring/redis_metrics.py](src/monitoring/redis_metrics.py)

## ⚙️ Configuration

The dashboard auto-refreshes every 5 seconds. To change:
1. Open Grafana dashboard
2. Click the refresh icon (top right)
3. Select desired interval

## 🎨 Customization

Edit the dashboard:
1. Go to http://localhost:3000
2. Open the "GPU Performance & Training Monitor" dashboard
3. Click the gear icon (Settings) → Edit
4. Modify panels, add new ones, or change queries

Save changes:
- Changes are saved to Grafana's internal database
- To persist, export as JSON and replace [grafana/dashboards/gpu-performance.json](grafana/dashboards/gpu-performance.json)

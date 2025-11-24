"""
FastAPI Backend for Heterogeneous Cluster Dashboard
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import asyncio
from pathlib import Path
from datetime import datetime
import os

app = FastAPI(
    title="Heterogeneous Cluster Dashboard API",
    description="API for monitoring distributed training on heterogeneous GPU clusters",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"


# Pydantic Models
class GPUProfile(BaseModel):
    device_id: int
    name: str
    compute_capability: str
    total_memory_mb: float
    memory_bandwidth_gbps: float
    cuda_cores: Optional[int]
    sm_count: int
    clock_rate_mhz: float
    memory_clock_rate_mhz: float
    pcie_link_gen: int
    pcie_link_width: int
    compute_score: float


class SystemProfile(BaseModel):
    hostname: str
    platform: str
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_frequency_mhz: float
    ram_total_gb: float
    ram_available_gb: float
    network_interfaces: List[str]
    ip_address: str


class TrainingMetric(BaseModel):
    timestamp: Optional[float] = None
    epoch: int
    iteration: int
    loss: float
    throughput: float
    iteration_time: float
    data_loading_time: float
    forward_time: float
    backward_time: float
    optimizer_time: float
    gpu_utilization: float
    gpu_memory_percent: float


class JobStatus(BaseModel):
    id: str
    name: str
    model: str
    dataset: str
    status: str  # running, completed, failed, pending
    start_time: str
    epochs_completed: int
    total_epochs: int
    current_loss: float
    throughput: float
    nodes: int


class ClusterStatus(BaseModel):
    total_nodes: int
    active_nodes: int
    total_gpus: int
    avg_gpu_utilization: float
    avg_memory_usage: float
    total_throughput: float


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# API Routes
@app.get("/")
async def root():
    return {"message": "Heterogeneous Cluster Dashboard API", "version": "1.0.0"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/nodes", response_model=List[GPUProfile])
async def get_nodes():
    """Get all GPU profiles in the cluster"""
    gpu_profiles_path = CONFIGS_DIR / "gpu_profiles.json"

    if not gpu_profiles_path.exists():
        return []

    with open(gpu_profiles_path, 'r') as f:
        profiles = json.load(f)

    return profiles


@app.get("/api/system")
async def get_system_info():
    """Get system profile"""
    system_profile_path = CONFIGS_DIR / "system_profile.json"

    if not system_profile_path.exists():
        return {}

    with open(system_profile_path, 'r') as f:
        profile = json.load(f)

    return profile


@app.get("/api/cluster/status", response_model=ClusterStatus)
async def get_cluster_status():
    """Get overall cluster status"""
    gpu_profiles_path = CONFIGS_DIR / "gpu_profiles.json"

    if not gpu_profiles_path.exists():
        return ClusterStatus(
            total_nodes=0,
            active_nodes=0,
            total_gpus=0,
            avg_gpu_utilization=0.0,
            avg_memory_usage=0.0,
            total_throughput=0.0
        )

    with open(gpu_profiles_path, 'r') as f:
        profiles = json.load(f)

    # Calculate aggregated metrics
    total_gpus = len(profiles)

    # Get latest metrics if available
    latest_metrics = await get_latest_metrics()

    avg_gpu_util = 0.0
    avg_mem_usage = 0.0
    total_throughput = 0.0

    if latest_metrics:
        gpu_utils = [m.get('gpu_utilization', 0) for m in latest_metrics.values()]
        mem_usages = [m.get('gpu_memory_percent', 0) for m in latest_metrics.values()]
        throughputs = [m.get('throughput', 0) for m in latest_metrics.values()]

        if gpu_utils:
            avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
        if mem_usages:
            avg_mem_usage = sum(mem_usages) / len(mem_usages)
        total_throughput = sum(throughputs)

    return ClusterStatus(
        total_nodes=1,  # Single node for now
        active_nodes=1,
        total_gpus=total_gpus,
        avg_gpu_utilization=avg_gpu_util,
        avg_memory_usage=avg_mem_usage,
        total_throughput=total_throughput
    )


@app.get("/api/jobs")
async def get_jobs():
    """Get all training jobs"""
    jobs = []

    # Scan experiments directory for jobs
    if EXPERIMENTS_DIR.exists():
        for exp_dir in EXPERIMENTS_DIR.iterdir():
            if exp_dir.is_dir() and exp_dir.name not in ['configs', 'logs', 'results']:
                logs_dir = exp_dir / "logs"

                if logs_dir.exists():
                    # Get metrics from job
                    metrics_files = list(logs_dir.glob("rank_*_metrics.json"))

                    if metrics_files:
                        # Load first metrics file
                        with open(metrics_files[0], 'r') as f:
                            metrics = json.load(f)

                        if metrics:
                            last_metric = metrics[-1]
                            job = {
                                "id": exp_dir.name,
                                "name": exp_dir.name.replace('_', ' ').title(),
                                "model": "Unknown",
                                "dataset": "Unknown",
                                "status": "completed",
                                "start_time": datetime.now().isoformat(),
                                "epochs_completed": last_metric.get('epoch', 0) + 1,
                                "total_epochs": last_metric.get('epoch', 0) + 1,
                                "current_loss": last_metric.get('loss', 0),
                                "throughput": last_metric.get('throughput', 0),
                                "nodes": len(metrics_files)
                            }
                            jobs.append(job)

    return jobs


@app.get("/api/jobs/{job_id}/metrics")
async def get_job_metrics(job_id: str):
    """Get metrics for a specific job"""
    job_dir = EXPERIMENTS_DIR / job_id / "logs"

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    all_metrics = {}

    for metrics_file in job_dir.glob("rank_*_metrics.json"):
        rank = metrics_file.stem.split('_')[1]
        with open(metrics_file, 'r') as f:
            all_metrics[f"rank_{rank}"] = json.load(f)

    return all_metrics


@app.get("/api/metrics/latest")
async def get_latest_metrics():
    """Get latest metrics from all experiments"""
    latest = {}

    if EXPERIMENTS_DIR.exists():
        for exp_dir in EXPERIMENTS_DIR.iterdir():
            if exp_dir.is_dir() and exp_dir.name not in ['configs', 'logs', 'results']:
                logs_dir = exp_dir / "logs"

                if logs_dir.exists():
                    for metrics_file in logs_dir.glob("rank_*_metrics.json"):
                        rank = metrics_file.stem.split('_')[1]
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                            if metrics:
                                latest[f"{exp_dir.name}_rank_{rank}"] = metrics[-1]

    return latest


@app.get("/api/metrics/history/{experiment_name}")
async def get_metrics_history(experiment_name: str, limit: int = 100):
    """Get historical metrics for an experiment"""
    exp_dir = EXPERIMENTS_DIR / experiment_name / "logs"

    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")

    history = {}

    for metrics_file in exp_dir.glob("rank_*_metrics.json"):
        rank = metrics_file.stem.split('_')[1]
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            # Limit to last N entries
            history[f"rank_{rank}"] = metrics[-limit:] if len(metrics) > limit else metrics

    return history


@app.get("/api/experiments")
async def get_experiments():
    """Get list of all experiments"""
    experiments = []

    if EXPERIMENTS_DIR.exists():
        for exp_dir in EXPERIMENTS_DIR.iterdir():
            if exp_dir.is_dir() and exp_dir.name not in ['configs', 'logs', 'results']:
                logs_dir = exp_dir / "logs"

                if logs_dir.exists():
                    metrics_count = len(list(logs_dir.glob("rank_*_metrics.json")))
                    experiments.append({
                        "name": exp_dir.name,
                        "path": str(exp_dir),
                        "ranks": metrics_count,
                        "created": datetime.fromtimestamp(exp_dir.stat().st_mtime).isoformat()
                    })

    return experiments


# WebSocket endpoints
@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics streaming"""
    await manager.connect(websocket)

    try:
        while True:
            # Send latest metrics every second
            latest_metrics = await get_latest_metrics()
            cluster_status = await get_cluster_status()

            await websocket.send_json({
                "type": "metrics_update",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "latest_metrics": latest_metrics,
                    "cluster_status": {
                        "total_nodes": cluster_status.total_nodes,
                        "active_nodes": cluster_status.active_nodes,
                        "total_gpus": cluster_status.total_gpus,
                        "avg_gpu_utilization": cluster_status.avg_gpu_utilization,
                        "avg_memory_usage": cluster_status.avg_memory_usage,
                        "total_throughput": cluster_status.total_throughput
                    }
                }
            })

            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming"""
    await manager.connect(websocket)

    try:
        while True:
            # Simulate log streaming
            await websocket.send_json({
                "type": "log",
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": "Training iteration completed"
            })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

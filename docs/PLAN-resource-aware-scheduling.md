# PLAN: Resource-Aware Scheduling & Fault Tolerance

> **Goal**: Implement a robust scheduling system that optimizes initial node placement, handles dynamic node failures (Elasticity/Fault Tolerance), and proactively mitigates stragglers (Preemption) using existing resource metrics.

## 1. Architecture Enhancements

### 1.1 `NodeHealthManager`
A new component responsible for tracking the lifecycle of worker nodes.
- **States**: `ACTIVE`, `SUSPECT` (straggler), `DEAD` (no heartbeat), `DRAINING` (will be removed at next safe checkpoint).
- **Mechanism**: Heartbeat monitoring via Redis.
- **Action**: Triggers `ClusterReconfiguration` (Stop -> Checkpoint -> Restart).

### 1.2 `JobScheduler`
Orchestrator that sits above the `DistributedTrainer`.
- **Responsibilities**:
  - Initial Placement (Option A).
  - Monitoring Health Manager.
  - Triggering Checkpoint/Restore on failure (Option B).
  - Deciding when to preempt a slow node (Option C).

---

## 2. Implementation Phases

### Phase 1: Intelligent Placement (Option A)
**Goal**: Optimize initial worker selection and batch size distribution.

- [ ] **Extend `AdaptiveLoadBalancer`**:
    - Add `score_placement(nodes, model_requirements)`: Returns a fitness score.
    - Logic: Ensure `node.memory > model.memory_required`. Prefer nodes with highest `compute_score`.
- [ ] **Modify `main.py`**:
    - Before starting `DistributedTrainer`, filter available nodes.
    - If user requests N nodes but M are available, pick top N.

### Phase 2: Fault Tolerance & Elasticity (Option B)
**Goal**: Survive node crashes without losing significant progress.

- [ ] **Heartbeat Mechanism**:
    - **Worker**: Background thread updating `worker:{rank}:heartbeat` in Redis every 5s.
    - **Master**: Background thread checking all heartbeats.
- [ ] **Elastic Checkpointing**:
    - **Trigger**: Periodic (every epoch) + Event-based (before reconfiguration).
    - **Storage**: Shared filesystem (already in `experiments/checkpoints`).
- [ ] **Recovery Routine (Stop & Restart)**:
    - **Mechanism**: Coordinated Stop -> Save Checkpoint -> Restart with N-1 Nodes.
    - **Note**: Elasticity here means "Resumable Training", not "Live Rank Changes".
    - **Coordinator**: Assumed stable for this project (external orchestration not required).

### Phase 3: Performance Preemption (Option C)
**Goal**: Proactively remove nodes that are degrading system performance (Stragglers).

- [ ] **Straggler Detection**:
    - Monitor `iteration_time` per rank in `AdaptiveLoadBalancer`.
    - Rule: If `node.iter_time > 1.5 * median_iter_time` (using median prevents outlier noise).
- [ ] **Preemption Decision**:
    - If `SUSPECT` continues to lag -> Mark `DRAINING`.
    - Trigger **Elastic Restart** (Phase 2) to exclude this node.
    - **Dataset**: `DistributedSampler` automatically handles re-sharding on restart.

---

## 3. Agents & file Assignments

| Component | File Path | Responsibility |
|-----------|-----------|----------------|
| **Scheduler** | `src/scheduling/scheduler.py` | [NEW] Manages job lifecycle and placement decisions. |
| **Health** | `src/scheduling/health_monitor.py` | [NEW] Redis-based heartbeat checks. |
| **Balancer** | `src/scheduling/load_balancer.py` | Update with placement scoring logic. |
| **Training** | `src/training/distributed_trainer.py` | Add support for elastic restart (re-init DDP). |
| **Main** | `src/training/main.py` | Integrate Scheduler loop. |

## 4. Verification Plan

### 4.1 Automated Tests
- [ ] **Placement Test**: Mock 3 nodes (2 Strong, 1 Weak), verify strong ones picked for small jobs.
- [ ] **Fault Test**:
    - Start training with 3 processes.
    - `kill -9` one worker process.
    - Verify Master detects `DEAD` state (via Redis heartbeat timeout).
    - Verify remaining 2 workers save checkpoint and "restart" (mocked re-init).
- [ ] **Preemption Test**:
    - Inject artificial delay (`time.sleep`) in one worker.
    - Verify Scheduler detects high `iteration_time`.
    - Verify Scheduler removes that worker and continues with N-1 workers.

### 4.2 Manual Verification
- Run `multinode_training.py` (simulated locally).
- Kill a terminal window.
- Watch logs for "Recovering from failure...".

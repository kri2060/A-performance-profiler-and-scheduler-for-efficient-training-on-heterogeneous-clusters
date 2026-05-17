"""
Node Health Monitor
Handles heartbeat mechanism and failure detection via Redis
"""

import redis
import time
import threading
import logging
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)

class NodeHealthManager:
    """
    Manages node health via Redis heartbeats
    """
    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        heartbeat_interval: int = 5,
        timeout_threshold: int = 15
    ):
        """
        Args:
            redis_host: Redis host
            redis_port: Redis port
            heartbeat_interval: Seconds between heartbeats
            timeout_threshold: Seconds before declaring a node DEAD
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.heartbeat_interval = heartbeat_interval
        self.timeout_threshold = timeout_threshold
        
        self.redis_client = None
        self.running = False
        self.rank = -1
        self._lock = threading.Lock()
        
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_connect_timeout=3
            )
            self.redis_client.ping()
            logger.info(f"NodeHealthManager connected to Redis at {redis_host}")
        except Exception as e:
            logger.warning(f"NodeHealthManager failed to connect to Redis: {e}")
            self.redis_client = None

    def start_heartbeat(self, rank: int):
        """Start background heartbeat thread for this worker"""
        if not self.redis_client:
            return

        self.rank = rank
        self.running = True
        
        # Start thread
        self.thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started heartbeat for Rank {rank}")

    def stop(self):
        """Stop heartbeat thread"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
            
    def _heartbeat_loop(self):
        """Internal loop to send heartbeats"""
        while self.running:
            try:
                timestamp = int(time.time())
                key = f"worker:{self.rank}:heartbeat"
                self.redis_client.set(key, timestamp, ex=self.timeout_threshold * 2)
            except Exception as e:
                logger.debug(f"Heartbeat failed: {e}")
                
            time.sleep(self.heartbeat_interval)

    def check_cluster_health(self, expected_ranks: List[int]) -> List[int]:
        """
        Check health of all expected nodes.
        Returns list of DEAD ranks.
        To be called by Master (Rank 0).
        """
        if not self.redis_client:
            return []

        dead_nodes = []
        current_time = int(time.time())

        for rank in expected_ranks:
            if rank == self.rank:
                continue # Don't check self
                
            key = f"worker:{rank}:heartbeat"
            last_seen = self.redis_client.get(key)
            
            if last_seen:
                age = current_time - int(last_seen)
                if age > self.timeout_threshold:
                    logger.warning(f"Node {rank} is timeout (seen {age}s ago)")
                    dead_nodes.append(rank)
            else:
                # Never seen or expired
                # Only mark dead if we expect it to be alive (logic handled by caller usually)
                # For now, if key missing, assume dead/not started
                # logger.warning(f"Node {rank} heartbeat missing") 
                # Not marking dead immediately on missing key to avoid startup race, 
                # but in strict mode we should.
                pass
                
        return dead_nodes


# ---------------------------------------------------------------------------
# DBS: Straggler simulation (from fault_tolerance_wait())
# ---------------------------------------------------------------------------

def simulate_straggler(
    rank: int,
    epoch: int,
    batch_num: int,
    state: dict,
    fault_chance: float = 0.1,
) -> float:
    """Simulate a slow/faulty worker to test adaptive balancing behaviour.

    Ported from DBS ``fault_tolerance_wait()`` with all globals removed.
    The caller owns *state* and passes it on every call so the function
    stays stateless and fully testable.

    Args:
        rank:         Worker rank (used for logging only).
        epoch:        Current epoch number.
        batch_num:    Total batches in this epoch (to split wait evenly).
        state:        Mutable dict carrying fault state across calls::

                          {
                            "fault_wait":      bool,   # currently waiting?
                            "fault_round":     int,    # stop-waiting epoch
                            "fault_wait_time": float,  # seconds to add per epoch
                            "saved_epoch":     int,    # last epoch seen
                          }

        fault_chance: Probability [0, 1] that a worker begins a slow phase.

    Returns:
        Seconds slept in this call (0.0 if no fault active).
    """
    import random
    import time as _time

    # --- still in a waiting phase ---
    if state.get("fault_wait", False):
        if epoch <= state["fault_round"]:
            sleep_s = float(state["fault_wait_time"]) / max(1, batch_num)
            _time.sleep(sleep_s)
            return sleep_s
        else:
            state["fault_wait"] = False
            return 0.0

    if fault_chance <= 0.0:
        return 0.0

    # --- only enter the lottery once per epoch ---
    if state.get("saved_epoch") == epoch:
        return 0.0
    state["saved_epoch"] = epoch

    luck = random.random()
    logger.debug(f"simulate_straggler rank={rank} luck={luck:.4f} limit={fault_chance}")

    if luck < fault_chance:
        wait_time = random.uniform(5, 10)
        wait_until = epoch + random.randint(4, 20)
        state.update(
            fault_wait=True,
            fault_round=wait_until,
            fault_wait_time=wait_time,
        )
        logger.info(
            f"Rank {rank}: straggler active — +{wait_time:.1f}s/epoch "
            f"until epoch {wait_until}"
        )
        # Apply first instalment immediately
        sleep_s = wait_time / max(1, batch_num)
        _time.sleep(sleep_s)
        return sleep_s

    return 0.0

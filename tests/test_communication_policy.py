
from src.scheduling.load_balancer import AdaptiveLoadBalancer, NodeCapability

def test_communication_policy():
    lb = AdaptiveLoadBalancer()
    
    # Scene 1: Fast Network -> No Compression, No Accumulation
    nodes_fast = [
        NodeCapability(rank=0, device_id=0, compute_score=10, memory_mb=8000, bandwidth_gbps=300, network_mbps=10000),
        NodeCapability(rank=1, device_id=1, compute_score=10, memory_mb=8000, bandwidth_gbps=300, network_mbps=10000),
    ]
    lb.nodes = nodes_fast
    comp, acc = lb.calculate_communication_config()
    assert comp == 'none'
    assert acc == 1
    
    # Scene 2: Slow Network (< 1000 Mbps) -> FP16, Acc=2
    nodes_slow = [
        NodeCapability(rank=0, device_id=0, compute_score=10, memory_mb=8000, bandwidth_gbps=300, network_mbps=500),
        NodeCapability(rank=1, device_id=1, compute_score=10, memory_mb=8000, bandwidth_gbps=300, network_mbps=500),
    ]
    lb.nodes = nodes_slow
    comp, acc = lb.calculate_communication_config()
    assert comp == 'fp16'
    assert acc == 2
    
    # Scene 3: Very Slow Network (< 100 Mbps) -> FP16, Acc=4
    nodes_vslow = [
        NodeCapability(rank=0, device_id=0, compute_score=10, memory_mb=8000, bandwidth_gbps=300, network_mbps=50),
    ]
    lb.nodes = nodes_vslow
    comp, acc = lb.calculate_communication_config()
    assert comp == 'fp16'
    assert acc == 4
    
    # Scene 4: Stragglers -> Increase Accumulation
    nodes_straggler = [
        NodeCapability(rank=0, device_id=0, compute_score=10, memory_mb=8000, bandwidth_gbps=300, network_mbps=10000),
        NodeCapability(rank=1, device_id=1, compute_score=10, memory_mb=8000, bandwidth_gbps=300, network_mbps=10000, is_straggler=True),
        NodeCapability(rank=2, device_id=2, compute_score=10, memory_mb=8000, bandwidth_gbps=300, network_mbps=10000, is_straggler=True),
    ] 
    # 2 out of 3 are stragglers (> 30%)
    lb.nodes = nodes_straggler
    comp, acc = lb.calculate_communication_config()
    # Should increase accumulation to at least 2 despite fast network
    assert acc >= 2

if __name__ == "__main__":
    test_communication_policy()
    print("All tests passed!")

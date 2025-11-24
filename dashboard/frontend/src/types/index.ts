export interface GPUProfile {
  device_id: number;
  name: string;
  compute_capability: string;
  total_memory_mb: number;
  memory_bandwidth_gbps: number;
  cuda_cores: number | null;
  sm_count: number;
  clock_rate_mhz: number;
  memory_clock_rate_mhz: number;
  pcie_link_gen: number;
  pcie_link_width: number;
  compute_score: number;
}

export interface SystemProfile {
  hostname: string;
  platform: string;
  cpu_model: string;
  cpu_cores_physical: number;
  cpu_cores_logical: number;
  cpu_frequency_mhz: number;
  ram_total_gb: number;
  ram_available_gb: number;
  network_interfaces: string[];
  ip_address: string;
}

export interface TrainingMetric {
  timestamp?: number;
  epoch: number;
  iteration: number;
  loss: number;
  throughput: number;
  iteration_time: number;
  data_loading_time: number;
  forward_time: number;
  backward_time: number;
  optimizer_time: number;
  gpu_utilization: number;
  gpu_memory_percent: number;
}

export interface ClusterStatus {
  total_nodes: number;
  active_nodes: number;
  total_gpus: number;
  avg_gpu_utilization: number;
  avg_memory_usage: number;
  total_throughput: number;
}

export interface Job {
  id: string;
  name: string;
  model: string;
  dataset: string;
  status: 'running' | 'completed' | 'failed' | 'pending';
  start_time: string;
  epochs_completed: number;
  total_epochs: number;
  current_loss: number;
  throughput: number;
  nodes: number;
}

export interface Experiment {
  name: string;
  path: string;
  ranks: number;
  created: string;
}

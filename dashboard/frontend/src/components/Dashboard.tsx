'use client'

import { useQuery } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { ClusterStatus, GPUProfile, TrainingMetric } from '@/types'
import {
  Cpu,
  MemoryStick,
  Activity,
  Zap,
  Server,
  TrendingUp,
  Clock,
  Layers
} from 'lucide-react'
import { GPUChart } from './charts/GPUChart'
import { LossChart } from './charts/LossChart'
import { ThroughputChart } from './charts/ThroughputChart'
import { TimeBreakdownChart } from './charts/TimeBreakdownChart'

const API_BASE = 'http://localhost:8000'

export function Dashboard() {
  // Fetch cluster status
  const { data: clusterStatus } = useQuery<ClusterStatus>({
    queryKey: ['clusterStatus'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/cluster/status`)
      return res.json()
    },
  })

  // Fetch GPU profiles
  const { data: gpuProfiles } = useQuery<GPUProfile[]>({
    queryKey: ['gpuProfiles'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/nodes`)
      return res.json()
    },
  })

  // Fetch experiments
  const { data: experiments } = useQuery({
    queryKey: ['experiments'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/experiments`)
      return res.json()
    },
  })

  // Fetch latest metrics for first experiment
  const { data: metricsData } = useQuery({
    queryKey: ['metrics', experiments?.[0]?.name],
    queryFn: async () => {
      if (!experiments?.[0]) return null
      const res = await fetch(
        `${API_BASE}/api/metrics/history/${experiments[0].name}?limit=200`
      )
      return res.json()
    },
    enabled: !!experiments?.[0],
  })

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Server className="h-8 w-8 text-blue-600" />
              <h1 className="text-2xl font-bold text-gray-900">
                Heterogeneous Cluster Monitor
              </h1>
            </div>
            <div className="flex items-center space-x-2">
              <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
                <span className="w-2 h-2 mr-2 bg-green-500 rounded-full animate-pulse"></span>
                Connected
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total GPUs</CardTitle>
              <Cpu className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {clusterStatus?.total_gpus || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                {clusterStatus?.active_nodes || 0} active nodes
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Avg GPU Utilization
              </CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {clusterStatus?.avg_gpu_utilization?.toFixed(1) || 0}%
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div
                  className="bg-blue-600 h-2 rounded-full"
                  style={{
                    width: `${clusterStatus?.avg_gpu_utilization || 0}%`,
                  }}
                ></div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
              <MemoryStick className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {clusterStatus?.avg_memory_usage?.toFixed(1) || 0}%
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div
                  className="bg-green-600 h-2 rounded-full"
                  style={{
                    width: `${clusterStatus?.avg_memory_usage || 0}%`,
                  }}
                ></div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Throughput</CardTitle>
              <Zap className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {clusterStatus?.total_throughput?.toFixed(1) || 0}
              </div>
              <p className="text-xs text-muted-foreground">samples/sec</p>
            </CardContent>
          </Card>
        </div>

        {/* GPU Profiles */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Layers className="h-5 w-5 mr-2" />
              GPU Hardware Profiles
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {gpuProfiles?.map((gpu) => (
                <div
                  key={gpu.device_id}
                  className="p-4 border rounded-lg bg-white hover:shadow-md transition-shadow"
                >
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold text-lg">GPU {gpu.device_id}</h3>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      Score: {gpu.compute_score}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mb-2">{gpu.name}</p>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-500">Memory:</span>
                      <span className="font-medium">
                        {(gpu.total_memory_mb / 1024).toFixed(1)} GB
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Compute Capability:</span>
                      <span className="font-medium">
                        {gpu.compute_capability}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">SM Count:</span>
                      <span className="font-medium">{gpu.sm_count}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Clock Rate:</span>
                      <span className="font-medium">
                        {(gpu.clock_rate_mhz / 1000).toFixed(2)} GHz
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <TrendingUp className="h-5 w-5 mr-2" />
                Training Loss
              </CardTitle>
            </CardHeader>
            <CardContent>
              <LossChart data={metricsData} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Zap className="h-5 w-5 mr-2" />
                Training Throughput
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ThroughputChart data={metricsData} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Activity className="h-5 w-5 mr-2" />
                GPU Utilization
              </CardTitle>
            </CardHeader>
            <CardContent>
              <GPUChart data={metricsData} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Clock className="h-5 w-5 mr-2" />
                Time Breakdown
              </CardTitle>
            </CardHeader>
            <CardContent>
              <TimeBreakdownChart data={metricsData} />
            </CardContent>
          </Card>
        </div>

        {/* Experiments List */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Experiments</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-3 px-4 font-medium text-gray-600">
                      Name
                    </th>
                    <th className="text-left py-3 px-4 font-medium text-gray-600">
                      Ranks
                    </th>
                    <th className="text-left py-3 px-4 font-medium text-gray-600">
                      Created
                    </th>
                    <th className="text-left py-3 px-4 font-medium text-gray-600">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {experiments?.map((exp: any) => (
                    <tr key={exp.name} className="border-b hover:bg-gray-50">
                      <td className="py-3 px-4 font-medium">{exp.name}</td>
                      <td className="py-3 px-4">{exp.ranks}</td>
                      <td className="py-3 px-4">
                        {new Date(exp.created).toLocaleDateString()}
                      </td>
                      <td className="py-3 px-4">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                          Completed
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  )
}

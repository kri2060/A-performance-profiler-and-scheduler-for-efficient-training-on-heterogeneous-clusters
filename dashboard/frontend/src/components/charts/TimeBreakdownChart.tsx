'use client'

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'

interface TimeBreakdownChartProps {
  data: Record<string, any[]> | null | undefined
}

export function TimeBreakdownChart({ data }: TimeBreakdownChartProps) {
  if (!data) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500">
        No data available
      </div>
    )
  }

  // Calculate average times for each rank
  const chartData: any[] = []
  const ranks = Object.keys(data)

  if (ranks.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500">
        No metrics available
      </div>
    )
  }

  ranks.forEach((rank) => {
    const metrics = data[rank]
    const avgDataLoading =
      metrics.reduce((sum, m) => sum + m.data_loading_time, 0) / metrics.length
    const avgForward =
      metrics.reduce((sum, m) => sum + m.forward_time, 0) / metrics.length
    const avgBackward =
      metrics.reduce((sum, m) => sum + m.backward_time, 0) / metrics.length
    const avgOptimizer =
      metrics.reduce((sum, m) => sum + m.optimizer_time, 0) / metrics.length

    chartData.push({
      name: rank.replace('_', ' ').toUpperCase(),
      'Data Loading': avgDataLoading * 1000,
      'Forward Pass': avgForward * 1000,
      'Backward Pass': avgBackward * 1000,
      'Optimizer': avgOptimizer * 1000,
    })
  })

  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} layout="horizontal">
          <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
          <XAxis dataKey="name" stroke="#6B7280" fontSize={12} />
          <YAxis
            stroke="#6B7280"
            fontSize={12}
            tickFormatter={(value) => `${value.toFixed(0)}ms`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #E5E7EB',
              borderRadius: '8px',
            }}
            formatter={(value: number) => [`${value.toFixed(2)} ms`, '']}
          />
          <Legend />
          <Bar
            dataKey="Data Loading"
            stackId="a"
            fill="#F59E0B"
            radius={[0, 0, 0, 0]}
          />
          <Bar
            dataKey="Forward Pass"
            stackId="a"
            fill="#3B82F6"
            radius={[0, 0, 0, 0]}
          />
          <Bar
            dataKey="Backward Pass"
            stackId="a"
            fill="#10B981"
            radius={[0, 0, 0, 0]}
          />
          <Bar
            dataKey="Optimizer"
            stackId="a"
            fill="#8B5CF6"
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

'use client'

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'

interface GPUChartProps {
  data: Record<string, any[]> | null | undefined
}

export function GPUChart({ data }: GPUChartProps) {
  if (!data) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500">
        No data available
      </div>
    )
  }

  // Transform data for Recharts
  const chartData: any[] = []
  const ranks = Object.keys(data)

  if (ranks.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500">
        No metrics available
      </div>
    )
  }

  // Use first rank's data as base
  const baseData = data[ranks[0]]
  baseData.forEach((metric, index) => {
    const point: any = {
      iteration: metric.iteration,
    }
    ranks.forEach((rank) => {
      if (data[rank][index]) {
        point[`${rank}_util`] = data[rank][index].gpu_utilization
        point[`${rank}_mem`] = data[rank][index].gpu_memory_percent
      }
    })
    chartData.push(point)
  })

  const colors = {
    util: '#3B82F6',
    mem: '#10B981',
  }

  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
          <XAxis
            dataKey="iteration"
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
          />
          <YAxis
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
            domain={[0, 100]}
            tickFormatter={(value) => `${value}%`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #E5E7EB',
              borderRadius: '8px',
            }}
            formatter={(value: number) => [`${value.toFixed(1)}%`, '']}
          />
          <Legend />
          {ranks.map((rank) => (
            <>
              <Line
                key={`${rank}_util`}
                type="monotone"
                dataKey={`${rank}_util`}
                name={`${rank.replace('_', ' ').toUpperCase()} Util`}
                stroke={colors.util}
                strokeWidth={2}
                dot={false}
              />
              <Line
                key={`${rank}_mem`}
                type="monotone"
                dataKey={`${rank}_mem`}
                name={`${rank.replace('_', ' ').toUpperCase()} Mem`}
                stroke={colors.mem}
                strokeWidth={2}
                dot={false}
                strokeDasharray="5 5"
              />
            </>
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

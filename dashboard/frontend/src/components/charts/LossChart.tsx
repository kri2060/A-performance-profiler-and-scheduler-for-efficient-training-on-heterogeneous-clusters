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

interface LossChartProps {
  data: Record<string, any[]> | null | undefined
}

export function LossChart({ data }: LossChartProps) {
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
        point[rank] = data[rank][index].loss
      }
    })
    chartData.push(point)
  })

  const colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']

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
          <YAxis stroke="#6B7280" fontSize={12} tickLine={false} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #E5E7EB',
              borderRadius: '8px',
            }}
          />
          <Legend />
          {ranks.map((rank, index) => (
            <Line
              key={rank}
              type="monotone"
              dataKey={rank}
              name={rank.replace('_', ' ').toUpperCase()}
              stroke={colors[index % colors.length]}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

import type React from "react"

export const BarChart = () => {
  return <div>BarChart Component</div>
}

export const LineChart = () => {
  return <div>LineChart Component</div>
}

export const PieChart = () => {
  return <div>PieChart Component</div>
}

export const ChartContainer = ({ children }: { children: React.ReactNode }) => {
  return <div className="chart-container">{children}</div>
}

export const ChartTooltip = () => {
  return <div>ChartTooltip Component</div>
}

export const ChartLegend = () => {
  return <div>ChartLegend Component</div>
}

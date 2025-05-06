"use client";

import { ChartContainer } from "@/components/ui/chart";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface CostStatsChartProps {
  data: Array<{
    name: string;
    labor: number;
    supplies: number;
    equipment: number;
  }>;
}

export function CostStatsChart({ data }: CostStatsChartProps) {
  return (
    <div className="h-[300px]">
      <ChartContainer>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="labor" 
              stroke="hsl(var(--chart-1))" 
              name="Labor" 
            />
            <Line 
              type="monotone" 
              dataKey="supplies" 
              stroke="hsl(var(--chart-2))" 
              name="Supplies" 
            />
            <Line 
              type="monotone" 
              dataKey="equipment" 
              stroke="hsl(var(--chart-3))" 
              name="Equipment" 
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartContainer>
    </div>
  );
}

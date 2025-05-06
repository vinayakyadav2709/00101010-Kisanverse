"use client";

import { ChartContainer, ChartTooltip, PieChart } from "@/components/ui/chart";
import { Cell, Legend, Pie, ResponsiveContainer } from "recharts";

interface ResourcesChartProps {
  data: Array<{
    name: string;
    value: number;
    color: string;
  }>;
}

export function ResourcesChart({ data }: ResourcesChartProps) {
  return (
    <div className="h-[300px]">
      <ChartContainer>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={90}
              paddingAngle={2}
              dataKey="value"
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <ChartTooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </ChartContainer>
    </div>
  );
}

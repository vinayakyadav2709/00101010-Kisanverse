"use client";

import { BarChart, ChartContainer } from "@/components/ui/chart";
import { Bar, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface UserStatsChartProps {
  data: Array<{
    name: string;
    active: number;
    new: number;
    churned: number;
  }>;
}

export function UserStatsChart({ data }: UserStatsChartProps) {
  return (
    <div className="h-[300px]">
      <ChartContainer>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="active" fill="#4ade80" name="Active Users" />
            <Bar dataKey="new" fill="#60a5fa" name="New Users" />
            <Bar dataKey="churned" fill="#f87171" name="Churned Users" />
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>
    </div>
  );
}

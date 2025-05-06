"use client"
import { Header } from "@/components/header"
import { DashboardShell } from "@/components/dashboard-shell"
import { DashboardCards } from "@/components/dashboard-cards"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { 
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip as ChartTooltip,
  Legend as ChartLegend,
  Filler
} from 'chart.js'
import { Bar, Line, Pie } from 'react-chartjs-2'
import { WorldMap } from "@/components/client/world-map"

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  ChartTooltip,
  ChartLegend,
  Filler
)

export default function DashboardPage() {
  // Sample data for user stats
  const userStatsData = [
    { name: "Jan", active: 400, new: 240, churned: 20 },
    { name: "Feb", active: 430, new: 138, churned: 28 },
    { name: "Mar", active: 448, new: 98, churned: 18 },
    { name: "Apr", active: 470, new: 108, churned: 26 },
    { name: "May", active: 540, new: 120, churned: 32 },
    { name: "Jun", active: 580, new: 150, churned: 40 },
  ]

  // Sample data for resources used stats
  const resourcesUsedData = [
    { name: "Water", value: 34500, color: "#4ade80" },
    { name: "Fertilizer", value: 25700, color: "#60a5fa" },
    { name: "Pesticides", value: 15500, color: "#f87171" },
    { name: "Electricity", value: 28900, color: "#fbbf24" },
  ]

  // Sample data for cost stats
  const costStatsData = [
    { name: "Jan", labor: 4000, supplies: 2400, equipment: 2000 },
    { name: "Feb", labor: 3000, supplies: 1398, equipment: 2210 },
    { name: "Mar", labor: 2000, supplies: 9800, equipment: 2290 },
    { name: "Apr", labor: 2780, supplies: 3908, equipment: 2000 },
    { name: "May", labor: 1890, supplies: 4800, equipment: 2181 },
    { name: "Jun", labor: 2390, supplies: 3800, equipment: 2500 },
  ]

  // Sample data for world map
  const mapMarkers = [
    { name: "New York", coordinates: [-74.006, 40.7128] as [number, number], users: 1245 },
    { name: "London", coordinates: [-0.1278, 51.5074] as [number, number], users: 986 },
    { name: "Paris", coordinates: [2.3522, 48.8566] as [number, number], users: 754 },
    { name: "Tokyo", coordinates: [139.6917, 35.6895] as [number, number], users: 1102 },
    { name: "Sydney", coordinates: [151.2093, -33.8688] as [number, number], users: 689 },
    { name: "Mumbai", coordinates: [72.8777, 19.076] as [number, number], users: 842 },
    { name: "São Paulo", coordinates: [-46.6333, -23.5505] as [number, number], users: 567 },
    { name: "Cairo", coordinates: [31.2357, 30.0444] as [number, number], users: 423 },
  ]

  // Prepare data for Chart.js
  const labels = userStatsData.map(item => item.name);
  
  // User Stats Chart data
  const userStatsChartData = {
    labels,
    datasets: [
      {
        fill: true,
        label: 'Active Users',
        data: userStatsData.map(item => item.active),
        backgroundColor: 'rgba(136, 132, 216, 0.6)',
        borderColor: 'rgb(136, 132, 216)',
      },
      {
        fill: true,
        label: 'New Users',
        data: userStatsData.map(item => item.new),
        backgroundColor: 'rgba(130, 202, 157, 0.6)',
        borderColor: 'rgb(130, 202, 157)',
      },
      {
        fill: true,
        label: 'Churned Users',
        data: userStatsData.map(item => item.churned),
        backgroundColor: 'rgba(255, 198, 88, 0.6)',
        borderColor: 'rgb(255, 198, 88)',
      }
    ],
  };
  
  // Resources Used Chart data
  const resourcesChartData = {
    labels: resourcesUsedData.map(item => item.name),
    datasets: [
      {
        data: resourcesUsedData.map(item => item.value),
        backgroundColor: resourcesUsedData.map(item => item.color),
        borderWidth: 1,
      },
    ],
  };
  
  // Cost Stats Chart data
  const costStatsChartData = {
    labels,
    datasets: [
      {
        label: 'Labor',
        data: costStatsData.map(item => item.labor),
        backgroundColor: 'rgba(136, 132, 216, 0.7)',
      },
      {
        label: 'Supplies',
        data: costStatsData.map(item => item.supplies),
        backgroundColor: 'rgba(130, 202, 157, 0.7)',
      },
      {
        label: 'Equipment',
        data: costStatsData.map(item => item.equipment),
        backgroundColor: 'rgba(255, 198, 88, 0.7)',
      }
    ],
  };
  
  // Chart options
  const lineOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      }
    }
  };
  
  const pieOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            const label = context.label || '';
            const value = context.parsed || 0;
            const total = context.dataset.data.reduce((a: number, b: number) => a + b, 0);
            const percentage = Math.round((value / total) * 100);
            return `${label}: ${value} (${percentage}%)`;
          }
        }
      }
    }
  };
  
  const barOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
      x: {
        stacked: true,
      }
    }
  };

  function cn(...classes: (string | undefined | null | false)[]): string {
    return classes.filter(Boolean).join(' ');
  }
  return (
    <DashboardShell>
      <Header />
      <div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
        <DashboardCards />
        {/* Weather component removed */}
      </div>
      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-6">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-2">
          {/* Stats for Users */}
          <Card>
            <CardHeader>
              <CardTitle>Stats for Users</CardTitle>
            </CardHeader>
            <CardContent>
              <div style={{ height: '300px', width: '100%' }}>
                <Line options={lineOptions} data={userStatsChartData} />
              </div>
            </CardContent>
          </Card>

          {/* Resources Used Stats */}
          <Card>
            <CardHeader>
              <CardTitle>Resources Used Stats</CardTitle>
            </CardHeader>
            <CardContent>
              <div style={{ height: '300px', width: '100%' }}>
                <Pie options={pieOptions} data={resourcesChartData} />
              </div>
            </CardContent>
          </Card>

          {/* Cost Stats */}
          <Card>
            <CardHeader>
              <CardTitle>Cost Stats</CardTitle>
            </CardHeader>
            <CardContent>
              <div style={{ height: '300px', width: '100%' }}>
                <Bar options={barOptions} data={costStatsChartData} />
              </div>
            </CardContent>
          </Card>

          {/* World Wide Stats (Map) */}
          <Card>
            <CardHeader>
              <CardTitle>World Wide Stats (Map)</CardTitle>
            </CardHeader>
            <CardContent>
              <WorldMap markers={mapMarkers} />
            </CardContent>
          </Card>
        </div>
      </main>
    </DashboardShell>
  )
}

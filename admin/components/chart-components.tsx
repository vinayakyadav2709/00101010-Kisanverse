"use client"
import { 
  BarChart, LineChart, PieChart, 
  Bar, Line, Pie, Cell, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from "recharts"
import { ComposableMap, Geographies, Geography, Marker } from "react-simple-maps"
import { ChartContainer, ChartTooltip } from "@/components/ui/chart"

// User Stats Bar Chart
const UserStats = ({ data }: { data: any[] }) => (
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
)

// Resources Used Pie Chart
const ResourcesUsed = ({ data }: { data: any[] }) => (
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
          label={({ name, percent }: any) => `${name} ${(percent * 100).toFixed(0)}%`}
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  </ChartContainer>
)

// Cost Stats Line Chart
const CostStats = ({ data }: { data: any[] }) => (
  <ChartContainer>
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="labor" stroke="#4ade80" name="Labor" />
        <Line type="monotone" dataKey="supplies" stroke="#60a5fa" name="Supplies" />
        <Line type="monotone" dataKey="equipment" stroke="#f87171" name="Equipment" />
      </LineChart>
    </ResponsiveContainer>
  </ChartContainer>
)

// World Map
const WorldMap = ({ markers, geoUrl }: { markers: any[], geoUrl: string }) => (
  <ComposableMap>
    <Geographies geography={geoUrl}>
      {({ geographies }) =>
        geographies.map((geo) => (
          <Geography 
            key={geo.rsmKey} 
            geography={geo} 
            fill="#EAEAEC" 
            stroke="#D6D6DA" 
          />
        ))
      }
    </Geographies>
    {markers.map(({ name, coordinates, users }) => (
      <Marker key={name} coordinates={coordinates}>
        <circle r={Math.sqrt(users) / 10} fill="#4ade80" opacity={0.8} />
        <text
          textAnchor="middle"
          y={-10}
          style={{ fontFamily: "system-ui", fill: "#5D5A6D", fontSize: "8px" }}
        >
          {name}
        </text>
      </Marker>
    ))}
  </ComposableMap>
)

// Export all chart components
const ChartComponents = {
  UserStats,
  ResourcesUsed,
  CostStats,
  WorldMap,
}

export default ChartComponents

import { Header } from "@/components/header"
import { DashboardShell } from "@/components/dashboard-shell"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ComposableMap, Geographies, Geography, Marker } from "react-simple-maps"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

// Sample data for the map
const markers = [
  { name: "New York", coordinates: [-74.006, 40.7128], users: 1245 },
  { name: "London", coordinates: [-0.1278, 51.5074], users: 986 },
  { name: "Paris", coordinates: [2.3522, 48.8566], users: 754 },
  { name: "Tokyo", coordinates: [139.6917, 35.6895], users: 1102 },
  { name: "Sydney", coordinates: [151.2093, -33.8688], users: 689 },
  { name: "Mumbai", coordinates: [72.8777, 19.076], users: 842 },
  { name: "São Paulo", coordinates: [-46.6333, -23.5505], users: 567 },
  { name: "Cairo", coordinates: [31.2357, 30.0444], users: 423 },
]

export default function GlobalStatsPage() {
  return (
    <DashboardShell>
      <Header />
      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold tracking-tight">Global Statistics</h1>
        </div>
        <Tabs defaultValue="map" className="space-y-4">
          <TabsList>
            <TabsTrigger value="map">World Map</TabsTrigger>
            <TabsTrigger value="regions">By Region</TabsTrigger>
          </TabsList>
          <TabsContent value="map" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>World Wide User Distribution</CardTitle>
                <CardDescription>Geographic distribution of users across the globe</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[500px] w-full">
                  <ComposableMap>
                    <Geographies geography="/world-110m.json">
                      {({ geographies }) =>
                        geographies.map((geo) => (
                          <Geography key={geo.rsmKey} geography={geo} fill="#EAEAEC" stroke="#D6D6DA" />
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
                </div>
              </CardContent>
            </Card>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">North America</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">4,235</div>
                  <p className="text-xs text-muted-foreground">34% of total users</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Europe</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">3,890</div>
                  <p className="text-xs text-muted-foreground">31% of total users</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Asia</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">3,120</div>
                  <p className="text-xs text-muted-foreground">25% of total users</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Other Regions</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">1,100</div>
                  <p className="text-xs text-muted-foreground">10% of total users</p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          <TabsContent value="regions" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Top Countries</CardTitle>
                  <CardDescription>Countries with the most users</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {[
                      { country: "United States", users: 3245, percent: 26 },
                      { country: "India", users: 1890, users: 15 },
                      { country: "United Kingdom", users: 1245, percent: 10 },
                      { country: "Germany", users: 980, percent: 8 },
                      { country: "Brazil", users: 865, percent: 7 },
                    ].map((item, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className="font-medium">{item.country}</div>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="text-sm text-muted-foreground">{item.users} users</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Growth by Region</CardTitle>
                  <CardDescription>User growth rate by region</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {[
                      { region: "North America", growth: "+12%" },
                      { region: "Europe", growth: "+8%" },
                      { region: "Asia", growth: "+24%" },
                      { region: "South America", growth: "+18%" },
                      { region: "Africa", growth: "+32%" },
                      { region: "Oceania", growth: "+5%" },
                    ].map((item, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className="font-medium">{item.region}</div>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="text-sm font-medium text-green-600">{item.growth}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </DashboardShell>
  )
}

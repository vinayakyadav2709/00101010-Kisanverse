import { Header } from "@/components/header"
import { DashboardShell } from "@/components/dashboard-shell"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Skeleton } from "@/components/ui/skeleton"

export default function Loading() {
  return (
    <DashboardShell>
      <Header />
      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-6">
        <div className="flex items-center justify-between">
          <Skeleton className="h-8 w-48" />
        </div>

        <div className="flex flex-col gap-4 md:flex-row md:items-center">
          <Skeleton className="h-10 md:w-80" />
          <Skeleton className="h-10 md:w-40" />
          <Skeleton className="h-10 md:w-40" />
        </div>

        <Tabs defaultValue="subsidies" className="space-y-4">
          <TabsList>
            <TabsTrigger value="subsidies">Subsidies</TabsTrigger>
            <TabsTrigger value="requests">Requests</TabsTrigger>
          </TabsList>
          <TabsContent value="subsidies" className="space-y-4">
            {Array(3).fill(0).map((_, i) => (
              <Card key={i}>
                <CardHeader className="flex flex-row items-start justify-between">
                  <div className="space-y-2">
                    <Skeleton className="h-5 w-40" />
                    <Skeleton className="h-4 w-60" />
                  </div>
                  <Skeleton className="h-6 w-24" />
                </CardHeader>
                <CardContent className="space-y-4">
                  <Skeleton className="h-4 w-20" />
                  <div className="flex gap-2">
                    <Skeleton className="h-6 w-20" />
                    <Skeleton className="h-6 w-20" />
                  </div>
                  <Skeleton className="h-4 w-24" />
                  <Skeleton className="h-6 w-12" />
                  <Skeleton className="h-20 w-full" />
                </CardContent>
              </Card>
            ))}
          </TabsContent>
        </Tabs>
      </main>
    </DashboardShell>
  )
}

"use client"

import { useState } from "react"
import { Header } from "@/components/header"
import { DashboardShell } from "@/components/dashboard-shell"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Search, MoreHorizontal, MessageSquare, Settings, AlertTriangle } from "lucide-react"

export default function SearchPage() {
  const [searchMode, setSearchMode] = useState<"admin" | "user">("admin")
  const [searchQuery, setSearchQuery] = useState("")

  // Sample data for admin activities
  const adminActivities = [
    {
      id: 1,
      admin: "John Doe",
      action: "Updated user profile",
      target: "User #12345",
      timestamp: "Today at 10:23 AM",
      avatar: "/placeholder-user.jpg",
    },
    {
      id: 2,
      admin: "Jane Smith",
      action: "Approved marketplace listing",
      target: "Listing #78901",
      timestamp: "Today at 09:15 AM",
      avatar: "/placeholder-user.jpg",
    },
    {
      id: 3,
      admin: "Mike Johnson",
      action: "Removed inappropriate content",
      target: "Post #45678",
      timestamp: "Yesterday at 04:30 PM",
      avatar: "/placeholder-user.jpg",
    },
    {
      id: 4,
      admin: "Sarah Williams",
      action: "Updated system settings",
      target: "Global Settings",
      timestamp: "Yesterday at 02:45 PM",
      avatar: "/placeholder-user.jpg",
    },
    {
      id: 5,
      admin: "David Brown",
      action: "Banned user",
      target: "User #56789",
      timestamp: "Apr 26, 2024 at 11:20 AM",
      avatar: "/placeholder-user.jpg",
    },
  ]

  // Sample data for users
  const users = [
    {
      id: 1,
      name: "Alex Johnson",
      email: "alex.johnson@example.com",
      role: "Farmer",
      status: "Active",
      joinDate: "Jan 15, 2024",
      avatar: "/placeholder-user.jpg",
    },
    {
      id: 2,
      name: "Maria Garcia",
      email: "maria.garcia@example.com",
      role: "Supplier",
      status: "Active",
      joinDate: "Feb 3, 2024",
      avatar: "/placeholder-user.jpg",
    },
    {
      id: 3,
      name: "Robert Chen",
      email: "robert.chen@example.com",
      role: "Farmer",
      status: "Inactive",
      joinDate: "Mar 22, 2024",
      avatar: "/placeholder-user.jpg",
    },
    {
      id: 4,
      name: "Sophia Kim",
      email: "sophia.kim@example.com",
      role: "Distributor",
      status: "Active",
      joinDate: "Dec 10, 2023",
      avatar: "/placeholder-user.jpg",
    },
    {
      id: 5,
      name: "James Wilson",
      email: "james.wilson@example.com",
      role: "Farmer",
      status: "Active",
      joinDate: "Apr 5, 2024",
      avatar: "/placeholder-user.jpg",
    },
  ]

  // Filter data based on search query
  const filteredAdminActivities = adminActivities.filter(
    (activity) =>
      activity.admin.toLowerCase().includes(searchQuery.toLowerCase()) ||
      activity.action.toLowerCase().includes(searchQuery.toLowerCase()) ||
      activity.target.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  const filteredUsers = users.filter(
    (user) =>
      user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      user.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
      user.role.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  return (
    <DashboardShell>
      <Header />
      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold tracking-tight">Search</h1>
        </div>
        <Tabs
          defaultValue={searchMode}
          onValueChange={(value) => setSearchMode(value as "admin" | "user")}
          className="space-y-4"
        >
          <TabsList>
            <TabsTrigger value="admin">Admin Search</TabsTrigger>
            <TabsTrigger value="user">User Search</TabsTrigger>
          </TabsList>
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              type="search"
              placeholder={searchMode === "admin" ? "Search admin activities..." : "Search users..."}
              className="w-full pl-8"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <TabsContent value="admin" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Admin Activity</CardTitle>
                <CardDescription>Recent actions performed by administrators</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {filteredAdminActivities.length > 0 ? (
                    filteredAdminActivities.map((activity) => (
                      <div key={activity.id} className="flex items-start justify-between space-x-4">
                        <div className="flex items-start space-x-4">
                          <Avatar>
                            <AvatarImage src={activity.avatar || "/placeholder.svg"} alt={activity.admin} />
                            <AvatarFallback>{activity.admin.substring(0, 2)}</AvatarFallback>
                          </Avatar>
                          <div>
                            <p className="text-sm font-medium leading-none">{activity.admin}</p>
                            <p className="text-sm text-muted-foreground">
                              {activity.action} - {activity.target}
                            </p>
                            <p className="text-xs text-muted-foreground">{activity.timestamp}</p>
                          </div>
                        </div>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon">
                              <MoreHorizontal className="h-4 w-4" />
                              <span className="sr-only">More options</span>
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem>View details</DropdownMenuItem>
                            <DropdownMenuItem>Revert action</DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    ))
                  ) : (
                    <div className="flex h-[200px] items-center justify-center">
                      <p className="text-muted-foreground">No activities found</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="user" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>User Search Results</CardTitle>
                <CardDescription>Users matching your search criteria</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {filteredUsers.length > 0 ? (
                    filteredUsers.map((user) => (
                      <div key={user.id} className="flex items-start justify-between space-x-4">
                        <div className="flex items-start space-x-4">
                          <Avatar>
                            <AvatarImage src={user.avatar || "/placeholder.svg"} alt={user.name} />
                            <AvatarFallback>{user.name.substring(0, 2)}</AvatarFallback>
                          </Avatar>
                          <div>
                            <p className="text-sm font-medium leading-none">{user.name}</p>
                            <p className="text-sm text-muted-foreground">{user.email}</p>
                            <div className="mt-1 flex items-center gap-2">
                              <span className="text-xs">{user.role}</span>
                              <span
                                className={`text-xs ${user.status === "Active" ? "text-green-600" : "text-red-600"}`}
                              >
                                • {user.status}
                              </span>
                              <span className="text-xs text-muted-foreground">Joined: {user.joinDate}</span>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button variant="ghost" size="icon">
                            <MessageSquare className="h-4 w-4" />
                            <span className="sr-only">Message</span>
                          </Button>
                          <Button variant="ghost" size="icon">
                            <Settings className="h-4 w-4" />
                            <span className="sr-only">Settings</span>
                          </Button>
                          <Button variant="ghost" size="icon">
                            <AlertTriangle className="h-4 w-4" />
                            <span className="sr-only">Report</span>
                          </Button>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="flex h-[200px] items-center justify-center">
                      <p className="text-muted-foreground">No users found</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </DashboardShell>
  )
}

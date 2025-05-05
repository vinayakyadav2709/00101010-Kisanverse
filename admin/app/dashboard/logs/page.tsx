"use client"
import React from "react"
import { DashboardShell } from "@/components/dashboard-shell"
import { Header } from "@/components/header"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Info, AlertTriangle, XCircle, ChevronDown, ChevronUp, FileSearch } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { useState } from "react"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { Badge } from "@/components/ui/badge"
import Link from "next/link"
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

type LogType = "SYSTEM" | "CALL"

type BaseLog = {
  id: string
  timestamp: string
  level: "INFO" | "WARNING" | "ERROR"
  source: string
  message: string
  user: string
  ip: string
  details?: string
  type: LogType
}

type SystemLog = BaseLog & {
  type: "SYSTEM"
}

type Log = SystemLog

export default function LogsPage() {
  const [expandedLogs, setExpandedLogs] = useState<string[]>([])
  const [levelFilter, setLevelFilter] = useState<string>("all")
  const [searchTerm, setSearchTerm] = useState("")

  const toggleLogExpand = (id: string) => {
    setExpandedLogs(prev => 
      prev.includes(id) 
        ? prev.filter(logId => logId !== id) 
        : [...prev, id]
    )
  }
  
  const logs: Log[] = [
    {
      id: "LOG123458",
      timestamp: "2024-04-27 10:20:45",
      level: "ERROR",
      source: "Database",
      message: "Database connection error",
      user: "system",
      ip: "internal",
      type: "SYSTEM",
      details: "Database connection timeout after 30s. Connection string: mongodb://farm-admin:****@db-server:27017/farm-admin-db. Error code: ETIMEDOUT",
    },
    {
      id: "LOG123459",
      timestamp: "2024-04-27 10:18:22",
      level: "INFO",
      source: "Crop Service",
      message: "Crop data updated",
      user: "maria.garcia@example.com",
      ip: "192.168.1.42",
      type: "SYSTEM",
      details: "Crop data updated: crop_id=789, field_id=456, species='Corn', variety='Sweet Corn XH-231', planting_date='2024-03-15'",
    },
    {
      id: "LOG123460",
      timestamp: "2024-04-27 10:15:10",
      level: "INFO",
      source: "User Service",
      message: "New user registered",
      user: "robert.chen@example.com",
      ip: "198.51.100.23",
      type: "SYSTEM",
      details: "New user registered: user_id=13579, email='robert.chen@example.com', role='farmer', region='East', farms_count=2",
    },
    {
      id: "LOG123461",
      timestamp: "2024-04-27 10:12:05",
      level: "ERROR",
      source: "Payment Service",
      message: "Payment processing failed",
      user: "sophia.kim@example.com",
      ip: "192.168.1.56",
      type: "SYSTEM",
      details: "Payment processing failed: transaction_id=T98765, amount=$1,250.00, payment_method='credit_card', error='Insufficient funds'",
    },
    {
      id: "LOG123462",
      timestamp: "2024-04-27 10:10:30",
      level: "WARNING",
      source: "Security",
      message: "Multiple failed login attempts",
      user: "unknown",
      ip: "203.0.113.100",
      type: "SYSTEM",
      details: "Suspicious activity detected: multiple failed login attempts. Attempts: 5, Username: admin, Client: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    },
    {
      id: "LOG123463",
      timestamp: "2024-04-27 10:08:15",
      level: "INFO",
      source: "Marketplace",
      message: "New listing created",
      user: "james.wilson@example.com",
      ip: "192.168.1.78",
      type: "SYSTEM",
      details: "New listing created: listing_id=L4567, product='Organic Tomatoes', quantity='500 kg', price='$3.50/kg', farmer_id=F789",
    },
    {
      id: "LOG123464",
      timestamp: "2024-04-27 10:05:20",
      level: "INFO",
      source: "Notification",
      message: "Email notification sent",
      user: "system",
      ip: "internal",
      type: "SYSTEM",
      details: "Email notification sent: notification_id=N7890, recipient='farmers@example.com', template='subsidy_announcement', status='delivered'",
    },
    {
      id: "LOG123465",
      timestamp: "2024-04-27 10:02:45",
      level: "ERROR",
      source: "API Service",
      message: "API rate limit exceeded",
      user: "api_client",
      ip: "198.51.100.42",
      type: "SYSTEM",
      details: "API rate limit exceeded: client_id=C12345, endpoint='/api/v1/crops', limit=100, current=142, reset_time='2024-04-27 11:02:45'",
    },
  ]

  // Filter logs based on level and search term
  const filteredLogs = logs.filter(log => {
    const matchesLevel = levelFilter === "all" || log.level === levelFilter;
    const matchesSearch = searchTerm === "" || 
      log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.source.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.user.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (log.details && log.details.toLowerCase().includes(searchTerm.toLowerCase()));
    
    return matchesLevel && matchesSearch;
  });

  // Count logs by level
  const infoCount = logs.filter(log => log.level === "INFO").length;
  const warningCount = logs.filter(log => log.level === "WARNING").length;
  const errorCount = logs.filter(log => log.level === "ERROR").length;

  return (
    <DashboardShell>
      <Header />
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight mb-1">System Logs</h1>
          <p className="text-muted-foreground">Monitor system activity and track errors</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" asChild>
            <Link href="/dashboard/call-logs">View Call Logs</Link>
          </Button>
          <Button>Export Logs</Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Logs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{logs.length}</div>
            <p className="text-xs text-muted-foreground mt-1">Past 24 hours</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Errors</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{errorCount}</div>
            <p className="text-xs text-muted-foreground mt-1">
              {Math.round((errorCount / logs.length) * 100)}% of total logs
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Warnings</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{warningCount}</div>
            <p className="text-xs text-muted-foreground mt-1">Requires attention</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>System Logs</CardTitle>
          <CardDescription>
            Technical logs for system activities, errors, and warnings
          </CardDescription>
          <div className="flex flex-col sm:flex-row gap-4 mt-4">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="level-filter">Filter by Level</Label>
              <Select 
                value={levelFilter}
                onValueChange={setLevelFilter}
              >
                <SelectTrigger id="level-filter" className="w-[180px]">
                  <SelectValue placeholder="All Levels" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Levels</SelectItem>
                  <SelectItem value="INFO">Info</SelectItem>
                  <SelectItem value="WARNING">Warning</SelectItem>
                  <SelectItem value="ERROR">Error</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex-1 flex flex-col gap-1.5">
              <Label htmlFor="search">Search</Label>
              <Input
                id="search"
                placeholder="Search by message, source or user..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[600px] w-full">
            <div className="border rounded-md">
              <Table>
                <TableHeader className="bg-muted/50">
                  <TableRow>
                    <TableHead className="w-[60px] text-center"></TableHead>
                    <TableHead className="w-[180px] font-medium">Timestamp</TableHead>
                    <TableHead className="w-[100px] font-medium">Level</TableHead>
                    <TableHead className="w-[130px] font-medium">Source</TableHead>
                    <TableHead className="font-medium">Message</TableHead>
                    <TableHead className="w-[150px] font-medium">User</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredLogs.length > 0 ? (
                    filteredLogs.map((log) => (
                      <React.Fragment key={log.id}>
                        <TableRow 
                          className="hover:bg-muted/50 transition-colors"
                        >
                          <TableCell className="p-2 text-center">
                            <Button 
                              variant="ghost" 
                              size="sm" 
                              className="h-8 w-8 p-0 mx-auto"
                              onClick={() => toggleLogExpand(log.id)}
                            >
                              {expandedLogs.includes(log.id) ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                            </Button>
                          </TableCell>
                          <TableCell className="font-mono text-xs">
                            {log.timestamp}
                          </TableCell>
                          <TableCell>
                            {log.level === "INFO" && (
                              <div className="flex items-center gap-2">
                                <Info className="h-4 w-4 text-blue-500" />
                                <Badge variant="outline" className="bg-blue-50 text-xs">INFO</Badge>
                              </div>
                            )}
                            {log.level === "WARNING" && (
                              <div className="flex items-center gap-2">
                                <AlertTriangle className="h-4 w-4 text-yellow-500" />
                                <Badge variant="outline" className="bg-yellow-50 text-xs">WARNING</Badge>
                              </div>
                            )}
                            {log.level === "ERROR" && (
                              <div className="flex items-center gap-2">
                                <XCircle className="h-4 w-4 text-red-500" />
                                <Badge variant="outline" className="bg-red-50 text-xs">ERROR</Badge>
                              </div>
                            )}
                          </TableCell>
                          <TableCell className="text-xs">{log.source}</TableCell>
                          <TableCell className="text-xs">{log.message}</TableCell>
                          <TableCell className="text-xs">{log.user}</TableCell>
                        </TableRow>
                        {expandedLogs.includes(log.id) && (
                          <TableRow 
                            className="bg-muted/30 border-t border-dashed"
                          >
                            <TableCell colSpan={6} className="p-0">
                              <div className="px-6 py-4 space-y-4">
                                {log.details && (
                                  <div className="space-y-1">
                                    <p className="text-sm font-medium">Additional Details</p>
                                    <p className="text-sm text-muted-foreground whitespace-pre-wrap bg-muted/50 p-3 rounded-md">{log.details}</p>
                                  </div>
                                )}
                                
                                {log.ip !== "internal" && (
                                  <div className="space-y-1">
                                    <p className="text-sm font-medium">IP Address</p>
                                    <p className="text-sm font-mono">{log.ip}</p>
                                  </div>
                                )}
                                
                                <div className="flex justify-end gap-2 pt-2 border-t">
                                  <Button variant="outline" size="sm">
                                    <FileSearch className="h-3 w-3 mr-1" />
                                    Investigate
                                  </Button>
                                  <Button size="sm">
                                    View full details
                                  </Button>
                                </div>
                              </div>
                            </TableCell>
                          </TableRow>
                        )}
                      </React.Fragment>
                    ))
                  ) : (
                    <TableRow>
                      <TableCell colSpan={6} className="h-24 text-center">
                        <div className="flex flex-col items-center gap-2 py-4">
                          <FileSearch className="h-6 w-6 text-muted-foreground" />
                          <p className="text-muted-foreground">No logs found matching your criteria</p>
                        </div>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </DashboardShell>
  )
}

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
import { PhoneCall, PhoneOff, PhoneForwarded, ChevronDown, ChevronUp, Phone } from "lucide-react"
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

type BaseLog = {
  id: string
  timestamp: string
  level: "INFO" | "WARNING" | "ERROR"
  source: string
  message: string
  user: string
  ip: string
  details?: string
  type: "CALL"
}

type CallLog = BaseLog & {
  callDetails: {
    duration: string
    status: "COMPLETED" | "MISSED" | "FAILED" | "VOICEMAIL"
    recording?: string
    phoneNumber: string
    agent?: string
    summary?: string
    issue?: string
    convertedToSale?: boolean
  }
}

export default function CallLogsPage() {
  const [expandedLogs, setExpandedLogs] = useState<string[]>([])
  const [statusFilter, setStatusFilter] = useState<string>("all")
  const [searchTerm, setSearchTerm] = useState("")
  
  const toggleLogExpand = (id: string) => {
    setExpandedLogs(prev => 
      prev.includes(id) 
        ? prev.filter(logId => logId !== id) 
        : [...prev, id]
    )
  }
  
  const callLogs: CallLog[] = [
    {
      id: "CALL123001",
      timestamp: "2024-04-28 15:42:33",
      level: "INFO",
      source: "Call Center",
      message: "Incoming call from customer",
      user: "agent_emma",
      ip: "192.168.1.35",
      type: "CALL",
      callDetails: {
        duration: "08:24",
        status: "COMPLETED",
        recording: "https://storage.farm-admin.com/recordings/call123001.mp3",
        phoneNumber: "+1 (555) 123-4567",
        agent: "Emma Watson",
        summary: "Customer inquired about organic certification process. Provided information about documentation requirements and timelines.",
        convertedToSale: true
      },
      details: "Customer: John Farmer\nTopic: Organic certification\nFollow-up: Scheduled for 05/02/2024"
    },
    {
      id: "CALL123002",
      timestamp: "2024-04-28 14:20:15",
      level: "WARNING",
      source: "Call Center",
      message: "Call disconnected prematurely",
      user: "agent_michael",
      ip: "192.168.1.42",
      type: "CALL",
      callDetails: {
        duration: "03:12",
        status: "FAILED",
        recording: "https://storage.farm-admin.com/recordings/call123002.mp3",
        phoneNumber: "+1 (555) 987-6543",
        agent: "Michael Brown",
        summary: "Customer was discussing subsidy application when call dropped unexpectedly.",
        issue: "Connection failure during peak hours",
        convertedToSale: false
      },
      details: "Customer: Sarah Miller\nTopic: Subsidy application\nNetworkStatus: Unstable\nAction: Agent attempted callback but no answer"
    },
    {
      id: "CALL123003",
      timestamp: "2024-04-28 11:35:50",
      level: "ERROR",
      source: "Call Center",
      message: "Missed call from priority customer",
      user: "system",
      ip: "internal",
      type: "CALL",
      callDetails: {
        duration: "00:00",
        status: "MISSED",
        phoneNumber: "+1 (555) 555-7890",
        issue: "No available agents during call queue overflow",
        convertedToSale: false
      },
      details: "Customer: Robert Johnson (Gold tier)\nMissed during peak hours\nAction: Manager to personally follow up"
    },
    {
      id: "CALL123004",
      timestamp: "2024-04-28 10:05:22",
      level: "INFO",
      source: "Call Center",
      message: "Voicemail received",
      user: "system",
      ip: "internal",
      type: "CALL",
      callDetails: {
        duration: "01:45",
        status: "VOICEMAIL",
        recording: "https://storage.farm-admin.com/recordings/vm123004.mp3",
        phoneNumber: "+1 (555) 222-3333",
        summary: "Customer left message about technical issues with the marketplace app"
      },
      details: "Voicemail transcription: 'Hi, this is David from Green Valley Farms. I'm having trouble listing my products on your marketplace app. It keeps crashing when I try to upload photos. Please call me back at this number.'"
    }
  ]

  // Filter logs based on status and search term
  const filteredLogs = callLogs.filter(log => {
    const matchesStatus = statusFilter === "all" || log.callDetails.status === statusFilter;
    const matchesSearch = searchTerm === "" || 
      log.callDetails.phoneNumber.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (log.callDetails.agent && log.callDetails.agent.toLowerCase().includes(searchTerm.toLowerCase())) ||
      log.details?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (log.callDetails.summary && log.callDetails.summary.toLowerCase().includes(searchTerm.toLowerCase()));
    
    return matchesStatus && matchesSearch;
  });

  return (
    <DashboardShell>
      <Header />
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight mb-1">Call Logs</h1>
          <p className="text-muted-foreground">Track customer calls and follow-ups</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" asChild>
            <Link href="/dashboard/logs">View System Logs</Link>
          </Button>
          <Button>Export Calls</Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Calls</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{callLogs.length}</div>
            <p className="text-xs text-muted-foreground mt-1">Past 7 days</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Completed Calls</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{callLogs.filter(log => log.callDetails.status === "COMPLETED").length}</div>
            <p className="text-xs text-muted-foreground mt-1">
              {Math.round((callLogs.filter(log => log.callDetails.status === "COMPLETED").length / callLogs.length) * 100)}% completion rate
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Missed Calls</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{callLogs.filter(log => log.callDetails.status === "MISSED").length}</div>
            <p className="text-xs text-muted-foreground mt-1">Requires follow-up</p>
          </CardContent>
        </Card>
        {/* <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Conversion Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {Math.round((callLogs.filter(log => log.callDetails.convertedToSale).length / callLogs.length) * 100)}%
            </div>
            <p className="text-xs text-muted-foreground mt-1">Calls converted to sales</p>
          </CardContent>
        </Card> */}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Call Records</CardTitle>
          <CardDescription>
            Recent customer interactions via phone
          </CardDescription>
          <div className="flex flex-col sm:flex-row gap-4 mt-4">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="status-filter">Filter by Status</Label>
              <Select 
                value={statusFilter}
                onValueChange={setStatusFilter}
              >
                <SelectTrigger id="status-filter" className="w-[180px]">
                  <SelectValue placeholder="All Statuses" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Statuses</SelectItem>
                  <SelectItem value="COMPLETED">Completed</SelectItem>
                  <SelectItem value="MISSED">Missed</SelectItem>
                  <SelectItem value="FAILED">Failed</SelectItem>
                  <SelectItem value="VOICEMAIL">Voicemail</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex-1 flex flex-col gap-1.5">
              <Label htmlFor="search">Search</Label>
              <Input
                id="search"
                placeholder="Search by phone, agent or content..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[500px] w-full">
            <div className="border rounded-md">
              <Table>
                <TableHeader className="bg-muted/50">
                  <TableRow>
                    <TableHead className="w-[60px] text-center"></TableHead>
                    <TableHead className="w-[180px] font-medium">Timestamp</TableHead>
                    <TableHead className="w-[120px] font-medium">Status</TableHead>
                    <TableHead className="w-[150px] font-medium">Phone Number</TableHead>
                    <TableHead className="font-medium">Agent</TableHead>
                    <TableHead className="w-[100px] font-medium text-center">Duration</TableHead>
                    <TableHead className="w-[100px] font-medium text-center">Converted</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredLogs.length > 0 ? (
                    filteredLogs.map((log) => (
                      <React.Fragment key={log.id}>
                        <TableRow 
                          key={`main-${log.id}`}
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
                            {log.callDetails.status === "COMPLETED" && (
                              <div className="flex items-center gap-2">
                                <PhoneCall className="h-4 w-4 text-blue-500 flex-shrink-0" />
                                <Badge variant="outline" className="bg-blue-50 text-xs">Completed</Badge>
                              </div>
                            )}
                            {log.callDetails.status === "MISSED" && (
                              <div className="flex items-center gap-2">
                                <PhoneOff className="h-4 w-4 text-red-500" />
                                <Badge variant="outline" className="bg-red-50 text-xs">Missed</Badge>
                              </div>
                            )}
                            {log.callDetails.status === "FAILED" && (
                              <div className="flex items-center gap-2">
                                <PhoneOff className="h-4 w-4 text-red-500" />
                                <Badge variant="outline" className="bg-red-50 text-xs">Failed</Badge>
                              </div>
                            )}
                            {log.callDetails.status === "VOICEMAIL" && (
                              <div className="flex items-center gap-2">
                                <PhoneForwarded className="h-4 w-4 text-yellow-500" />
                                <Badge variant="outline" className="bg-yellow-50 text-xs">Voicemail</Badge>
                              </div>
                            )}
                          </TableCell>
                          <TableCell className="font-mono text-xs">
                            {log.callDetails.phoneNumber}
                          </TableCell>
                          <TableCell className="text-sm">
                            {log.callDetails.agent || <span className="text-muted-foreground text-xs italic">Unassigned</span>}
                          </TableCell>
                          <TableCell className="text-sm font-medium text-center">
                            {log.callDetails.duration}
                          </TableCell>
                          <TableCell className="text-center">
                            {log.callDetails.convertedToSale ? (
                              <Badge className="bg-green-100 text-green-800 hover:bg-green-100">Yes</Badge>
                            ) : (
                              <Badge variant="outline" className="text-muted-foreground">No</Badge>
                            )}
                          </TableCell>
                        </TableRow>
                        {expandedLogs.includes(log.id) && (
                          <TableRow 
                            key={`detail-${log.id}`}
                            className="bg-muted/30 border-t border-dashed"
                          >
                            <TableCell colSpan={7} className="p-0">
                              <div className="px-6 py-4 space-y-4">
                                <div className="grid gap-4 md:grid-cols-2">
                                  {log.callDetails.summary && (
                                    <div className="space-y-1">
                                      <p className="text-sm font-medium">Conversation Summary</p>
                                      <p className="text-sm text-muted-foreground">{log.callDetails.summary}</p>
                                    </div>
                                  )}
                                  
                                  {log.callDetails.issue && (
                                    <div className="space-y-1">
                                      <p className="text-sm font-medium">Issue</p>
                                      <p className="text-sm text-muted-foreground">{log.callDetails.issue}</p>
                                    </div>
                                  )}
                                </div>
                                
                                {log.details && (
                                  <div className="space-y-1">
                                    <p className="text-sm font-medium">Details</p>
                                    <p className="text-sm text-muted-foreground whitespace-pre-wrap bg-muted/50 p-3 rounded-md">{log.details}</p>
                                  </div>
                                )}
                                
                                {log.callDetails.recording && (
                                  <div className="space-y-1">
                                    <p className="text-sm font-medium">Recording</p>
                                    <audio controls className="w-full h-8 mt-1">
                                      <source src={log.callDetails.recording} type="audio/mpeg" />
                                      Your browser does not support the audio element.
                                    </audio>
                                  </div>
                                )}
                                
                                <div className="flex justify-end gap-2 pt-2 border-t">
                                  <Button variant="outline" size="sm">
                                    <Phone className="h-3 w-3 mr-1" />
                                    Call back
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
                      <TableCell colSpan={7} className="h-24 text-center">
                        <div className="flex flex-col items-center gap-2 py-4">
                          <PhoneOff className="h-6 w-6 text-muted-foreground" />
                          <p className="text-muted-foreground">No call logs found matching your criteria</p>
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

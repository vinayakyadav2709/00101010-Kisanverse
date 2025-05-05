import type React from "react"
import { MainSidebar } from "@/components/sidebar"
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar"

interface DashboardShellProps {
  children: React.ReactNode
}

export function DashboardShell({ children }: DashboardShellProps) {
  return (
    <SidebarProvider>
      <div className="grid min-h-screen w-full lg:grid-cols-[280px_1fr]">
        <MainSidebar />
        <SidebarInset>
          <div className="flex flex-col">{children}</div>
        </SidebarInset>
      </div>
    </SidebarProvider>
  )
}

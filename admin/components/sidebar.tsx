"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Users, Settings, Store, MessageSquare, LayoutDashboard, FileText } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"

export function MainSidebar() {
  const pathname = usePathname()

  const routes = [
    {
      title: "Dashboard",
      icon: LayoutDashboard,
      href: "/dashboard",
      variant: "default",
    },
    {
      title: "Subsidy",
      icon: MessageSquare,
      href: "/dashboard/subsidy",
      variant: "default",
    },
    {
      title: "Marketplace",
      icon: Store,
      href: "/dashboard/marketplace",
      variant: "default",
    },
    {
      title: "Users (Directory)",
      icon: Users,
      href: "/dashboard/users",
      variant: "default",
    },
    {
      title: "Settings",
      icon: Settings,
      href: "/dashboard/settings",
      variant: "default",
    },
    {
      title: "Calls",
      icon: Settings,
      href: "/dashboard/call-logs",
      variant: "default",
    },
    {
      title: "Logging",
      icon: FileText,
      href: "/dashboard/logs",
      variant: "default",
    },
  ]

  return (
    <Sidebar className="border-r">
      <SidebarHeader className="border-b px-6 py-3">
        <Link href="/" className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-5 w-5 text-white"
            >
              <path d="M12 2a10 10 0 1 0 10 10H12V2z" />
              <path d="M21 12a9 9 0 0 0-9-9v9h9z" />
              <circle cx="12" cy="12" r="4" />
            </svg>
          </div>
          <div className="flex flex-col">
            <span className="text-lg font-bold leading-none text-primary">KrishiVerse</span>
          </div>
        </Link>
      </SidebarHeader>
      <SidebarContent className="p-4">
        <ScrollArea className="h-[calc(100vh-10rem)]">
          <SidebarMenu className="space-y-2">
            {routes.map((route) => (
              <SidebarMenuItem key={route.href} className="px-2">
                <SidebarMenuButton 
                  asChild 
                  isActive={pathname === route.href}
                  className="py-2.5"
                >
                  <Link href={route.href} className="flex items-center gap-3">
                    <route.icon className="h-5 w-5" />
                    <span className="font-medium">{route.title}</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </ScrollArea>
      </SidebarContent>
    </Sidebar>
  )
}

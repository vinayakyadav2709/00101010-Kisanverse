import { Bell } from "lucide-react"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"

type Notification = {
  id: string
  title: string
  description: string
  time: string
  read: boolean
}

export function Notifications() {
  // Dummy notification data
  const notifications: Notification[] = [
    {
      id: "n1",
      title: "New Subsidy Available",
      description: "A new subsidy program has been added for organic farming",
      time: "10 minutes ago",
      read: false,
    },
    {
      id: "n2",
      title: "Marketplace Update",
      description: "3 new items have been listed in the marketplace",
      time: "2 hours ago",
      read: false,
    },
    {
      id: "n3",
      title: "User Registration",
      description: "New farmer registered: John Smith",
      time: "Yesterday",
      read: true,
    },
    {
      id: "n4",
      title: "System Update",
      description: "The system will undergo maintenance tonight at 2 AM",
      time: "2 days ago",
      read: true,
    },
  ]

  const unreadCount = notifications.filter(n => !n.read).length

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon" className="relative">
          <Bell className="h-5 w-5" />
          {unreadCount > 0 && (
            <Badge className="absolute -top-1 -right-1 h-5 w-5 flex items-center justify-center p-0 text-xs">
              {unreadCount}
            </Badge>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80 p-0" align="end">
        <div className="p-4 border-b">
          <div className="font-medium">Notifications</div>
          <div className="text-xs text-muted-foreground">
            You have {unreadCount} unread notifications
          </div>
        </div>
        <ScrollArea className="h-80">
          <div className="flex flex-col gap-1 py-2">
            {notifications.map((notification) => (
              <div
                key={notification.id}
                className={`p-3 hover:bg-muted cursor-pointer ${
                  notification.read ? "opacity-60" : ""
                }`}
              >
                <div className="font-medium text-sm">{notification.title}</div>
                <div className="text-xs text-muted-foreground">
                  {notification.description}
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {notification.time}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
        <div className="p-2 border-t">
          <Button variant="ghost" size="sm" className="w-full">
            Mark all as read
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  )
}

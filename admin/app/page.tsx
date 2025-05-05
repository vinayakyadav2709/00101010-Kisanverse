import Link from "next/link"
import { Button } from "@/components/ui/button"

export default function HomePage() {
  return (
    <div className="flex h-screen w-full flex-col items-center justify-center bg-primary/5">
      <div className="mx-auto flex max-w-[420px] flex-col items-center justify-center text-center">
        <div className="flex h-20 w-20 items-center justify-center rounded-full bg-primary">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-10 w-10 text-white"
          >
            <path d="M12 2a10 10 0 1 0 10 10H12V2z" />
            <path d="M21 12a9 9 0 0 0-9-9v9h9z" />
            <circle cx="12" cy="12" r="4" />
          </svg>
        </div>
        <h1 className="mt-6 text-3xl font-bold">Farm Management System</h1>
        <p className="mt-2 text-muted-foreground">Admin Dashboard for managing your farming application</p>
        <div className="mt-6 flex gap-2">
          <Button asChild>
            <Link href="/dashboard">Go to Dashboard</Link>
          </Button>
        </div>
      </div>
    </div>
  )
}

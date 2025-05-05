"use client"

import { useState, useEffect } from "react"
import { useParams, useRouter } from "next/navigation"
import { getListingById, getBids, acceptBid, rejectBid, Listing, Bid } from "@/services/api"
import { Header } from "@/components/header"
import { DashboardShell } from "@/components/dashboard-shell"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { toast } from "@/components/ui/use-toast"
import { ArrowLeft, Clock, Check, X, Tag, Loader2 } from "lucide-react"

export default function ListingDetailPage() {
  const [listing, setListing] = useState<Listing | null>(null)
  const [bids, setBids] = useState<Bid[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isProcessing, setIsProcessing] = useState(false)
  const params = useParams()
  const router = useRouter()
  const listingId = params.id as string
  
  // Mock user email - in a real app this would come from authentication
  const userEmail = "admin@example.com"

  useEffect(() => {
    async function fetchListingAndBids() {
      try {
        setIsLoading(true)
        const listingData = await getListingById(listingId)
        setListing(listingData)
        
        const bidsResponse = await getBids(userEmail, 'all', listingId)
        setBids(bidsResponse.documents)
      } catch (error) {
        console.error("Failed to fetch listing details:", error)
        toast({
          title: "Error",
          description: "Could not load listing details. Please try again later.",
          variant: "destructive",
        })
      } finally {
        setIsLoading(false)
      }
    }

    if (listingId) {
      fetchListingAndBids()
    }
  }, [listingId, userEmail])

  const handleAcceptBid = async (bidId: string) => {
    try {
      setIsProcessing(true)
      await acceptBid(bidId, userEmail)
      
      // Refresh bids
      const bidsResponse = await getBids(userEmail, 'all', listingId)
      setBids(bidsResponse.documents)
      
      toast({
        title: "Bid accepted",
        description: "The bid has been accepted successfully.",
      })
    } catch (error) {
      console.error("Failed to accept bid:", error)
      toast({
        title: "Error",
        description: "Could not accept the bid. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsProcessing(false)
    }
  }

  const handleRejectBid = async (bidId: string) => {
    try {
      setIsProcessing(true)
      await rejectBid(bidId, userEmail)
      
      // Refresh bids
      const bidsResponse = await getBids(userEmail, 'all', listingId)
      setBids(bidsResponse.documents)
      
      toast({
        title: "Bid rejected",
        description: "The bid has been rejected.",
      })
    } catch (error) {
      console.error("Failed to reject bid:", error)
      toast({
        title: "Error",
        description: "Could not reject the bid. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsProcessing(false)
    }
  }

  const getBadgeColor = (status: string) => {
    switch (status.toLowerCase()) {
      case "pending": return "bg-yellow-500"
      case "accepted": return "bg-green-500"
      case "rejected": return "bg-red-500"
      case "fulfilled": return "bg-blue-500"
      default: return "bg-gray-500"
    }
  }

  if (isLoading) {
    return (
      <DashboardShell>
        <Header />
        <main className="flex flex-1 items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </main>
      </DashboardShell>
    )
  }

  if (!listing) {
    return (
      <DashboardShell>
        <Header />
        <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-6">
          <div className="flex items-center">
            <Button variant="ghost" onClick={() => router.push("/dashboard/marketplace")}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Marketplace
            </Button>
          </div>
          <div className="flex flex-col items-center justify-center p-8">
            <h2 className="text-xl font-semibold">Listing not found</h2>
            <p className="text-muted-foreground">The listing you're looking for doesn't exist or has been removed.</p>
          </div>
        </main>
      </DashboardShell>
    )
  }

  return (
    <DashboardShell>
      <Header />
      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-6">
        <div className="flex items-center">
          <Button variant="ghost" onClick={() => router.push("/dashboard/marketplace")}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Marketplace
          </Button>
        </div>
        
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Listing Details */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <div className="flex items-center justify-between">
                <Badge variant="outline">{listing.crop_type}</Badge>
                <Badge className={`
                  ${listing.status === "listed" ? "bg-green-500" : 
                    listing.status === "cancelled" || listing.status === "removed" ? "bg-red-500" : 
                    listing.status === "fulfilled" ? "bg-blue-500" : "bg-gray-500"}
                `}>
                  {listing.status}
                </Badge>
              </div>
              <CardTitle className="text-2xl">{listing.crop_type}</CardTitle>
              <CardDescription>
                <div className="flex items-center mt-2">
                  <Avatar className="h-6 w-6 mr-2">
                    <AvatarImage src="/placeholder-user.jpg" alt="Farmer" />
                    <AvatarFallback>{listing.farmer_id.substring(0, 2).toUpperCase()}</AvatarFallback>
                  </Avatar>
                  <span>Farmer ID: {listing.farmer_id}</span>
                </div>
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Price per kg</p>
                  <p className="text-xl font-semibold">${listing.price_per_kg}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Total Quantity</p>
                  <p className="text-xl font-semibold">{listing.total_quantity} kg</p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Available Quantity</p>
                  <p className="text-xl font-semibold">{listing.available_quantity} kg</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Listed Date</p>
                  <p className="text-xl font-semibold">{new Date(listing.$createdAt).toLocaleDateString()}</p>
                </div>
              </div>
            </CardContent>
            <CardFooter className="border-t bg-muted/50 px-6 py-3">
              <div className="flex items-center w-full justify-between">
                <div className="flex items-center gap-1">
                  <Tag className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">ID: {listing.$id}</span>
                </div>
                <div className="flex items-center gap-1">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">
                    Last updated: {new Date(listing.$updatedAt).toLocaleDateString()}
                  </span>
                </div>
              </div>
            </CardFooter>
          </Card>

          {/* Bids Section */}
          <Card>
            <CardHeader>
              <CardTitle>Bids ({bids.length})</CardTitle>
              <CardDescription>All bids placed on this listing</CardDescription>
            </CardHeader>
            <CardContent className="max-h-[400px] overflow-auto space-y-4">
              {bids.length === 0 ? (
                <p className="text-center text-muted-foreground py-8">No bids yet</p>
              ) : (
                bids.map((bid) => (
                  <div key={bid.$id} className="border rounded-lg p-3 space-y-2">
                    <div className="flex justify-between items-center">
                      <div>
                        <p className="font-medium">{bid.quantity} kg</p>
                        <p className="text-sm text-muted-foreground">at ${bid.price_per_kg}/kg</p>
                      </div>
                      <Badge className={getBadgeColor(bid.status)}>{bid.status}</Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <Avatar className="h-5 w-5">
                        <AvatarImage src="/placeholder-user.jpg" alt="Buyer" />
                        <AvatarFallback>{bid.buyer_id.substring(0, 2).toUpperCase()}</AvatarFallback>
                      </Avatar>
                      <p className="text-xs">Buyer ID: {bid.buyer_id.substring(0, 8)}</p>
                    </div>
                    {bid.status === "pending" && (
                      <div className="flex justify-end gap-2 mt-2">
                        <Button 
                          size="sm" 
                          variant="outline" 
                          disabled={isProcessing}
                          onClick={() => handleAcceptBid(bid.$id)}
                        >
                          <Check className="h-3 w-3 mr-1" /> Accept
                        </Button>
                        <Button 
                          size="sm" 
                          variant="outline" 
                          disabled={isProcessing}
                          onClick={() => handleRejectBid(bid.$id)}
                        >
                          <X className="h-3 w-3 mr-1" /> Reject
                        </Button>
                      </div>
                    )}
                    <div className="text-xs text-muted-foreground flex justify-between mt-1">
                      <span>Placed: {new Date(bid.$createdAt).toLocaleDateString()}</span>
                      <span>ID: {bid.$id.substring(0, 8)}</span>
                    </div>
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </DashboardShell>
  )
}

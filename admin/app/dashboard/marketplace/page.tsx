"use client"

import { Header } from "@/components/header"
import { DashboardShell } from "@/components/dashboard-shell"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Search, Plus, MoreHorizontal, Check, X, Filter, Tag, Clock, MapPin } from "lucide-react"
import { useEffect, useState } from "react"
import { getListings, cancelListing, Listing, createListing } from "@/services/api"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { toast } from "@/components/ui/use-toast"
import { useRouter } from "next/navigation"

export default function MarketplacePage() {
  const [listings, setListings] = useState<Listing[]>([])
  const [filteredListings, setFilteredListings] = useState<Listing[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedCategory, setSelectedCategory] = useState("All Categories")
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [newListing, setNewListing] = useState({
    crop_type: "",
    price_per_kg: 0,
    total_quantity: 0,
  })
  const router = useRouter()
  
  // Mock user email - in a real app this would come from authentication
  const userEmail = "admin@example.com"

  useEffect(() => {
    async function fetchListings() {
      try {
        setIsLoading(true)
        const response = await getListings()
        setListings(response.documents)
        setFilteredListings(response.documents)
      } catch (error) {
        console.error("Failed to fetch listings:", error)
        toast({
          title: "Error fetching listings",
          description: "Could not load marketplace listings. Please try again later.",
          variant: "destructive",
        })
      } finally {
        setIsLoading(false)
      }
    }

    fetchListings()
  }, [])

  useEffect(() => {
    // Filter listings based on search term and category
    const filtered = listings.filter((listing) => {
      const matchesSearch = searchTerm === "" || 
        listing.crop_type.toLowerCase().includes(searchTerm.toLowerCase())
      
      const matchesCategory = selectedCategory === "All Categories" || 
        listing.crop_type === selectedCategory
      
      return matchesSearch && matchesCategory
    })
    
    setFilteredListings(filtered)
  }, [searchTerm, selectedCategory, listings])

  const handleCreateListing = async () => {
    try {
      await createListing(userEmail, newListing)
      toast({
        title: "Listing created",
        description: "Your listing has been created successfully.",
      })
      setCreateDialogOpen(false)
      
      // Refresh listings
      const response = await getListings()
      setListings(response.documents)
      setFilteredListings(response.documents)
    } catch (error) {
      console.error("Failed to create listing:", error)
      toast({
        title: "Error creating listing",
        description: "Could not create your listing. Please try again.",
        variant: "destructive",
      })
    }
  }

  const handleCancelListing = async (listingId: string) => {
    try {
      await cancelListing(listingId, userEmail)
      toast({
        title: "Listing cancelled",
        description: "The listing has been removed from the marketplace.",
      })
      
      // Update local state
      const updatedListings = listings.map(listing => 
        listing.$id === listingId ? {...listing, status: "cancelled"} : listing
      )
      setListings(updatedListings)
      setFilteredListings(updatedListings.filter(l => l.status !== "cancelled"))
    } catch (error) {
      console.error("Failed to cancel listing:", error)
      toast({
        title: "Error cancelling listing",
        description: "Could not cancel the listing. Please try again.",
        variant: "destructive",
      })
    }
  }

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case "listed":
        return "bg-green-500"
      case "pending review":
        return "bg-yellow-500"
      case "flagged":
      case "cancelled":
      case "removed":
        return "bg-red-500"
      case "fulfilled":
        return "bg-blue-500"
      default:
        return "bg-gray-500"
    }
  }

  // Categories derived from listings
  const categories = ["All Categories", ...new Set(listings.map(listing => listing.crop_type))]

  return (
    <DashboardShell>
      <Header />
      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold tracking-tight">Marketplace</h1>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input 
              type="search" 
              placeholder="Search marketplace..." 
              className="w-full pl-8" 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <Button variant="outline" size="icon">
            <Filter className="h-4 w-4" />
            <span className="sr-only">Filter</span>
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">
                {selectedCategory}
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="ml-2 h-4 w-4"
                >
                  <path d="m6 9 6 6 6-6" />
                </svg>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {categories.map(category => (
                <DropdownMenuItem 
                  key={category} 
                  onClick={() => setSelectedCategory(category)}
                >
                  {category}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        <Tabs defaultValue="all" className="space-y-4">
          <TabsList>
            <TabsTrigger value="all">All Listings</TabsTrigger>
            <TabsTrigger value="pending">Pending Review</TabsTrigger>
            <TabsTrigger value="flagged">Flagged</TabsTrigger>
          </TabsList>
          <TabsContent value="all" className="space-y-4">
            {isLoading ? (
              <div className="flex justify-center p-8">Loading listings...</div>
            ) : filteredListings.length === 0 ? (
              <div className="text-center p-8">
                <p className="text-muted-foreground">No listings found.</p>
              </div>
            ) : (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {filteredListings.map((listing) => (
                  <Card key={listing.$id} className="overflow-hidden">
                    <div className="relative h-[200px] w-full">
                      <img
                        src={`/placeholder.svg?height=200&width=300&text=${encodeURIComponent(listing.crop_type)}`}
                        alt={listing.crop_type}
                        className="h-full w-full object-cover"
                      />
                      <Badge
                        className={`absolute right-2 top-2 ${getStatusColor(listing.status)}`}
                      >
                        {listing.status}
                      </Badge>
                    </div>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <Badge variant="outline">{listing.crop_type}</Badge>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon">
                              <MoreHorizontal className="h-4 w-4" />
                              <span className="sr-only">More options</span>
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={() => router.push(`/dashboard/marketplace/${listing.$id}`)}>
                              View details
                            </DropdownMenuItem>
                            <DropdownMenuItem>Edit listing</DropdownMenuItem>
                            <DropdownMenuItem onClick={() => handleCancelListing(listing.$id)}>
                              {listing.status === "listed" ? "Cancel listing" : "Activate listing"}
                            </DropdownMenuItem>
                            <DropdownMenuItem className="text-red-600">
                              Remove listing
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                      <CardTitle className="line-clamp-1">{listing.crop_type}</CardTitle>
                      <CardDescription className="line-clamp-2">
                        Available Quantity: {listing.available_quantity} kg
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Tag className="h-4 w-4 text-muted-foreground" />
                          <span className="text-sm font-medium">${listing.price_per_kg} per kg</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <MapPin className="h-4 w-4 text-muted-foreground" />
                          <span className="text-sm">Farmer ID: {listing.farmer_id.substring(0, 8)}</span>
                        </div>
                      </div>
                    </CardContent>
                    <CardFooter className="border-t bg-muted/50 px-6 py-3">
                      <div className="flex items-center justify-between w-full">
                        <div className="flex items-center gap-2">
                          <Avatar className="h-6 w-6">
                            <AvatarImage src="/placeholder-user.jpg" alt="Farmer" />
                            <AvatarFallback>{listing.farmer_id.substring(0, 2).toUpperCase()}</AvatarFallback>
                          </Avatar>
                          <span className="text-xs">Farmer</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Clock className="h-3 w-3 text-muted-foreground" />
                          <span className="text-xs text-muted-foreground">
                            Listed {new Date(listing.$createdAt).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                    </CardFooter>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>
          <TabsContent value="pending" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {isLoading ? (
                <div className="flex justify-center p-8">Loading listings...</div>
              ) : (
                filteredListings.filter(listing => listing.status.toLowerCase() === "pending review")
                  .map((listing) => (
                    <Card key={listing.$id} className="overflow-hidden">
                      <div className="relative h-[200px] w-full">
                        <img
                          src={`/placeholder.svg?height=200&width=300&text=${encodeURIComponent(listing.crop_type)}`}
                          alt={listing.crop_type}
                          className="h-full w-full object-cover"
                        />
                        <Badge className="absolute right-2 top-2 bg-yellow-500">{listing.status}</Badge>
                      </div>
                      <CardHeader>
                        <div className="flex items-center justify-between">
                          <Badge variant="outline">{listing.crop_type}</Badge>
                          <div className="flex items-center gap-1">
                            <Button variant="outline" size="icon" className="h-8 w-8 rounded-full">
                              <Check className="h-4 w-4 text-green-600" />
                              <span className="sr-only">Approve</span>
                            </Button>
                            <Button variant="outline" size="icon" className="h-8 w-8 rounded-full">
                              <X className="h-4 w-4 text-red-600" />
                              <span className="sr-only">Reject</span>
                            </Button>
                          </div>
                        </div>
                        <CardTitle className="line-clamp-1">{listing.crop_type}</CardTitle>
                        <CardDescription className="line-clamp-2">
                          Available Quantity: {listing.available_quantity} kg
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Tag className="h-4 w-4 text-muted-foreground" />
                            <span className="text-sm font-medium">${listing.price_per_kg} per kg</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <MapPin className="h-4 w-4 text-muted-foreground" />
                            <span className="text-sm">Farmer ID: {listing.farmer_id.substring(0, 8)}</span>
                          </div>
                        </div>
                      </CardContent>
                      <CardFooter className="border-t bg-muted/50 px-6 py-3">
                        <div className="flex items-center justify-between w-full">
                          <div className="flex items-center gap-2">
                            <Avatar className="h-6 w-6">
                              <AvatarImage src="/placeholder-user.jpg" alt="Farmer" />
                              <AvatarFallback>{listing.farmer_id.substring(0, 2).toUpperCase()}</AvatarFallback>
                            </Avatar>
                            <span className="text-xs">Farmer</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Clock className="h-3 w-3 text-muted-foreground" />
                            <span className="text-xs text-muted-foreground">
                              Listed {new Date(listing.$createdAt).toLocaleDateString()}
                            </span>
                          </div>
                        </div>
                      </CardFooter>
                    </Card>
                  ))
              )}
            </div>
          </TabsContent>
          <TabsContent value="flagged" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {isLoading ? (
                <div className="flex justify-center p-8">Loading listings...</div>
              ) : (
                filteredListings.filter(listing => listing.status.toLowerCase() === "flagged")
                  .map((listing) => (
                    <Card key={listing.$id} className="overflow-hidden">
                      <div className="relative h-[200px] w-full">
                        <img
                          src={`/placeholder.svg?height=200&width=300&text=${encodeURIComponent(listing.crop_type)}`}
                          alt={listing.crop_type}
                          className="h-full w-full object-cover"
                        />
                        <Badge className="absolute right-2 top-2 bg-red-500">{listing.status}</Badge>
                      </div>
                      <CardHeader>
                        <div className="flex items-center justify-between">
                          <Badge variant="outline">{listing.crop_type}</Badge>
                          <div className="flex items-center gap-1">
                            <Button variant="outline" size="icon" className="h-8 w-8 rounded-full">
                              <Check className="h-4 w-4 text-green-600" />
                              <span className="sr-only">Approve</span>
                            </Button>
                            <Button variant="outline" size="icon" className="h-8 w-8 rounded-full">
                              <X className="h-4 w-4 text-red-600" />
                              <span className="sr-only">Remove</span>
                            </Button>
                          </div>
                        </div>
                        <CardTitle className="line-clamp-1">{listing.crop_type}</CardTitle>
                        <CardDescription className="line-clamp-2">
                          Available Quantity: {listing.available_quantity} kg
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Tag className="h-4 w-4 text-muted-foreground" />
                            <span className="text-sm font-medium">${listing.price_per_kg} per kg</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <MapPin className="h-4 w-4 text-muted-foreground" />
                            <span className="text-sm">Farmer ID: {listing.farmer_id.substring(0, 8)}</span>
                          </div>
                        </div>
                      </CardContent>
                      <CardFooter className="border-t bg-muted/50 px-6 py-3">
                        <div className="flex items-center justify-between w-full">
                          <div className="flex items-center gap-2">
                            <Avatar className="h-6 w-6">
                              <AvatarImage src="/placeholder-user.jpg" alt="Farmer" />
                              <AvatarFallback>{listing.farmer_id.substring(0, 2).toUpperCase()}</AvatarFallback>
                            </Avatar>
                            <span className="text-xs">Farmer</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Clock className="h-3 w-3 text-muted-foreground" />
                            <span className="text-xs text-muted-foreground">
                              Listed {new Date(listing.$createdAt).toLocaleDateString()}
                            </span>
                          </div>
                        </div>
                      </CardFooter>
                    </Card>
                  ))
              )}
            </div>
          </TabsContent>
        </Tabs>
      </main>

      {/* Create Listing Dialog */}
      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            
            <DialogDescription>
              Add a new crop listing to the marketplace.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="crop_type" className="text-right">
                Crop Type
              </Label>
              <Input
                id="crop_type"
                value={newListing.crop_type}
                onChange={(e) => setNewListing({ ...newListing, crop_type: e.target.value })}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="price" className="text-right">
                Price (per kg)
              </Label>
              <Input
                id="price"
                type="number"
                value={newListing.price_per_kg}
                onChange={(e) => setNewListing({ ...newListing, price_per_kg: parseFloat(e.target.value) })}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="quantity" className="text-right">
                Quantity (kg)
              </Label>
              <Input
                id="quantity"
                type="number"
                value={newListing.total_quantity}
                onChange={(e) => setNewListing({ ...newListing, total_quantity: parseFloat(e.target.value) })}
                className="col-span-3"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateListing}>Create Listing</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </DashboardShell>
  )
}

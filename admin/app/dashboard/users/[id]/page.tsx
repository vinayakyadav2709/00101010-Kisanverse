"use client"
import React from "react"
import { notFound } from "next/navigation"
import Link from "next/link"
import { Header } from "@/components/header"
import { DashboardShell } from "@/components/dashboard-shell"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { MessageSquare, Settings, AlertTriangle, ArrowLeft, Edit, UserPlus, Shield, Lock, Activity, Save } from "lucide-react"
import { getUserByEmail, updateUser, deleteUser, User } from "@/services/api"
import { useEffect, useState } from "react"
import { Input } from "@/components/ui/input"
import { useSearchParams } from "next/navigation"
import { Textarea } from "@/components/ui/textarea"

interface UserPageProps {
  params: {
    id: string // This will be the encoded email
  }
}

export default function UserPage({ params }: UserPageProps) {
  // Unwrap params with React.use()
  const unwrappedParams = React.use(params);
  
  return (
    <DashboardShell>
      <Header />
      <UserPageContent email={decodeURIComponent(unwrappedParams.id)} />
    </DashboardShell>
  )
}

// Client component to handle data fetching and state
function UserPageContent({ email }: { email: string }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [activeTab, setActiveTab] = useState("details");
  const [editForm, setEditForm] = useState({
    name: "",
    email: "",
    role: "",
    address: "",
    zipcode: "" // Added zipcode field
  });
  const [isSaving, setIsSaving] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const searchParams = useSearchParams();

  useEffect(() => {
    // Check if edit mode should be activated from query params
    if (searchParams.get('edit') === 'true') {
      setIsEditing(true);
    }
  }, [searchParams]);

  useEffect(() => {
    async function fetchUser() {
      try {
        setLoading(true);
        const userData = await getUserByEmail(email);
        setUser(userData);
        // Initialize form with user data
        setEditForm({
          name: userData.name,
          email: userData.email,
          role: userData.role,
          address: userData.address || "",
          zipcode: userData.zipcode || "" // Added zipcode field
        });
      } catch (err) {
        console.error("Failed to fetch user:", err);
        setError("Failed to load user information. Please try again.");
      } finally {
        setLoading(false);
      }
    }

    fetchUser();
  }, [email]);

  if (loading) {
    return (
      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-6">
        <div className="flex justify-center items-center h-[calc(100vh-200px)]">
          <div className="flex flex-col items-center gap-4">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent"></div>
            <p className="text-muted-foreground">Loading user data...</p>
          </div>
        </div>
      </main>
    );
  }

  if (error || !user) {
    return (
      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-6">
        <div className="bg-red-100 text-red-800 p-4 mb-4 rounded-md">
          {error || "User not found"}
        </div>
        <Button variant="outline" asChild>
          <Link href="/dashboard/users">Back to Users</Link>
        </Button>
      </main>
    );
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setEditForm(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSaveChanges = async () => {
    try {
      setIsSaving(true);
      const updatedUser = await updateUser(user.email, editForm);
      setUser(updatedUser);
      setIsEditing(false);
      // Show success message
      setShowSuccess(true);
      setTimeout(() => setShowSuccess(false), 3000); // Hide after 3 seconds
    } catch (err) {
      console.error("Failed to update user:", err);
      setError("Failed to update user information. Please try again.");
    } finally {
      setIsSaving(false);
    }
  };

  const handleDeleteUser = async () => {
    if (!window.confirm("Are you sure you want to delete this user? This action cannot be undone.")) {
      return;
    }
    
    try {
      setIsDeleting(true);
      // For this example, we'll use a mock admin email
      await deleteUser(user.email, "admin@example.com");
      // Redirect to users list after successful deletion
      window.location.href = "/dashboard/users";
    } catch (err) {
      console.error("Failed to delete user:", err);
      setError("Failed to delete user. Please try again.");
      setIsDeleting(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric"
    });
  };

  return (
    <main className="flex flex-1 flex-col gap-6 p-6 md:gap-8 md:p-8 bg-muted/5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="outline" size="icon" asChild className="rounded-full shadow-sm hover:shadow">
            <Link href="/dashboard/users">
              <ArrowLeft className="h-4 w-4" />
              <span className="sr-only">Back</span>
            </Link>
          </Button>
          <h1 className="text-2xl font-bold tracking-tight">User Profile</h1>
        </div>
        {!isEditing ? (
          <Button 
            onClick={() => setIsEditing(true)}
            className="gap-2 shadow-sm hover:shadow" 
            size="sm"
          >
            <Edit className="h-4 w-4" />
            Edit Profile
          </Button>
        ) : (
          <div className="flex gap-2">
            <Button 
              onClick={() => setIsEditing(false)} 
              variant="outline" 
              size="sm"
              className="gap-2"
            >
              Cancel
            </Button>
            <Button 
              onClick={handleSaveChanges} 
              disabled={isSaving}
              variant="default" 
              size="sm"
              className="gap-2 shadow-sm hover:shadow"
            >
              {isSaving ? (
                <>
                  <div className="animate-spin h-4 w-4 border-2 rounded-full border-t-transparent"></div>
                  Saving
                </>
              ) : (
                <>
                  <Save className="h-4 w-4" />
                  Save Changes
                </>
              )}
            </Button>
          </div>
        )}
      </div>
      
      {error && (
        <div className="bg-destructive/10 text-destructive p-4 rounded-lg border border-destructive/20 flex items-center gap-3 animate-in slide-in-from-top">
          <AlertTriangle className="h-5 w-5" />
          <p>{error}</p>
        </div>
      )}
      
      {showSuccess && (
        <div className="bg-green-50 text-green-800 p-4 rounded-lg border border-green-200 flex items-center gap-3 animate-in slide-in-from-top">
          <div className="rounded-full bg-green-100 p-1">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-600">
              <path d="M20 6L9 17l-5-5" />
            </svg>
          </div>
          <p>User profile updated successfully</p>
        </div>
      )}

      <div className="grid gap-6 md:grid-cols-[3fr_1fr]">
        <div className="space-y-6">
          <Card className="overflow-hidden border-none shadow-sm">
            <div className="bg-gradient-to-r from-primary/20 to-primary/5 h-32"></div>
            <div className="relative px-6">
              <Avatar className="h-24 w-24 absolute -top-12 ring-4 ring-background">
                <AvatarFallback className="text-xl bg-primary text-primary-foreground">{user.name.substring(0, 2)}</AvatarFallback>
              </Avatar>
              <div className="pt-16 pb-4">
                <CardTitle className="text-2xl">{user.name}</CardTitle>
                <CardDescription className="text-base flex items-center gap-2">
                  {user.email}
                  <span className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium bg-primary/10 text-primary">
                    {user.role}
                  </span>
                </CardDescription>
                <div className="mt-2 flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
                  <div className="flex items-center gap-1">
                    <UserPlus className="h-4 w-4" />
                    <span>Joined {formatDate(user.$createdAt)}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Activity className="h-4 w-4" />
                    <span>Updated {formatDate(user.$updatedAt)}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Shield className="h-4 w-4" />
                    <span className="capitalize">{user.role} account</span>
                  </div>
                </div>
              </div>
            </div>
            <CardContent className="px-0 pb-0">
              <Tabs 
                value={activeTab} 
                onValueChange={setActiveTab}
                className="w-full"
              >
                <div className="border-b">
                  <TabsList className="w-full justify-start rounded-none h-12 bg-transparent border-b px-6">
                    <TabsTrigger 
                      value="details" 
                      className="rounded-none data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none h-12"
                    >
                      Details
                    </TabsTrigger>
                    <TabsTrigger 
                      value="activity" 
                      className="rounded-none data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none h-12"
                    >
                      Activity
                    </TabsTrigger>
                    <TabsTrigger 
                      value="security" 
                      className="rounded-none data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none h-12"
                    >
                      Security
                    </TabsTrigger>
                  </TabsList>
                </div>
                <div className="p-6">
                  <TabsContent value="details" className="m-0">
                    {isEditing ? (
                      <div className="space-y-6">
                        <div className="grid gap-5 md:grid-cols-2">
                          <div className="space-y-2">
                            <label htmlFor="name" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">Name</label>
                            <Input 
                              id="name"
                              name="name"
                              value={editForm.name}
                              onChange={handleInputChange}
                              className="shadow-sm"
                              placeholder="Full name"
                            />
                          </div>
                          <div className="space-y-2">
                            <label htmlFor="email" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">Email</label>
                            <Input 
                              id="email"
                              name="email"
                              value={editForm.email}
                              onChange={handleInputChange}
                              className="shadow-sm"
                              placeholder="email@example.com"
                              type="email"
                            />
                          </div>
                          <div className="space-y-2">
                            <label htmlFor="role" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">Role</label>
                            <select 
                              id="role"
                              name="role"
                              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                              value={editForm.role}
                              onChange={handleInputChange}
                            >
                              <option value="admin">Admin</option>
                              <option value="farmer">Farmer</option>
                              <option value="buyer">Buyer</option>
                              <option value="provider">Provider</option>
                            </select>
                          </div>
                          <div className="space-y-2">
                            <label htmlFor="address" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">Address</label>
                            <Input 
                              id="address"
                              name="address"
                              value={editForm.address}
                              onChange={handleInputChange}
                              className="shadow-sm"
                              placeholder="123 Main St"
                            />
                          </div>
                          <div className="space-y-2">
                            <label htmlFor="zipcode" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">Zipcode</label>
                            <Input 
                              id="zipcode"
                              name="zipcode"
                              value={editForm.zipcode}
                              onChange={handleInputChange}
                              className="shadow-sm"
                              placeholder="12345"
                            />
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="grid gap-6 md:grid-cols-2">
                        <div className="space-y-1">
                          <h3 className="text-sm font-medium text-muted-foreground">Role</h3>
                          <p className="text-base font-medium capitalize">{user.role}</p>
                        </div>
                        <div className="space-y-1">
                          <h3 className="text-sm font-medium text-muted-foreground">Address</h3>
                          <p className="text-base font-medium">{user.address || "No address provided"}</p>
                        </div>
                        <div className="space-y-1">
                          <h3 className="text-sm font-medium text-muted-foreground">Email</h3>
                          <p className="text-base font-medium">{user.email}</p>
                        </div>
                        <div className="space-y-1">
                          <h3 className="text-sm font-medium text-muted-foreground">Zipcode</h3>
                          <p className="text-base font-medium">{user.zipcode || "No zipcode provided"}</p>
                        </div>
                        <div className="space-y-1">
                          <h3 className="text-sm font-medium text-muted-foreground">User ID</h3>
                          <p className="text-base font-medium font-mono text-xs bg-muted px-2 py-1 rounded">{user.$id}</p>
                        </div>
                        <div className="space-y-1">
                          <h3 className="text-sm font-medium text-muted-foreground">Status</h3>
                          <div className="flex items-center">
                            <div className="mr-2 h-3 w-3 rounded-full bg-green-500"></div>
                            <p className="text-base font-medium">Active</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </TabsContent>
                  <TabsContent value="activity" className="space-y-6 m-0">
                    <div className="space-y-4">
                      <div className="border-l-2 border-primary/20 pl-4 relative">
                        <div className="absolute w-3 h-3 bg-primary rounded-full -left-[6.5px] top-1"></div>
                        <div className="space-y-1">
                          <p className="text-sm font-medium">Account created</p>
                          <p className="text-xs text-muted-foreground">{formatDate(user.$createdAt)}</p>
                        </div>
                      </div>
                      <div className="border-l-2 border-primary/20 pl-4 relative">
                        <div className="absolute w-3 h-3 bg-primary rounded-full -left-[6.5px] top-1"></div>
                        <div className="space-y-1">
                          <p className="text-sm font-medium">Account updated</p>
                          <p className="text-xs text-muted-foreground">{formatDate(user.$updatedAt)}</p>
                        </div>
                      </div>
                      <div className="border-l-2 border-primary/20 pl-4 relative">
                        <div className="absolute w-3 h-3 bg-primary rounded-full -left-[6.5px] top-1"></div>
                        <div className="space-y-1">
                          <p className="text-sm font-medium">Last login</p>
                          <p className="text-xs text-muted-foreground">Today at 10:35 AM</p>
                        </div>
                      </div>
                    </div>
                  </TabsContent>
                  <TabsContent value="security" className="space-y-6 m-0">
                    <div className="space-y-4">
                      <div className="flex items-start gap-4 p-4 bg-muted/50 rounded-lg">
                        <Lock className="mt-0.5 h-5 w-5 text-primary" />
                        <div className="space-y-2 flex-1">
                          <div className="flex items-center justify-between">
                            <p className="text-sm font-medium">Password Management</p>
                            <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full">Requires verification</span>
                          </div>
                          <p className="text-xs text-muted-foreground">Password management requires additional verification</p>
                          <Button variant="outline" size="sm" className="mt-2 shadow-sm">
                            Send Reset Link
                          </Button>
                        </div>
                      </div>
                      <div className="flex items-start gap-4 p-4 bg-muted/50 rounded-lg">
                        <Shield className="mt-0.5 h-5 w-5 text-primary" />
                        <div className="space-y-2 flex-1">
                          <div className="flex items-center justify-between">
                            <p className="text-sm font-medium">Two-Factor Authentication</p>
                            <span className="text-xs bg-red-100 text-red-800 px-2 py-1 rounded-full">Not enabled</span>
                          </div>
                          <p className="text-xs text-muted-foreground">Add an extra layer of security to your account</p>
                          <Button variant="outline" size="sm" className="mt-2 shadow-sm">
                            Enable 2FA
                          </Button>
                        </div>
                      </div>
                    </div>
                  </TabsContent>
                </div>
              </Tabs>
            </CardContent>
          </Card>
        </div>
        <div className="space-y-6">
          <Card className="shadow-sm border-none">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">User Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button variant="ghost" className="w-full justify-start h-auto py-3 px-4 hover:bg-muted">
                <MessageSquare className="mr-3 h-4 w-4 text-primary" />
                <div className="text-left">
                  <div className="font-medium">Send Message</div>
                  <div className="text-xs text-muted-foreground">Contact the user directly</div>
                </div>
              </Button>
              <Button variant="ghost" className="w-full justify-start h-auto py-3 px-4 hover:bg-muted">
                <UserPlus className="mr-3 h-4 w-4 text-primary" />
                <div className="text-left">
                  <div className="font-medium">Change Role</div>
                  <div className="text-xs text-muted-foreground">Modify user permissions</div>
                </div>
              </Button>
              <Button 
                variant="ghost"
                className="w-full justify-start h-auto py-3 px-4 hover:bg-red-50 hover:text-red-700 group"
                onClick={handleDeleteUser}
                disabled={isDeleting}
              >
                <AlertTriangle className="mr-3 h-4 w-4 text-red-500 group-hover:text-red-700" />
                <div className="text-left">
                  <div className="font-medium">Delete User</div>
                  <div className="text-xs text-muted-foreground group-hover:text-red-700/70">Permanently remove this account</div>
                </div>
                {isDeleting && (
                  <div className="animate-spin ml-auto h-4 w-4 border-2 rounded-full border-red-600 border-t-transparent"></div>
                )}
              </Button>
            </CardContent>
          </Card>
          
          <Card className="shadow-sm border-none">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Admin Notes</CardTitle>
            </CardHeader>
            <CardContent>
              <Textarea 
                placeholder="Add notes about this user..." 
                className="min-h-[120px] resize-none shadow-sm"
              />
              <Button size="sm" className="mt-4 shadow-sm">
                Save Note
              </Button>
            </CardContent>
          </Card>
          
          <Card className="shadow-sm border-none">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Related Accounts</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">No related accounts found.</p>
              <Button variant="outline" size="sm" className="mt-4 w-full shadow-sm">
                Link Account
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </main>
  )
}

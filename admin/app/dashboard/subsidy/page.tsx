"use client";

import { useEffect, useState } from "react";
import { Header } from "@/components/header";
import { DashboardShell } from "@/components/dashboard-shell";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CheckCircle, Clock, MoreHorizontal, Plus, ThumbsDown, ThumbsUp, XCircle } from "lucide-react";
import { Subsidy, SubsidyRequest, approveSubsidy, getSubsidies, rejectSubsidy, getSubsidyRequests, acceptSubsidyRequest, rejectSubsidyRequest, createSubsidy } from "@/services/api";
import { format, parseISO } from "date-fns";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useForm } from "react-hook-form";

export default function SubsidyPage() {
  // State variables
  const [subsidies, setSubsidies] = useState<Subsidy[]>([]);
  const [subsidyRequests, setSubsidyRequests] = useState<SubsidyRequest[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [selectedSubsidy, setSelectedSubsidy] = useState<Subsidy | null>(null);
  const [showConfirmDialog, setShowConfirmDialog] = useState<boolean>(false);
  const [confirmAction, setConfirmAction] = useState<{ type: 'approve' | 'reject'; id: string } | null>(null);
  const [adminEmail] = useState<string>("admin@example.com"); // In a real app, this would come from authentication
  const [showAddDialog, setShowAddDialog] = useState<boolean>(false);
  const [addSubsidyLoading, setAddSubsidyLoading] = useState<boolean>(false);
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});

  // Form state for new subsidy
  const [newSubsidy, setNewSubsidy] = useState({
    program: "",
    description: "",
    eligibility: "",
    type: "cash",
    benefits: "",
    application_process: "",
    locations: [] as string[],
    dynamic_fields: "{}",
    max_recipients: 0,
    provider: ""
  });

  // Fetch subsidies and subsidy requests
  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      try {
        console.debug("Fetching subsidies for:", adminEmail);
        const subsidiesData = await getSubsidies(adminEmail);
        console.debug("Subsidies fetched successfully:", subsidiesData);
        setSubsidies(subsidiesData.documents);
        
        console.debug("Fetching subsidy requests for:", adminEmail);
        const requestsData = await getSubsidyRequests(adminEmail);
        console.debug("Subsidy requests fetched successfully:", requestsData);
        setSubsidyRequests(requestsData.documents);
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [adminEmail]);

  // Handle adding a new subsidy
  const handleAddSubsidy = async () => {
    // Basic validation
    const errors: Record<string, string> = {};
    if (!newSubsidy.program.trim()) errors.program = "Program name is required";
    if (!newSubsidy.description.trim()) errors.description = "Description is required";
    if (!newSubsidy.eligibility.trim()) errors.eligibility = "Eligibility criteria is required";
    if (!newSubsidy.benefits.trim()) errors.benefits = "Benefits is required";
    if (!newSubsidy.application_process.trim()) errors.application_process = "Application process is required";
    if (newSubsidy.locations.length === 0) errors.locations = "At least one location is required";
    if (newSubsidy.max_recipients <= 0) errors.max_recipients = "Max recipients must be greater than 0";
    if (!newSubsidy.provider.trim()) errors.provider = "Provider is required";
    
    if (Object.keys(errors).length > 0) {
      setFormErrors(errors);
      return;
    }

    setAddSubsidyLoading(true);
    try {
      // Prepare locations as an array
      const locationsArray = newSubsidy.locations.length ? 
        typeof newSubsidy.locations === 'string' ? 
          [newSubsidy.locations as unknown as string] : 
          newSubsidy.locations : 
        [];

      const subsidyPayload = {
        ...newSubsidy,
        locations: locationsArray
      };
      console.debug("Creating new subsidy:", subsidyPayload, "with admin email:", adminEmail);
      const createdSubsidy = await createSubsidy(subsidyPayload, adminEmail);
      console.debug("Subsidy created successfully:", createdSubsidy);
      
      // Add the newly created subsidy to the list
      setSubsidies(prev => [...prev, createdSubsidy]);
      
      // Reset form and close dialog
      setNewSubsidy({
        program: "",
        description: "",
        eligibility: "",
        type: "cash",
        benefits: "",
        application_process: "",
        locations: [],
        dynamic_fields: "{}",
        max_recipients: 0,
        provider: ""
      });
      setShowAddDialog(false);
    } catch (error) {
      console.error("Error adding subsidy:", error);
      setFormErrors({ submit: "Failed to create subsidy. Please try again." });
    } finally {
      setAddSubsidyLoading(false);
    }
  };

  // Filter subsidies based on search query, type, and status
  const filteredSubsidies = subsidies.filter(subsidy => {
    const matchesSearch = searchQuery === "" || 
      subsidy.type.toLowerCase().includes(searchQuery.toLowerCase()) ||
      subsidy.locations.some(loc => loc.toLowerCase().includes(searchQuery.toLowerCase()));
    const matchesType = typeFilter === "all" || subsidy.type === typeFilter;
    const matchesStatus = statusFilter === "all" || subsidy.status === statusFilter;
    return matchesSearch && matchesType && matchesStatus;
  });

  // Handle approve/reject actions
  const handleApproveSubsidy = async (id: string) => {
    console.debug("Initiating subsidy approval for ID:", id);
    setConfirmAction({ type: 'approve', id });
    setShowConfirmDialog(true);
  };

  const handleRejectSubsidy = async (id: string) => {
    console.debug("Initiating subsidy rejection for ID:", id);
    setConfirmAction({ type: 'reject', id });
    setShowConfirmDialog(true);
  };

  const confirmActionHandler = async () => {
    if (!confirmAction) return;

    try {
      const { type, id } = confirmAction;
      let updatedSubsidy;
      
      if (type === 'approve') {
        console.debug("Approving subsidy with ID:", id, "by admin:", adminEmail);
        updatedSubsidy = await approveSubsidy(id, adminEmail);
        console.debug("Subsidy approved successfully:", updatedSubsidy);
      } else {
        console.debug("Rejecting subsidy with ID:", id, "by admin:", adminEmail);
        updatedSubsidy = await rejectSubsidy(id, adminEmail);
        console.debug("Subsidy rejected successfully:", updatedSubsidy);
      }
      
      // Update the subsidies list with the updated subsidy
      setSubsidies(prev => 
        prev.map(subsidy => subsidy.$id === id ? updatedSubsidy : subsidy)
      );

    } catch (error) {
      console.error(`Error ${confirmAction.type}ing subsidy:`, error);
    } finally {
      setShowConfirmDialog(false);
      setConfirmAction(null);
    }
  };

  // Format date
  const formatDate = (dateString: string): string => {
    try {
      return format(parseISO(dateString), "MMM d, yyyy");
    } catch (e) {
      return dateString;
    }
  };

  // Get status badge
  const getStatusBadge = (status: string) => {
    switch (status) {
      case "pending":
      case "listed":
        return <Badge variant="outline" className="bg-yellow-50 text-yellow-800 border-yellow-300"><Clock className="h-3 w-3 mr-1" /> {status === "pending" ? "Pending" : "Listed"}</Badge>;
      case "approved":
      case "accepted":
        return <Badge variant="outline" className="bg-green-50 text-green-800 border-green-300"><CheckCircle className="h-3 w-3 mr-1" /> {status === "approved" ? "Approved" : "Accepted"}</Badge>;
      case "rejected":
        return <Badge variant="outline" className="bg-red-50 text-red-800 border-red-300"><XCircle className="h-3 w-3 mr-1" /> Rejected</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  // Handle location input for the form
  const handleLocationInput = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      const value = (e.target as HTMLInputElement).value.trim();
      if (value && !newSubsidy.locations.includes(value)) {
        setNewSubsidy(prev => ({
          ...prev,
          locations: [...prev.locations, value]
        }));
        (e.target as HTMLInputElement).value = '';
      }
    }
  };

  const removeLocation = (location: string) => {
    setNewSubsidy(prev => ({
      ...prev,
      locations: prev.locations.filter(loc => loc !== location)
    }));
  };

  return (
    <DashboardShell>
      <Header />
      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold tracking-tight">Subsidy Management</h1>
          <Button onClick={() => setShowAddDialog(true)}>
            <Plus className="h-4 w-4 mr-2" />
            Add Subsidy
          </Button>
        </div>

        <div className="flex flex-col gap-4 md:flex-row md:items-center">
          <Input
            placeholder="Search by type or location..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="md:w-80"
          />
          <Select value={typeFilter} onValueChange={setTypeFilter}>
            <SelectTrigger className="md:w-40">
              <SelectValue placeholder="Filter by type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All types</SelectItem>
              <SelectItem value="cash">Cash</SelectItem>
              <SelectItem value="asset">Asset</SelectItem>
              <SelectItem value="training">Training</SelectItem>
              <SelectItem value="loan">Loan</SelectItem>
            </SelectContent>
          </Select>
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="md:w-40">
              <SelectValue placeholder="Filter by status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All statuses</SelectItem>
              <SelectItem value="pending">Pending</SelectItem>
              <SelectItem value="approved">Approved</SelectItem>
              <SelectItem value="rejected">Rejected</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Tabs defaultValue="subsidies" className="space-y-4">
          <TabsList>
            <TabsTrigger value="subsidies">Subsidies</TabsTrigger>
            <TabsTrigger value="requests">Requests</TabsTrigger>
          </TabsList>
          <TabsContent value="subsidies" className="space-y-4">
            {loading ? (
              <div className="text-center py-8">Loading subsidies...</div>
            ) : filteredSubsidies.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">No subsidies found</div>
            ) : (
              filteredSubsidies.map((subsidy) => (
                <Card key={subsidy.$id} className="overflow-hidden">
                  <CardHeader className="pb-3 bg-muted/50">
                    <div className="flex flex-row items-start justify-between">
                      <div>
                        <CardTitle className="text-xl font-semibold">
                          {subsidy.program}
                        </CardTitle>
                        <div className="flex items-center gap-2 mt-1">
                          <Badge variant="outline" className="capitalize">{subsidy.type}</Badge>
                          <span className="text-sm text-muted-foreground">•</span>
                          <span className="text-sm text-muted-foreground">Provider: {subsidy.provider}</span>
                        </div>
                      </div>
                      {getStatusBadge(subsidy.status)}
                    </div>
                  </CardHeader>
                  <CardContent className="pt-4">
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="space-y-3">
                        <div>
                          <h4 className="text-sm font-medium text-muted-foreground">Description</h4>
                          <p className="mt-1">{subsidy.description}</p>
                        </div>
                        
                        <div>
                          <h4 className="text-sm font-medium text-muted-foreground">Benefits</h4>
                          <p className="mt-1">{subsidy.benefits}</p>
                        </div>
                        
                        <div>
                          <h4 className="text-sm font-medium text-muted-foreground">Locations</h4>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {subsidy.locations.map((location, idx) => (
                              <Badge key={idx} variant="secondary">{location}</Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-3">
                        <div>
                          <h4 className="text-sm font-medium text-muted-foreground">Eligibility</h4>
                          <p className="mt-1">{subsidy.eligibility}</p>
                        </div>
                        
                        <div>
                          <h4 className="text-sm font-medium text-muted-foreground">Application Process</h4>
                          <p className="mt-1">{subsidy.application_process}</p>
                        </div>
                        
                        <div className="flex gap-4">
                          <div>
                            <h4 className="text-sm font-medium text-muted-foreground">Max Recipients</h4>
                            <p className="mt-1">{subsidy.max_recipients}</p>
                          </div>
                          
                          {subsidy.recipients_accepted !== undefined && (
                            <div>
                              <h4 className="text-sm font-medium text-muted-foreground">Current Recipients</h4>
                              <p className="mt-1">{subsidy.recipients_accepted}</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    <div className="mt-4 pt-3 border-t text-xs text-muted-foreground">
                      Created on {formatDate(subsidy.$createdAt)} • ID: {subsidy.$id.substring(0, 8)}
                      {subsidy.submitted_by && ` • Submitted by: ${subsidy.submitted_by}`}
                    </div>
                  </CardContent>
                  {subsidy.status === "pending" && (
                    <CardFooter className="flex justify-end gap-3 bg-muted/10 px-6 py-4 border-t border-muted">
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => handleRejectSubsidy(subsidy.$id)}
                        className="text-red-600 border-red-200 hover:bg-red-50 hover:text-red-700 hover:border-red-300 transition-colors font-medium px-4"
                      >
                        <ThumbsDown className="h-3.5 w-3.5 mr-2" />
                        Reject
                      </Button>
                      <Button 
                        variant="default" 
                        size="sm"
                        onClick={() => handleApproveSubsidy(subsidy.$id)}
                        className="bg-green-600 hover:bg-green-700 text-white font-medium px-4 transition-colors"
                      >
                        <ThumbsUp className="h-3.5 w-3.5 mr-2" />
                        Approve
                      </Button>
                    </CardFooter>
                  )}
                </Card>
              ))
            )}
          </TabsContent>
          
          <TabsContent value="requests" className="space-y-4">
            {loading ? (
              <div className="text-center py-8">Loading requests...</div>
            ) : subsidyRequests.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">No subsidy requests found</div>
            ) : (
              subsidyRequests.map((request) => {
                const relatedSubsidy = subsidies.find(s => s.$id === request.subsidy_id);
                return (
                  <Card key={request.$id}>
                    <CardHeader className="pb-3">
                      <div className="flex justify-between items-start">
                        <div>
                          <CardTitle className="text-base font-medium">
                            Request for {relatedSubsidy?.program || "Subsidy"}
                          </CardTitle>
                          <CardDescription>
                            From farmer {request.farmer_id.substring(0, 8)} • {formatDate(request.$createdAt)}
                          </CardDescription>
                        </div>
                        {getStatusBadge(request.status)}
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="grid gap-4 md:grid-cols-2">
                        {relatedSubsidy && (
                          <>
                            <div>
                              <h4 className="text-sm font-medium text-muted-foreground">Subsidy Type</h4>
                              <p className="mt-1 capitalize">{relatedSubsidy.type}</p>
                            </div>
                            <div>
                              <h4 className="text-sm font-medium text-muted-foreground">Requested Benefits</h4>
                              <p className="mt-1">{relatedSubsidy.benefits}</p>
                            </div>
                          </>
                        )}
                        <div className="md:col-span-2 text-xs text-muted-foreground">
                          Request ID: {request.$id}
                          {relatedSubsidy && ` • Subsidy ID: ${relatedSubsidy.$id}`}
                        </div>
                      </div>
                    </CardContent>
                    {request.status === "requested" && (
                      <CardFooter className="flex justify-end gap-2 bg-muted/20 pt-3">
                        <Button 
                          variant="destructive" 
                          size="sm"
                          onClick={async () => {
                            try {
                              console.debug("Rejecting subsidy request with ID:", request.$id, "by admin:", adminEmail);
                              const result = await rejectSubsidyRequest(request.$id, adminEmail);
                              console.debug("Subsidy request rejected successfully:", result);
                              setSubsidyRequests(prev =>
                                prev.map(r => r.$id === request.$id ? {...r, status: "rejected"} : r)
                              );
                            } catch (error) {
                              console.error("Error rejecting request:", error);
                            }
                          }}
                        >
                          <ThumbsDown className="h-4 w-4 mr-2" />
                          Reject
                        </Button>
                        <Button 
                          variant="default" 
                          size="sm"
                          onClick={async () => {
                            try {
                              console.debug("Accepting subsidy request with ID:", request.$id, "by admin:", adminEmail);
                              const result = await acceptSubsidyRequest(request.$id, adminEmail);
                              console.debug("Subsidy request accepted successfully:", result);
                              setSubsidyRequests(prev =>
                                prev.map(r => r.$id === request.$id ? {...r, status: "accepted"} : r)
                              );
                            } catch (error) {
                              console.error("Error accepting request:", error);
                            }
                          }}
                        >
                          <ThumbsUp className="h-4 w-4 mr-2" />
                          Accept
                        </Button>
                      </CardFooter>
                    )}
                  </Card>
                );
              })
            )}
          </TabsContent>
        </Tabs>
      </main>

      {/* Confirmation Dialog */}
      <AlertDialog open={showConfirmDialog} onOpenChange={setShowConfirmDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {confirmAction?.type === 'approve' ? 'Approve Subsidy' : 'Reject Subsidy'}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {confirmAction?.type === 'approve'
                ? 'Are you sure you want to approve this subsidy? This action cannot be undone.'
                : 'Are you sure you want to reject this subsidy? This action cannot be undone.'}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={confirmActionHandler}>
              {confirmAction?.type === 'approve' ? 'Approve' : 'Reject'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Add Subsidy Dialog */}
      <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
        <DialogContent className="sm:max-w-[600px] max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Add New Subsidy</DialogTitle>
            <DialogDescription>
              Create a new subsidy program to support farmers
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="program" className="text-right">
                Program Name
              </Label>
              <Input
                id="program"
                value={newSubsidy.program}
                onChange={(e) => setNewSubsidy({...newSubsidy, program: e.target.value})}
                className="col-span-3"
              />
              {formErrors.program && <div className="col-span-4 text-right text-red-500 text-sm">{formErrors.program}</div>}
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="type" className="text-right">
                Type
              </Label>
              <Select
                value={newSubsidy.type}
                onValueChange={(value) => setNewSubsidy({...newSubsidy, type: value})}
              >
                <SelectTrigger className="col-span-3">
                  <SelectValue placeholder="Select type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="cash">Cash</SelectItem>
                  <SelectItem value="asset">Asset</SelectItem>
                  <SelectItem value="training">Training</SelectItem>
                  <SelectItem value="loan">Loan</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="provider" className="text-right">
                Provider
              </Label>
              <Input
                id="provider"
                value={newSubsidy.provider}
                onChange={(e) => setNewSubsidy({...newSubsidy, provider: e.target.value})}
                className="col-span-3"
              />
              {formErrors.provider && <div className="col-span-4 text-right text-red-500 text-sm">{formErrors.provider}</div>}
            </div>
            <div className="grid grid-cols-4 items-start gap-4">
              <Label htmlFor="description" className="text-right pt-2">
                Description
              </Label>
              <Textarea
                id="description"
                value={newSubsidy.description}
                onChange={(e) => setNewSubsidy({...newSubsidy, description: e.target.value})}
                className="col-span-3"
                rows={3}
              />
              {formErrors.description && <div className="col-span-4 text-right text-red-500 text-sm">{formErrors.description}</div>}
            </div>
            <div className="grid grid-cols-4 items-start gap-4">
              <Label htmlFor="eligibility" className="text-right pt-2">
                Eligibility
              </Label>
              <Textarea
                id="eligibility"
                value={newSubsidy.eligibility}
                onChange={(e) => setNewSubsidy({...newSubsidy, eligibility: e.target.value})}
                className="col-span-3"
                rows={2}
              />
              {formErrors.eligibility && <div className="col-span-4 text-right text-red-500 text-sm">{formErrors.eligibility}</div>}
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="benefits" className="text-right">
                Benefits
              </Label>
              <Input
                id="benefits"
                value={newSubsidy.benefits}
                onChange={(e) => setNewSubsidy({...newSubsidy, benefits: e.target.value})}
                className="col-span-3"
                placeholder="e.g., 10000 cash, 2 goats, etc."
              />
              {formErrors.benefits && <div className="col-span-4 text-right text-red-500 text-sm">{formErrors.benefits}</div>}
            </div>
            <div className="grid grid-cols-4 items-start gap-4">
              <Label htmlFor="application_process" className="text-right pt-2">
                Application Process
              </Label>
              <Textarea
                id="application_process"
                value={newSubsidy.application_process}
                onChange={(e) => setNewSubsidy({...newSubsidy, application_process: e.target.value})}
                className="col-span-3"
                rows={2}
              />
              {formErrors.application_process && <div className="col-span-4 text-right text-red-500 text-sm">{formErrors.application_process}</div>}
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="max_recipients" className="text-right">
                Max Recipients
              </Label>
              <Input
                id="max_recipients"
                type="number"
                value={newSubsidy.max_recipients || ""}
                onChange={(e) => setNewSubsidy({...newSubsidy, max_recipients: parseInt(e.target.value) || 0})}
                className="col-span-3"
                min={1}
              />
              {formErrors.max_recipients && <div className="col-span-4 text-right text-red-500 text-sm">{formErrors.max_recipients}</div>}
            </div>
            <div className="grid grid-cols-4 items-start gap-4">
              <Label htmlFor="locations" className="text-right pt-2">
                Locations
              </Label>
              <div className="col-span-3">
                <Input
                  id="locations"
                  placeholder="Type location and press Enter"
                  onKeyDown={handleLocationInput}
                  className="mb-2"
                />
                <div className="flex flex-wrap gap-1 mt-2">
                  {newSubsidy.locations.map((location, index) => (
                    <Badge key={index} variant="secondary" className="flex items-center gap-1">
                      {location}
                      <button 
                        type="button" 
                        onClick={() => removeLocation(location)}
                        className="ml-1 text-xs hover:bg-gray-200 rounded-full h-4 w-4 inline-flex items-center justify-center"
                      >
                        ×
                      </button>
                    </Badge>
                  ))}
                </div>
              </div>
              {formErrors.locations && <div className="col-span-4 text-right text-red-500 text-sm">{formErrors.locations}</div>}
            </div>
            <div className="grid grid-cols-4 items-start gap-4">
              <Label htmlFor="dynamic_fields" className="text-right pt-2">
                Additional Fields (JSON)
              </Label>
              <Textarea
                id="dynamic_fields"
                value={newSubsidy.dynamic_fields}
                onChange={(e) => setNewSubsidy({...newSubsidy, dynamic_fields: e.target.value})}
                className="col-span-3 font-mono text-sm"
                rows={4}
                placeholder='{"field1": "value1", "field2": "value2"}'
              />
            </div>
            {formErrors.submit && <div className="text-center text-red-500">{formErrors.submit}</div>}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowAddDialog(false)}>Cancel</Button>
            <Button onClick={handleAddSubsidy} disabled={addSubsidyLoading}>
              {addSubsidyLoading ? "Creating..." : "Create Subsidy"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </DashboardShell>
  );
}

"use client"

import { useEffect, useState } from "react"
import { getSupabaseClient } from "@/lib/supabaseClient";
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Briefcase, MapPin, Building, Globe, DollarSign, Calendar, Plus, Edit, Trash2 } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"



import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import Link from "next/link"
import { Header } from "@/components/header"

interface JobPost {
  id: string
  title: string
  content: string
  created_at: string
  author_name: string
  avatar_url?: string
  user_id: string
  job_metadata?: {
    company_name: string
    location: string
    is_remote: boolean
    job_type: string
    salary_range?: string
    deadline?: string
    application_link?: string
    accessibility_features?: any
  }
}

export default function JobsPage() {
  const [jobPosts, setJobPosts] = useState<JobPost[]>([])
  const [loading, setLoading] = useState(true)
  const [currentUserId, setCurrentUserId] = useState<string | null>(null)
  const [editingJobId, setEditingJobId] = useState<string | null>(null)
  const [editTitle, setEditTitle] = useState("")
  const [editContent, setEditContent] = useState("")
  const [editCompanyName, setEditCompanyName] = useState("")
  const [editLocation, setEditLocation] = useState("")
  const [editIsRemote, setEditIsRemote] = useState(false)
  const [editJobType, setEditJobType] = useState("")
  const [editSalaryRange, setEditSalaryRange] = useState("")
  const [editDeadline, setEditDeadline] = useState("")
  const [editApplicationLink, setEditApplicationLink] = useState("")
  const [editAccessibilityFeatures, setEditAccessibilityFeatures] = useState({
    screenReaderSupport: false,
    signLanguageSupport: false,
    remoteWork: false,
    flexibleHours: false,
    assistiveTechnology: false,
    accessibleOffice: false
  })
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null)
  const [mounted, setMounted] = useState(false)
  const [editJobLoading, setEditJobLoading] = useState(false);
  const [editJobError, setEditJobError] = useState<string | null>(null);

  
  useEffect(() => {
    setMounted(true);
    fetchCurrentUser()
    fetchJobPosts()
  }, [])

  const fetchCurrentUser = async () => {
    try {
      const supabase = getSupabaseClient();
      const { data: { user } } = await supabase.auth.getUser();
      setCurrentUserId(user?.id || null);
    } catch (error) {
      console.error('Error fetching current user:', error);
    }
  }

  const fetchJobPosts = async () => {
    try {
      const supabase = getSupabaseClient();
      const { data: postsData, error: postsError } = await supabase
        .from('posts')
        .select('*')
        .eq('is_job_post', true)
        .order('created_at', { ascending: false })

      if (postsError) throw postsError

      // Fetch profiles for all users
      const userIds = Array.from(new Set(postsData?.map(post => post.user_id) || []))
      const { data: profilesData, error: profilesError } = await supabase
        .from('profiles')
        .select('id, first_name, last_name, avatar_url')
        .in('id', userIds)

      if (profilesError) throw profilesError

      // Create a map of user profiles
      const profilesMap = new Map()
      profilesData?.forEach(profile => {
        profilesMap.set(profile.id, profile)
      })

      // Transform the data
      const transformedJobPosts: JobPost[] = postsData?.map((post: any) => {
        const profile = profilesMap.get(post.user_id)
        return {
          id: post.id,
          title: post.title,
          content: post.content,
          created_at: post.created_at,
          author_name: profile ? `${profile.first_name} ${profile.last_name}`.trim() : 'Anonymous',
          avatar_url: profile?.avatar_url,
          user_id: post.user_id,
          job_metadata: post.job_metadata || {}
        }
      }) || []

      setJobPosts(transformedJobPosts)
    } catch (error) {
      console.error('Error fetching job posts:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleEditJob = async (jobId: string) => {
    if (!editTitle.trim() || !editContent.trim()) return;
    setEditJobLoading(true);
    setEditJobError(null);
    try {
      const supabase = getSupabaseClient();
      const { error } = await supabase
        .from('posts')
        .update({ 
          title: editTitle.trim(),
          content: editContent.trim(),
          job_metadata: {
            company_name: editCompanyName.trim(),
            location: editLocation.trim(),
            is_remote: editIsRemote,
            job_type: editJobType.trim(),
            salary_range: editSalaryRange.trim(),
            deadline: editDeadline.trim(),
            application_link: editApplicationLink.trim(),
            accessibility_features: editAccessibilityFeatures
          },
          updated_at: new Date().toISOString()
        })
        .eq('id', jobId);
      if (!error) {
        setEditingJobId(null);
        setEditTitle('');
        setEditContent('');
        setEditCompanyName('');
        setEditLocation('');
        setEditIsRemote(false);
        setEditJobType('');
        setEditSalaryRange('');
        setEditDeadline('');
        setEditApplicationLink('');
        setEditAccessibilityFeatures({
          screenReaderSupport: false,
          signLanguageSupport: false,
          remoteWork: false,
          flexibleHours: false,
          assistiveTechnology: false,
          accessibleOffice: false
        });
        fetchJobPosts(); // Refresh the list
      } else {
        setEditJobError(error.message || 'Error updating job post.');
      }
    } catch (error: any) {
      setEditJobError(error.message || 'Error updating job post.');
    } finally {
      setEditJobLoading(false);
    }
  };

  const handleDeleteJob = async (jobId: string) => {
    try {
      const supabase = getSupabaseClient();
      const { error } = await supabase
        .from('posts')
        .delete()
        .eq('id', jobId);
      
      if (!error) {
        setShowDeleteConfirm(null);
        fetchJobPosts(); // Refresh the list
      }
    } catch (error) {
      console.error('Error deleting job:', error);
    }
  };

  const startEditJob = (job: JobPost) => {
    setEditingJobId(job.id);
    setEditTitle(job.title);
    setEditContent(job.content);
    setEditCompanyName(job.job_metadata?.company_name || '');
    setEditLocation(job.job_metadata?.location || '');
    setEditIsRemote(job.job_metadata?.is_remote || false);
    setEditJobType(job.job_metadata?.job_type || '');
    setEditSalaryRange(job.job_metadata?.salary_range || '');
    setEditDeadline(job.job_metadata?.deadline || '');
    setEditApplicationLink(job.job_metadata?.application_link || '');
    setEditAccessibilityFeatures(job.job_metadata?.accessibility_features || {
      screenReaderSupport: false,
      signLanguageSupport: false,
      remoteWork: false,
      flexibleHours: false,
      assistiveTechnology: false,
      accessibleOffice: false
    });
  };

  const cancelEditJob = () => {
    setEditingJobId(null);
    setEditTitle('');
    setEditContent('');
    setEditCompanyName('');
    setEditLocation('');
    setEditIsRemote(false);
    setEditJobType('');
    setEditSalaryRange('');
    setEditDeadline('');
    setEditApplicationLink('');
    setEditAccessibilityFeatures({
      screenReaderSupport: false,
      signLanguageSupport: false,
      remoteWork: false,
      flexibleHours: false,
      assistiveTechnology: false,
      accessibleOffice: false
    });
  };

  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60))
    
    if (diffInHours < 1) return 'Just now'
    if (diffInHours < 24) return `${diffInHours} hour${diffInHours > 1 ? 's' : ''} ago`
    
    const diffInDays = Math.floor(diffInHours / 24)
    if (diffInDays < 7) return `${diffInDays} day${diffInDays > 1 ? 's' : ''} ago`
    
    const diffInWeeks = Math.floor(diffInDays / 7)
    if (diffInWeeks < 4) return `${diffInWeeks} week${diffInWeeks > 1 ? 's' : ''} ago`
    
    return date.toLocaleDateString()
  }

  if (!mounted) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <main id="main-content" className="max-w-6xl mx-auto px-4 py-8">
          <div className="text-center py-8 text-muted-foreground">Loading...</div>
        </main>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <main id="main-content" className="max-w-6xl mx-auto px-4 py-8">
          <div className="text-center py-8 text-muted-foreground">Loading jobs...</div>
        </main>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main id="main-content" className="container-responsive max-w-6xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-foreground mb-2">Jobs</h1>
            <p className="text-muted-foreground">Find accessible job opportunities</p>
          </div>
          <Link href="/jobs/post" className="mt-4 sm:mt-0">
            <Button className="bg-mustard hover:bg-forest-green text-white">
              <Plus className="h-4 w-4 mr-2" />
              Post a Job
            </Button>
          </Link>
        </div>

        {/* Job Posts */}
        <div className="space-y-6">
          {jobPosts.length === 0 ? (
            <Card className="bg-card">
              <CardContent className="text-center py-12">
                <Briefcase className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-foreground mb-2">No jobs posted yet</h3>
                <p className="text-muted-foreground mb-4">Be the first to share an accessible job opportunity!</p>
                <Link href="/jobs/post">
                  <Button className="bg-mustard hover:bg-forest-green text-white">
                    <Plus className="h-4 w-4 mr-2" />
                    Post the First Job
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ) : (
            jobPosts.map((job) => (
              <Card key={job.id} className="hover:shadow-md transition-shadow bg-card">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-4 flex-1 min-w-0">
                      <Avatar className="h-12 w-12 ring-2 ring-mustard flex-shrink-0">
                        <AvatarImage src={job.avatar_url} alt={job.author_name} />
                        <AvatarFallback className="bg-gradient-to-br from-mustard to-forest-green text-white">
                          {job.author_name.split(' ').map(n => n[0]).join('')}
                        </AvatarFallback>
                      </Avatar>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2 mb-1">
                          <h3 className="text-lg font-semibold text-foreground truncate">{job.title}</h3>
                          <Badge className="bg-green-100 text-green-800">Job</Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mb-2">{job.author_name} • {formatTimeAgo(job.created_at)}</p>
                        {job.job_metadata?.company_name && (
                          <p className="text-sm text-muted-foreground flex items-center">
                            <Building className="h-3 w-3 mr-1" />
                            {job.job_metadata.company_name}
                          </p>
                        )}
                      </div>
                    </div>
                    {/* Edit/Delete buttons for job owner */}
                    {currentUserId === job.user_id && (
                      <div className="flex items-center gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => startEditJob(job)}
                          aria-label="Edit job"
                        >
                          <Edit className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setShowDeleteConfirm(job.id)}
                          aria-label="Delete job"
                          className="text-destructive hover:text-destructive/80"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    )}
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                                         {editingJobId === job.id ? (
                       <div className="space-y-4">
                         {/* Basic Job Information */}
                         <div className="space-y-3">
                           <Input
                             value={editTitle}
                             onChange={(e) => setEditTitle(e.target.value)}
                             placeholder="Job title"
                             className="w-full bg-background border border-input text-foreground"
                             disabled={editJobLoading}
                           />
                           <Textarea
                             value={editContent}
                             onChange={(e) => setEditContent(e.target.value)}
                             placeholder="Job description"
                             className="w-full bg-background border border-input text-foreground"
                             rows={4}
                             disabled={editJobLoading}
                           />
                         </div>

                         {/* Company Information */}
                         <div className="space-y-3">
                           <h4 className="font-medium text-foreground">Company Information</h4>
                           <Input
                             value={editCompanyName}
                             onChange={(e) => setEditCompanyName(e.target.value)}
                             placeholder="Company name"
                             className="w-full bg-background border border-input text-foreground"
                           />
                           <Input
                             value={editLocation}
                             onChange={(e) => setEditLocation(e.target.value)}
                             placeholder="Location"
                             className="w-full bg-background border border-input text-foreground"
                           />
                           <div className="flex items-center space-x-2">
                             <Switch
                               id="remote-work"
                               checked={editIsRemote}
                               onCheckedChange={setEditIsRemote}
                             />
                             <label htmlFor="remote-work" className="text-sm font-medium text-foreground">
                               Remote work available
                             </label>
                           </div>
                         </div>

                         {/* Job Details */}
                         <div className="space-y-3">
                           <h4 className="font-medium text-foreground">Job Details</h4>
                           <Select value={editJobType} onValueChange={setEditJobType}>
                             <SelectTrigger>
                               <SelectValue placeholder="Select job type" />
                             </SelectTrigger>
                             <SelectContent>
                               <SelectItem value="Full-time">Full-time</SelectItem>
                               <SelectItem value="Part-time">Part-time</SelectItem>
                               <SelectItem value="Contract">Contract</SelectItem>
                               <SelectItem value="Internship">Internship</SelectItem>
                               <SelectItem value="Freelance">Freelance</SelectItem>
                             </SelectContent>
                           </Select>
                           <Input
                             value={editSalaryRange}
                             onChange={(e) => setEditSalaryRange(e.target.value)}
                             placeholder="Salary range (e.g., $50,000 - $70,000)"
                             className="w-full"
                           />
                           <Input
                             type="date"
                             value={editDeadline}
                             onChange={(e) => setEditDeadline(e.target.value)}
                             className="w-full"
                           />
                           <Input
                             value={editApplicationLink}
                             onChange={(e) => setEditApplicationLink(e.target.value)}
                             placeholder="Application link (URL)"
                             className="w-full"
                           />
                         </div>

                         {/* Accessibility Features */}
                         <div className="space-y-3">
                           <h4 className="font-medium text-foreground">Accessibility Features</h4>
                           <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                             <div className="flex items-center space-x-2">
                               <Switch
                                 id="screen-reader"
                                 checked={editAccessibilityFeatures.screenReaderSupport}
                                 onCheckedChange={(checked) => setEditAccessibilityFeatures(prev => ({
                                   ...prev,
                                   screenReaderSupport: checked
                                 }))}
                               />
                               <label htmlFor="screen-reader" className="text-sm text-foreground">
                                 Screen Reader Support
                               </label>
                             </div>
                             <div className="flex items-center space-x-2">
                               <Switch
                                 id="sign-language"
                                 checked={editAccessibilityFeatures.signLanguageSupport}
                                 onCheckedChange={(checked) => setEditAccessibilityFeatures(prev => ({
                                   ...prev,
                                   signLanguageSupport: checked
                                 }))}
                               />
                               <label htmlFor="sign-language" className="text-sm text-foreground">
                                 Sign Language Support
                               </label>
                             </div>
                             <div className="flex items-center space-x-2">
                               <Switch
                                 id="remote-work-feature"
                                 checked={editAccessibilityFeatures.remoteWork}
                                 onCheckedChange={(checked) => setEditAccessibilityFeatures(prev => ({
                                   ...prev,
                                   remoteWork: checked
                                 }))}
                               />
                               <label htmlFor="remote-work-feature" className="text-sm text-foreground">
                                 Remote Work
                               </label>
                             </div>
                             <div className="flex items-center space-x-2">
                               <Switch
                                 id="flexible-hours"
                                 checked={editAccessibilityFeatures.flexibleHours}
                                 onCheckedChange={(checked) => setEditAccessibilityFeatures(prev => ({
                                   ...prev,
                                   flexibleHours: checked
                                 }))}
                               />
                               <label htmlFor="flexible-hours" className="text-sm text-foreground">
                                 Flexible Hours
                               </label>
                             </div>
                             <div className="flex items-center space-x-2">
                               <Switch
                                 id="assistive-tech"
                                 checked={editAccessibilityFeatures.assistiveTechnology}
                                 onCheckedChange={(checked) => setEditAccessibilityFeatures(prev => ({
                                   ...prev,
                                   assistiveTechnology: checked
                                 }))}
                               />
                               <label htmlFor="assistive-tech" className="text-sm text-foreground">
                                 Assistive Technology
                               </label>
                             </div>
                             <div className="flex items-center space-x-2">
                               <Switch
                                 id="accessible-office"
                                 checked={editAccessibilityFeatures.accessibleOffice}
                                 onCheckedChange={(checked) => setEditAccessibilityFeatures(prev => ({
                                   ...prev,
                                   accessibleOffice: checked
                                 }))}
                               />
                               <label htmlFor="accessible-office" className="text-sm text-foreground">
                                 Accessible Office
                               </label>
                             </div>
                           </div>
                         </div>

                         {/* Action Buttons */}
                         <div className="flex space-x-2 pt-4">
                           <Button onClick={() => handleEditJob(job.id)} size="sm" disabled={editJobLoading}>
                             {editJobLoading ? 'Saving...' : 'Save Changes'}
                           </Button>
                           <Button variant="outline" size="sm" onClick={cancelEditJob} disabled={editJobLoading}>
                             Cancel
                           </Button>
                         </div>
                       </div>
                     ) : (
                       <p className="text-foreground whitespace-pre-wrap">{job.content}</p>
                     )}
                    
                    {/* Job Details */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
                      {job.job_metadata?.location && !job.job_metadata?.is_remote && (
                        <div className="flex items-center text-muted-foreground">
                          <MapPin className="h-4 w-4 mr-2" />
                          {job.job_metadata.location}
                        </div>
                      )}
                      {job.job_metadata?.is_remote && (
                        <div className="flex items-center text-muted-foreground">
                          <Globe className="h-4 w-4 mr-2" />
                          Remote
                        </div>
                      )}
                      {job.job_metadata?.job_type && (
                        <div className="flex items-center text-muted-foreground">
                          <Briefcase className="h-4 w-4 mr-2" />
                          {job.job_metadata.job_type}
                        </div>
                      )}
                      {job.job_metadata?.salary_range && (
                        <div className="flex items-center text-muted-foreground">
                          <DollarSign className="h-4 w-4 mr-2" />
                          {job.job_metadata.salary_range}
                        </div>
                      )}
                    </div>

                    {/* Accessibility Features */}
                    {job.job_metadata?.accessibility_features && (
                      <div className="mt-4">
                        <h4 className="text-sm font-medium text-foreground mb-2">Accessibility Features:</h4>
                        <div className="flex flex-wrap gap-2">
                          {Object.entries(job.job_metadata.accessibility_features).map(([feature, enabled]) => {
                            if (enabled) {
                              const featureLabels: { [key: string]: string } = {
                                screenReaderSupport: 'Screen Reader Support',
                                signLanguageSupport: 'Sign Language Support',
                                remoteWork: 'Remote Work',
                                flexibleHours: 'Flexible Hours',
                                assistiveTechnology: 'Assistive Technology',
                                accessibleOffice: 'Accessible Office'
                              }
                              return (
                                <Badge key={feature} variant="secondary" className="bg-mustard/10 text-mustard">
                                  {featureLabels[feature] || feature}
                                </Badge>
                              )
                            }
                            return null
                          })}
                        </div>
                      </div>
                    )}

                    {/* Application Link */}
                    {job.job_metadata?.application_link && (
                      <div className="mt-4">
                        <a
                          href={job.job_metadata.application_link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center text-mustard hover:text-forest-green font-medium"
                        >
                          Apply for this position →
                        </a>
                      </div>
                    )}
                  </div>
                </CardContent>
                
                {/* Delete Confirmation */}
                {showDeleteConfirm === job.id && (
                  <div className="bg-destructive/10 border border-destructive/20 p-4 rounded-lg m-4">
                    <p className="text-destructive mb-3">Are you sure you want to delete this job post?</p>
                    <div className="flex space-x-2">
                      <Button 
                        variant="destructive" 
                        size="sm" 
                        onClick={() => handleDeleteJob(job.id)}
                      >
                        Delete
                      </Button>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={() => setShowDeleteConfirm(null)}
                      >
                        Cancel
                      </Button>
                    </div>
                  </div>
                )}
              </Card>
            ))
          )}
        </div>
      </main>
    </div>
  )
} 
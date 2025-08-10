"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, Briefcase, Building, MapPin, Globe, DollarSign, Calendar, Accessibility } from "lucide-react"
import { getSupabaseClient } from "@/lib/supabaseClient";
import { filterContent, hasBadWords } from "@/lib/content-filter"
import Link from "next/link"
import { Header } from "@/components/header"

export default function PostJobPage() {
  const [formData, setFormData] = useState({
    jobTitle: "",
    description: "",
    companyName: "",
    location: "",
    isRemote: false,
    jobType: "",
    applicationLink: "",
    salaryRange: "",
    deadline: "",
  })
  
  const [accessibilityFeatures, setAccessibilityFeatures] = useState({
    screenReaderSupport: false,
    signLanguageSupport: false,
    remoteWork: false,
    flexibleHours: false,
    assistiveTechnology: false,
    accessibleOffice: false,
  })
  
  const [loading, setLoading] = useState(false)

  const handleInputChange = (field: string, value: string | boolean) => {
    setFormData(prev => ({ ...prev, [field]: value }))
  }

  const handleAccessibilityChange = (feature: string, enabled: boolean) => {
    setAccessibilityFeatures(prev => ({ ...prev, [feature]: enabled }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)

    const supabase = getSupabaseClient();
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) {
      alert("You must be logged in to post a job.")
      setLoading(false)
      return
    }

    // Check for bad words
    if (hasBadWords(formData.jobTitle) || hasBadWords(formData.description) || hasBadWords(formData.companyName)) {
      alert("Your job post contains inappropriate language. Please review and edit your content.")
      setLoading(false)
      return
    }

    // Filter content
    const filteredData = {
      jobTitle: filterContent(formData.jobTitle),
      description: filterContent(formData.description),
      companyName: filterContent(formData.companyName),
    }

    try {
      const { error } = await supabase.from("posts").insert([{
        user_id: user.id,
        title: filteredData.jobTitle,
        content: filteredData.description,
        is_job_post: true,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        likes: 0,
        comments: 0,
        shares: 0,
        tags: [],
        // Job-specific metadata
        job_metadata: {
          company_name: filteredData.companyName,
          location: formData.location,
          is_remote: formData.isRemote,
          job_type: formData.jobType,
          application_link: formData.applicationLink,
          salary_range: formData.salaryRange,
          deadline: formData.deadline,
          accessibility_features: accessibilityFeatures,
        }
      }])

      if (error) throw error

      alert("Job posted successfully!")
      // Reset form
      setFormData({
        jobTitle: "",
        description: "",
        companyName: "",
        location: "",
        isRemote: false,
        jobType: "",
        applicationLink: "",
        salaryRange: "",
        deadline: "",
      })
      setAccessibilityFeatures({
        screenReaderSupport: false,
        signLanguageSupport: false,
        remoteWork: false,
        flexibleHours: false,
        assistiveTechnology: false,
        accessibleOffice: false,
      })
    } catch (error) {
      console.error("Error posting job:", error)
      alert("Error posting job. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-cream">
      <Header />
      <main id="main-content" className="container-responsive max-w-4xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link href="/jobs" className="inline-flex items-center text-mustard hover:text-forest-green mb-4">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Jobs
          </Link>
          <h1 className="text-3xl font-bold text-charcoal mb-2">Post a Job</h1>
          <p className="text-gray-600">Share job opportunities that are accessible and inclusive for people with disabilities.</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Basic Job Information */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Briefcase className="h-5 w-5 mr-2 text-mustard" />
                Job Information
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="jobTitle">Job Title *</label>
                  <Input
                    id="jobTitle"
                    value={formData.jobTitle}
                    onChange={(e) => handleInputChange("jobTitle", e.target.value)}
                    placeholder="e.g., Software Engineer"
                    required
                    aria-label="Job title"
                  />
                </div>
                <div>
                  <label htmlFor="companyName">Company Name *</label>
                  <Input
                    id="companyName"
                    value={formData.companyName}
                    onChange={(e) => handleInputChange("companyName", e.target.value)}
                    placeholder="e.g., TechCorp Inc."
                    required
                    aria-label="Company name"
                  />
                </div>
              </div>

              <div>
                <label htmlFor="description">Job Description *</label>
                <Textarea
                  id="description"
                  value={formData.description}
                  onChange={(e) => handleInputChange("description", e.target.value)}
                  placeholder="Describe the role, responsibilities, and requirements..."
                  rows={6}
                  required
                  aria-label="Job description"
                />
              </div>
            </CardContent>
          </Card>

          {/* Job Details */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Building className="h-5 w-5 mr-2 text-mustard" />
                Job Details
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="location">Location</label>
                  <Input
                    id="location"
                    value={formData.location}
                    onChange={(e) => handleInputChange("location", e.target.value)}
                    placeholder="e.g., New York, NY"
                    disabled={formData.isRemote}
                    aria-label="Job location"
                  />
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="isRemote"
                    checked={formData.isRemote}
                    onCheckedChange={(checked) => handleInputChange("isRemote", checked)}
                  />
                  <label htmlFor="isRemote">Remote position</label>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="jobType">Job Type *</label>
                  <Select value={formData.jobType} onValueChange={(value) => handleInputChange("jobType", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select job type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="full-time">Full-time</SelectItem>
                      <SelectItem value="part-time">Part-time</SelectItem>
                      <SelectItem value="contract">Contract</SelectItem>
                      <SelectItem value="internship">Internship</SelectItem>
                      <SelectItem value="freelance">Freelance</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <label htmlFor="salaryRange">Salary Range (Optional)</label>
                  <Input
                    id="salaryRange"
                    value={formData.salaryRange}
                    onChange={(e) => handleInputChange("salaryRange", e.target.value)}
                    placeholder="e.g., $50,000 - $70,000"
                    aria-label="Salary range"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="applicationLink">Application Link *</label>
                  <Input
                    id="applicationLink"
                    value={formData.applicationLink}
                    onChange={(e) => handleInputChange("applicationLink", e.target.value)}
                    placeholder="https://company.com/apply"
                    type="url"
                    required
                    aria-label="Application link"
                  />
                </div>
                <div>
                  <label htmlFor="deadline">Application Deadline (Optional)</label>
                  <Input
                    id="deadline"
                    value={formData.deadline}
                    onChange={(e) => handleInputChange("deadline", e.target.value)}
                    type="date"
                    aria-label="Application deadline"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Accessibility Features */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Accessibility className="h-5 w-5 mr-2 text-mustard" />
                Accessibility Features
              </CardTitle>
              <p className="text-sm text-gray-600">Select the accessibility accommodations your company provides</p>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex items-center space-x-2">
                  <Switch
                    id="screenReaderSupport"
                    checked={accessibilityFeatures.screenReaderSupport}
                    onCheckedChange={(checked) => handleAccessibilityChange("screenReaderSupport", checked)}
                  />
                  <label htmlFor="screenReaderSupport">Screen Reader Support</label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="signLanguageSupport"
                    checked={accessibilityFeatures.signLanguageSupport}
                    onCheckedChange={(checked) => handleAccessibilityChange("signLanguageSupport", checked)}
                  />
                  <label htmlFor="signLanguageSupport">Sign Language Support</label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="remoteWork"
                    checked={accessibilityFeatures.remoteWork}
                    onCheckedChange={(checked) => handleAccessibilityChange("remoteWork", checked)}
                  />
                  <label htmlFor="remoteWork">Remote Work Options</label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="flexibleHours"
                    checked={accessibilityFeatures.flexibleHours}
                    onCheckedChange={(checked) => handleAccessibilityChange("flexibleHours", checked)}
                  />
                  <label htmlFor="flexibleHours">Flexible Hours</label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="assistiveTechnology"
                    checked={accessibilityFeatures.assistiveTechnology}
                    onCheckedChange={(checked) => handleAccessibilityChange("assistiveTechnology", checked)}
                  />
                  <label htmlFor="assistiveTechnology">Assistive Technology Provided</label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="accessibleOffice"
                    checked={accessibilityFeatures.accessibleOffice}
                    onCheckedChange={(checked) => handleAccessibilityChange("accessibleOffice", checked)}
                  />
                  <label htmlFor="accessibleOffice">Accessible Office Space</label>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Submit Button */}
          <div className="flex justify-end space-x-4">
            <Link href="/jobs">
              <Button variant="outline" type="button">
                Cancel
              </Button>
            </Link>
            <Button
              type="submit"
              disabled={loading || !formData.jobTitle || !formData.description || !formData.companyName || !formData.jobType || !formData.applicationLink}
              className="bg-mustard hover:bg-forest-green text-white"
            >
              {loading ? "Posting..." : "Post Job"}
            </Button>
          </div>
        </form>
      </main>
    </div>
  )
} 
"use client"

import { Card, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Checkbox } from "@/components/ui/checkbox"
import { Separator } from "@/components/ui/separator"
import { MapPin, DollarSign, Accessibility, Search } from "lucide-react"

export function JobSidebar() {
  const jobSuggestions = [
    {
      title: "Accessibility Specialist",
      company: "Global Tech",
      location: "Remote",
      salary: "$90k - $120k",
      accessibility: ["Screen Reader", "Keyboard Navigation"],
    },
    {
      title: "Inclusive UX Designer",
      company: "Innovate Solutions",
      location: "New York, NY",
      salary: "$80k - $110k",
      accessibility: ["Visual Impairment", "Cognitive Disability"],
    },
    {
      title: "Disability Inclusion Program Manager",
      company: "Diverse Corp",
      location: "San Francisco, CA",
      salary: "$100k - $130k",
      accessibility: ["Mobility Impairment"],
    },
  ]

  return (
    <Card className="bg-white shadow-md rounded-lg p-4 space-y-6">
      <CardHeader className="p-0">
        <CardTitle className="text-xl font-bold text-charcoal">Job Search</CardTitle>
      </CardHeader>

      {/* Search and Filters */}
      <div className="space-y-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
          <Input
            placeholder="Keywords (e.g., UX, accessibility)"
            className="pl-10 bg-cream border-gray-200 focus:border-mustard"
          />
        </div>
        <div className="relative">
          <MapPin className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
          <Input
            placeholder="Location (e.g., Remote, NYC)"
            className="pl-10 bg-cream border-gray-200 focus:border-mustard"
          />
        </div>

        <Separator className="my-4" />

        <h4 className="font-semibold text-charcoal">Accessibility Filters</h4>
        <div className="space-y-2">
          <div className="flex items-center space-x-2">
            <Checkbox id="screen-reader" />
            <label htmlFor="screen-reader" className="text-gray-700">
              Screen Reader Compatible
            </label>
          </div>
          <div className="flex items-center space-x-2">
            <Checkbox id="keyboard-nav" />
            <label htmlFor="keyboard-nav" className="text-gray-700">
              Keyboard Navigation
            </label>
          </div>
          <div className="flex items-center space-x-2">
            <Checkbox id="visual-aid" />
            <label htmlFor="visual-aid" className="text-gray-700">
              Visual Aid Friendly
            </label>
          </div>
          <div className="flex items-center space-x-2">
            <Checkbox id="mobility-support" />
            <label htmlFor="mobility-support" className="text-gray-700">
              Mobility Support
            </label>
          </div>
        </div>

        <Button className="w-full bg-mustard hover:bg-forest-green text-white font-medium">Apply Filters</Button>
      </div>

      <Separator className="my-6" />

      {/* Job Suggestions */}
      <CardHeader className="p-0">
        <CardTitle className="text-xl font-bold text-charcoal">Job Suggestions</CardTitle>
      </CardHeader>
      <div className="space-y-4">
        {jobSuggestions.map((job, index) => (
          <div key={index} className="border border-gray-200 rounded-lg p-3 hover:bg-light-beige transition-colors">
            <h5 className="font-semibold text-charcoal">{job.title}</h5>
            <p className="text-sm text-gray-600">
              {job.company} • {job.location}
            </p>
            <div className="flex items-center text-sm text-gray-700 mt-1">
              <DollarSign className="h-4 w-4 mr-1 text-forest-green" />
              <span>{job.salary}</span>
            </div>
            <div className="flex items-center text-sm text-gray-700 mt-1">
              <Accessibility className="h-4 w-4 mr-1 text-indigo" />
              <span className="text-xs text-gray-500">{job.accessibility.join(", ")}</span>
            </div>
            <Button variant="link" className="p-0 h-auto text-mustard hover:text-forest-green mt-2">
              View Details
            </Button>
          </div>
        ))}
      </div>
    </Card>
  )
}

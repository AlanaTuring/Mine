"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { 
  Contrast, 
  Type, 
  Volume2, 
  VolumeX, 
  ZoomIn, 
  ZoomOut,
  Accessibility,
  Moon
} from "lucide-react"
import { Switch } from "@/components/ui/switch"

interface AccessibilityToolbarProps {
  onHighContrastChange: (enabled: boolean) => void
  onLargeTextChange: (enabled: boolean) => void
  onAudioChange: (enabled: boolean) => void
  onZoomChange: (level: number) => void
}

export function AccessibilityToolbar({
  onHighContrastChange,
  onLargeTextChange,
  onAudioChange,
  onZoomChange
}: AccessibilityToolbarProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [highContrast, setHighContrast] = useState(false)
  const [largeText, setLargeText] = useState(false)
  const [audioEnabled, setAudioEnabled] = useState(true)
  const [zoomLevel, setZoomLevel] = useState(100)


  const handleHighContrast = (enabled: boolean) => {
    setHighContrast(enabled)
    onHighContrastChange(enabled)
  }

  const handleLargeText = (enabled: boolean) => {
    setLargeText(enabled)
    onLargeTextChange(enabled)
  }

  const handleAudio = (enabled: boolean) => {
    setAudioEnabled(enabled)
    onAudioChange(enabled)
  }

  const handleZoom = (level: number) => {
    setZoomLevel(level)
    onZoomChange(level)
  }



  return (
    <div className="fixed top-4 right-4 z-50">
      <Button
        onClick={() => setIsOpen(!isOpen)}
        className="bg-mustard hover:bg-forest-green text-white"
        aria-label="Accessibility settings"
      >
        <Accessibility className="h-4 w-4 mr-2" />
        Accessibility
      </Button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-white rounded-lg shadow-lg border border-gray-200 p-4">
          <h3 className="font-semibold text-charcoal mb-3">Accessibility Settings</h3>
          
          <div className="space-y-4">
            {/* Dark Mode */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Moon className="h-4 w-4 text-mustard" />
                <label htmlFor="dark-mode" className="text-sm">Dark Mode</label>
              </div>
              <Switch
                id="dark-mode"
                checked={highContrast}
                onCheckedChange={handleHighContrast}
              />
            </div>

            {/* Large Text */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Type className="h-4 w-4 text-mustard" />
                <label htmlFor="large-text" className="text-sm">Large Text</label>
              </div>
              <Switch
                id="large-text"
                checked={largeText}
                onCheckedChange={handleLargeText}
              />
            </div>

            {/* Audio */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {audioEnabled ? (
                  <Volume2 className="h-4 w-4 text-mustard" />
                ) : (
                  <VolumeX className="h-4 w-4 text-mustard" />
                )}
                <label htmlFor="audio" className="text-sm">Audio</label>
              </div>
              <Switch
                id="audio"
                checked={audioEnabled}
                onCheckedChange={handleAudio}
              />
            </div>

            {/* Zoom */}
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <ZoomIn className="h-4 w-4 text-mustard" />
                <label className="text-sm">Zoom: {zoomLevel}%</label>
              </div>
              <div className="flex items-center space-x-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleZoom(Math.max(zoomLevel - 10, 50))}
                >
                  <ZoomOut className="h-3 w-3" />
                </Button>
                <div className="flex-1 h-2 bg-gray-200 rounded">
                  <div 
                    className="h-full bg-mustard rounded"
                    style={{ width: `${((zoomLevel - 50) / 150) * 100}%` }}
                  />
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleZoom(Math.min(zoomLevel + 10, 200))}
                >
                  <ZoomIn className="h-3 w-3" />
                </Button>
              </div>
            </div>


          </div>
        </div>
      )}
    </div>
  )
} 
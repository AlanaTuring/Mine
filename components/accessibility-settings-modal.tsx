"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import { 
  Contrast, 
  Type, 
  Volume2, 
  VolumeX, 
  ZoomIn, 
  ZoomOut,
  Eye,
  Ear,
  Hand,
  Brain,
  X,
  Moon
} from "lucide-react"
import { useAccessibility } from "./accessibility-provider"

interface AccessibilitySettingsModalProps {
  isOpen: boolean
  onClose: () => void
}

export function AccessibilitySettingsModal({ isOpen, onClose }: AccessibilitySettingsModalProps) {
  const { settings, updateSettings } = useAccessibility()
  const { highContrast } = settings;

  useEffect(() => {
    if (highContrast) {
      document.documentElement.classList.add("high-contrast");
    } else {
      document.documentElement.classList.remove("high-contrast");
    }
  }, [highContrast]);

  const handleHighContrast = (enabled: boolean) => {
    updateSettings({ highContrast: enabled })
  }

  const handleLargeText = (enabled: boolean) => {
    updateSettings({ largeText: enabled })
  }

  const handleAudio = (enabled: boolean) => {
    updateSettings({ audioEnabled: enabled })
  }

  const handleZoom = (level: number) => {
    updateSettings({ zoomLevel: level })
  }



  const handleDisabilityToggle = (type: keyof typeof settings, enabled: boolean) => {
    updateSettings({ [type]: enabled })
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden dialog-content bg-card">
        <DialogHeader className="flex-shrink-0 dialog-header">
          <DialogTitle className="text-xl font-semibold text-foreground dialog-title">
            Accessibility Settings
          </DialogTitle>
        </DialogHeader>
        
        <div className="overflow-y-auto max-h-[calc(80vh-80px)] pr-2 space-y-8 p-6">
          {/* Visual Accessibility */}
          <div className="space-y-6">
            <h3 className="text-lg font-medium text-foreground flex items-center gap-3">
              <Eye className="h-5 w-5 text-mustard" aria-hidden="true" />
              Visual Accessibility
            </h3>
            
            <div className="space-y-6 pl-8">
              <div className="flex items-center justify-between p-4 bg-muted rounded-lg border border-border">
                <div className="flex items-center space-x-3">
                  <Moon className="h-5 w-5 text-mustard" aria-hidden="true" />
                  <div>
                    <label htmlFor="dark-mode" className="text-sm font-medium text-foreground">Dark Mode</label>
                    <p className="text-xs text-muted-foreground mt-1">Switch to dark theme for better visibility in low light</p>
                  </div>
                </div>
                <Switch
                  id="dark-mode"
                  checked={settings.highContrast}
                  onCheckedChange={handleHighContrast}
                  aria-label="Toggle dark mode"
                  className="switch"
                />
              </div>

              <div className="flex items-center justify-between p-4 bg-muted rounded-lg border border-border">
                <div className="flex items-center space-x-3">
                  <Type className="h-5 w-5 text-mustard" aria-hidden="true" />
                  <div>
                    <label htmlFor="large-text" className="text-sm font-medium text-foreground">Large Text</label>
                    <p className="text-xs text-muted-foreground mt-1">Increase text size for better readability</p>
                  </div>
                </div>
                <Switch
                  id="large-text"
                  checked={settings.largeText}
                  onCheckedChange={handleLargeText}
                  aria-label="Toggle large text mode"
                  className="switch"
                />
              </div>

              <div className="space-y-4 p-4 bg-muted rounded-lg border border-border">
                <div className="flex items-center space-x-3">
                  <ZoomIn className="h-5 w-5 text-mustard" aria-hidden="true" />
                  <div>
                    <label className="text-sm font-medium text-foreground">Zoom Level: {settings.zoomLevel}%</label>
                    <p className="text-xs text-muted-foreground mt-1">Adjust the overall zoom level of the interface</p>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleZoom(Math.max(settings.zoomLevel - 10, 50))}
                    aria-label="Decrease zoom level by 10%"
                    className="btn bg-card border border-border text-foreground hover:bg-accent dark:bg-card dark:border-border dark:text-foreground dark:shadow-md"
                  >
                    <ZoomOut className="h-4 w-4" aria-hidden="true" />
                  </Button>
                  <div className="flex-1 h-3 bg-muted-foreground/20 rounded-full slider-track" role="progressbar" aria-valuenow={settings.zoomLevel} aria-valuemin={50} aria-valuemax={200} aria-label="Zoom level slider">
                    <div 
                      className="h-full bg-mustard rounded-full transition-all slider-range"
                      style={{ width: `${((settings.zoomLevel - 50) / 150) * 100}%` }}
                    />
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleZoom(Math.min(settings.zoomLevel + 10, 200))}
                    aria-label="Increase zoom level by 10%"
                    className="btn bg-card border border-border text-foreground hover:bg-accent dark:bg-card dark:border-border dark:text-foreground dark:shadow-md"
                  >
                    <ZoomIn className="h-4 w-4" aria-hidden="true" />
                  </Button>
                </div>
              </div>
            </div>
          </div>

          {/* Audio Accessibility */}
          <div className="space-y-6">
            <h3 className="text-lg font-medium text-foreground flex items-center gap-3">
              <Ear className="h-5 w-5 text-mustard" aria-hidden="true" />
              Audio Accessibility
            </h3>
            
            <div className="pl-8">
              <div className="flex items-center justify-between p-4 bg-muted rounded-lg border border-border">
                <div className="flex items-center space-x-3">
                  {settings.audioEnabled ? (
                    <Volume2 className="h-5 w-5 text-mustard" aria-hidden="true" />
                  ) : (
                    <VolumeX className="h-5 w-5 text-mustard" aria-hidden="true" />
                  )}
                  <div>
                    <label htmlFor="audio" className="text-sm font-medium text-foreground">Audio</label>
                    <p className="text-xs text-muted-foreground mt-1">Enable audio feedback and notifications</p>
                  </div>
                </div>
                <Switch
                  id="audio"
                  checked={settings.audioEnabled}
                  onCheckedChange={handleAudio}
                  aria-label="Toggle audio accessibility features"
                  className="switch"
                />
              </div>
            </div>
          </div>

          {/* Disability Profile */}
          <div className="space-y-6">
            <h3 className="text-lg font-medium text-foreground flex items-center gap-3">
              <Brain className="h-5 w-5 text-mustard" aria-hidden="true" />
              Disability Profile
            </h3>
            <p className="text-sm text-muted-foreground pl-8">
              Help us personalize your experience by sharing your accessibility needs.
            </p>
            
            <div className="space-y-4 pl-8">
              <div className="flex items-center justify-between p-4 bg-muted rounded-lg border border-border">
                <div className="flex items-center space-x-3">
                  <Eye className="h-5 w-5 text-mustard" aria-hidden="true" />
                  <div>
                    <label htmlFor="visual-impairment" className="text-sm font-medium text-foreground">Visual Impairment</label>
                    <p className="text-xs text-muted-foreground mt-1">Optimize for visual accessibility needs</p>
                  </div>
                </div>
                <Switch
                  id="visual-impairment"
                  checked={settings.visualImpairment}
                  onCheckedChange={(enabled) => handleDisabilityToggle('visualImpairment', enabled)}
                  aria-label="Toggle visual impairment accommodation"
                  className="switch"
                />
              </div>

              <div className="flex items-center justify-between p-4 bg-muted rounded-lg border border-border">
                <div className="flex items-center space-x-3">
                  <Ear className="h-5 w-5 text-mustard" aria-hidden="true" />
                  <div>
                    <label htmlFor="hearing-impairment" className="text-sm font-medium text-foreground">Hearing Impairment</label>
                    <p className="text-xs text-muted-foreground mt-1">Optimize for hearing accessibility needs</p>
                  </div>
                </div>
                <Switch
                  id="hearing-impairment"
                  checked={settings.hearingImpairment}
                  onCheckedChange={(enabled) => handleDisabilityToggle('hearingImpairment', enabled)}
                  aria-label="Toggle hearing impairment accommodation"
                  className="switch"
                />
              </div>

              <div className="flex items-center justify-between p-4 bg-muted rounded-lg border border-border">
                <div className="flex items-center space-x-3">
                  <Hand className="h-5 w-5 text-mustard" aria-hidden="true" />
                  <div>
                    <label htmlFor="motor-impairment" className="text-sm font-medium text-foreground">Motor Impairment</label>
                    <p className="text-xs text-muted-foreground mt-1">Optimize for motor accessibility needs</p>
                  </div>
                </div>
                <Switch
                  id="motor-impairment"
                  checked={settings.motorImpairment}
                  onCheckedChange={(enabled) => handleDisabilityToggle('motorImpairment', enabled)}
                  aria-label="Toggle motor impairment accommodation"
                  className="switch"
                />
              </div>

              <div className="flex items-center justify-between p-4 bg-muted rounded-lg border border-border">
                <div className="flex items-center space-x-3">
                  <Brain className="h-5 w-5 text-mustard" aria-hidden="true" />
                  <div>
                    <label htmlFor="cognitive-impairment" className="text-sm font-medium text-foreground">Cognitive Impairment</label>
                    <p className="text-xs text-muted-foreground mt-1">Optimize for cognitive accessibility needs</p>
                  </div>
                </div>
                <Switch
                  id="cognitive-impairment"
                  checked={settings.cognitiveImpairment}
                  onCheckedChange={(enabled) => handleDisabilityToggle('cognitiveImpairment', enabled)}
                  aria-label="Toggle cognitive impairment accommodation"
                  className="switch"
                />
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
} 
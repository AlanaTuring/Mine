"use client"

import React, { createContext, useContext, useState, useEffect } from 'react'

interface AccessibilitySettings {
  highContrast: boolean
  largeText: boolean
  audioEnabled: boolean
  zoomLevel: number
  theme: 'light' | 'dark'
  visualImpairment: boolean
  hearingImpairment: boolean
  motorImpairment: boolean
  cognitiveImpairment: boolean
}

interface AccessibilityContextType {
  settings: AccessibilitySettings
  updateSettings: (newSettings: Partial<AccessibilitySettings>) => void
  resetSettings: () => void
}

const defaultSettings: AccessibilitySettings = {
  highContrast: false,
  largeText: false,
  audioEnabled: true,
  zoomLevel: 100,
  theme: 'light',
  visualImpairment: false,
  hearingImpairment: false,
  motorImpairment: false,
  cognitiveImpairment: false,
}

const AccessibilityContext = createContext<AccessibilityContextType | undefined>(undefined)

export function AccessibilityProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<AccessibilitySettings>(defaultSettings)

  // Load settings from localStorage on mount
  useEffect(() => {
    const savedSettings = localStorage.getItem('accessibility-settings')
    if (savedSettings) {
      try {
        const parsed = JSON.parse(savedSettings)
        setSettings({ ...defaultSettings, ...parsed })
      } catch (error) {
        console.error('Error loading accessibility settings:', error)
      }
    }
  }, [])

  // Save settings to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('accessibility-settings', JSON.stringify(settings))
    
    // Apply settings to document
    const root = document.documentElement
    
    // Apply dark mode (using highContrast setting as dark mode toggle)
    if (settings.highContrast) {
      root.classList.add('dark')
      root.setAttribute('data-theme', 'dark')
    } else {
      root.classList.remove('dark')
      root.setAttribute('data-theme', 'light')
    }
    
    // Apply large text
    if (settings.largeText) {
      root.setAttribute('data-large-text', 'true')
      root.classList.add('large-text')
    } else {
      root.removeAttribute('data-large-text')
      root.classList.remove('large-text')
    }
    
    // Apply zoom level
    root.style.fontSize = `${settings.zoomLevel}%`
    
  }, [settings])

  const updateSettings = (newSettings: Partial<AccessibilitySettings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }))
  }

  const resetSettings = () => {
    setSettings(defaultSettings)
  }

  return (
    <AccessibilityContext.Provider value={{ settings, updateSettings, resetSettings }}>
      {children}
    </AccessibilityContext.Provider>
  )
}

export function useAccessibility() {
  const context = useContext(AccessibilityContext)
  if (context === undefined) {
    throw new Error('useAccessibility must be used within an AccessibilityProvider')
  }
  return context
} 
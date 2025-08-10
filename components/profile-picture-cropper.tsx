"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Slider } from "@/components/ui/slider"
import { RotateCcw, ZoomIn, ZoomOut, Crop } from "lucide-react"

interface ProfilePictureCropperProps {
  isOpen: boolean
  onClose: () => void
  onSave: (croppedImage: string) => void
  originalImage: string
}

export function ProfilePictureCropper({ 
  isOpen, 
  onClose, 
  onSave, 
  originalImage 
}: ProfilePictureCropperProps) {
  const [scale, setScale] = useState(1)
  const [rotation, setRotation] = useState(0)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)

  const handleSave = () => {
    if (canvasRef.current) {
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      if (ctx) {
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        
        // Apply transformations
        ctx.save()
        ctx.translate(canvas.width / 2, canvas.height / 2)
        ctx.rotate((rotation * Math.PI) / 180)
        ctx.scale(scale, scale)
        ctx.translate(-canvas.width / 2 + position.x, -canvas.height / 2 + position.y)
        
        // Draw image
        if (imageRef.current) {
          ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height)
        }
        
        ctx.restore()
        
        // Get cropped image as data URL
        const croppedImage = canvas.toDataURL('image/jpeg', 0.9)
        onSave(croppedImage)
        onClose()
      }
    }
  }

  const handleZoomIn = () => {
    setScale(prev => Math.min(prev + 0.1, 3))
  }

  const handleZoomOut = () => {
    setScale(prev => Math.max(prev - 0.1, 0.5))
  }

  const handleRotate = () => {
    setRotation(prev => (prev + 90) % 360)
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    const startX = e.clientX
    const startY = e.clientY
    const startPosition = { ...position }

    const handleMouseMove = (e: MouseEvent) => {
      const deltaX = e.clientX - startX
      const deltaY = e.clientY - startY
      setPosition({
        x: startPosition.x + deltaX,
        y: startPosition.y + deltaY
      })
    }

    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Adjust Profile Picture</DialogTitle>
        </DialogHeader>
        
        <div className="space-y-4">
          <div className="relative bg-gray-100 rounded-lg overflow-hidden" style={{ height: '400px' }}>
            <div
              className="absolute inset-0 cursor-move"
              onMouseDown={handleMouseDown}
              style={{
                transform: `translate(${position.x}px, ${position.y}px) scale(${scale}) rotate(${rotation}deg)`,
                transformOrigin: 'center'
              }}
            >
              <img
                ref={imageRef}
                src={originalImage}
                alt="Profile picture"
                className="w-full h-full object-cover"
                style={{ maxWidth: 'none', maxHeight: 'none' }}
              />
            </div>
            
            {/* Crop overlay */}
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute inset-0 bg-black bg-opacity-50"></div>
              <div className="absolute top-1/2 left-1/2 w-48 h-48 -translate-x-1/2 -translate-y-1/2 border-2 border-white rounded-full"></div>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <Button variant="outline" size="sm" onClick={handleZoomOut}>
                <ZoomOut className="h-4 w-4" />
              </Button>
              <div className="flex-1">
                <Slider
                  value={[scale]}
                  onValueChange={(value) => setScale(value[0])}
                  min={0.5}
                  max={3}
                  step={0.1}
                  className="w-full"
                />
              </div>
              <Button variant="outline" size="sm" onClick={handleZoomIn}>
                <ZoomIn className="h-4 w-4" />
              </Button>
            </div>
            
            <div className="flex items-center space-x-4">
              <Button variant="outline" size="sm" onClick={handleRotate}>
                <RotateCcw className="h-4 w-4" />
              </Button>
              <span className="text-sm text-gray-600">Rotation: {rotation}°</span>
            </div>
          </div>
          
          <div className="flex justify-end space-x-2">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleSave}>
              Save
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
} 
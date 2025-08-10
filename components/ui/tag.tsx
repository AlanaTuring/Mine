import React from 'react';

interface TagProps {
  children: React.ReactNode;
  onRemove?: () => void;
  removable?: boolean;
  className?: string;
}

export function Tag({ children, onRemove, removable = false, className = '' }: TagProps) {
  return (
    <span
      className={`inline-flex items-center rounded-full bg-mustard/20 text-mustard font-semibold px-3 py-1 text-sm mr-2 mb-2 border border-mustard/30 ${className}`}
    >
      {children}
      {removable && (
        <button
          type="button"
          onClick={onRemove}
          className="ml-2 text-mustard hover:text-red-500 focus:outline-none"
          aria-label="Remove tag"
        >
          ×
        </button>
      )}
    </span>
  );
} 
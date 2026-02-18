'use client';

import { useState, useRef } from 'react';
import { uploadDocument } from '@/lib/api';

interface DocumentUploadProps {
  onUploadComplete?: (filename: string) => void;
  onError?: (error: string) => void;
}

export default function DocumentUpload({ onUploadComplete, onError }: DocumentUploadProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = async (files: FileList) => {
    const file = files[0];
    if (!file) return;

    // Check if it's a PDF
    if (file.type !== 'application/pdf') {
      onError?.('Please upload a PDF file');
      return;
    }

    setIsUploading(true);

    try {
      const response = await uploadDocument(file);
      onUploadComplete?.(response.document_name);
    } catch (error) {
      console.error('Upload failed:', error);
      onError?.('Failed to upload document. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragIn = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragOut = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(e.target.files);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        onChange={handleFileSelect}
        className="hidden"
      />
      
      <div
        className={`
          border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all
          ${dragActive 
            ? 'border-purple-500 bg-purple-50' 
            : 'border-gray-300 hover:border-purple-400 hover:bg-purple-50'
          }
          ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
        onDragEnter={handleDragIn}
        onDragLeave={handleDragOut}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={!isUploading ? handleClick : undefined}
      >
        {isUploading ? (
          <div className="flex flex-col items-center">
            <div className="w-8 h-8 border-2 border-purple-600 border-t-transparent rounded-full animate-spin mb-2"></div>
            <p className="text-gray-600">Uploading document...</p>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <svg
              className="w-12 h-12 text-gray-400 mb-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Upload PDF Document
            </h3>
            <p className="text-gray-600 mb-4">
              Drag and drop your PDF here, or click to browse
            </p>
            <button
              type="button"
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
            >
              Choose File
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
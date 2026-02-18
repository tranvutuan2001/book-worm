'use client';

import { useState, useEffect } from 'react';
import { listDocuments } from '@/lib/api';
import { DocumentInfo } from '@/lib/schemas';

interface DocumentListProps {
  onDocumentSelect?: (documentName: string) => void;
  selectedDocument?: string;
}

export default function DocumentList({ onDocumentSelect, selectedDocument }: DocumentListProps) {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadDocuments = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await listDocuments();
      setDocuments(response.documents);
    } catch (err) {
      console.error('Failed to load documents:', err);
      setError('Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  const handleDocumentClick = (documentName: string) => {
    onDocumentSelect?.(documentName);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-4">
        <div className="w-6 h-6 border-2 border-purple-600 border-t-transparent rounded-full animate-spin"></div>
        <span className="ml-2 text-gray-600">Loading documents...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-center">
        <p className="text-red-600 mb-2">{error}</p>
        <button
          onClick={loadDocuments}
          className="px-4 py-2 text-sm bg-purple-600 text-white rounded hover:bg-purple-700 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (documents.length === 0) {
    return (
      <div className="p-4 text-center text-gray-600">
        <svg
          className="w-12 h-12 mx-auto mb-2 text-gray-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
          />
        </svg>
        <p>No documents uploaded yet</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <h3 className="font-medium text-gray-900 mb-3">Available Documents</h3>
      {documents.map((doc, index) => {
        const isReady = doc.status === 'ready';
        const isProcessing = doc.status === 'processing' || doc.status === 'analyzing';
        const isError = doc.status === 'error';
        
        return (
          <button
            key={index}
            onClick={() => isReady ? handleDocumentClick(doc.name) : undefined}
            disabled={!isReady}
            className={`
              w-full text-left p-4 rounded-xl border-2 transition-all duration-200 relative
              ${selectedDocument === doc.name && isReady
                ? 'bg-gradient-to-r from-purple-600 to-purple-700 border-purple-600 text-white shadow-xl shadow-purple-400/40 scale-[1.02] transform'
                : isReady
                ? 'bg-white border-gray-200 hover:bg-purple-50 hover:border-purple-300 hover:shadow-md'
                : 'bg-gray-50 border-gray-200 cursor-not-allowed opacity-75'
              }
            `}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center min-w-0">
                {selectedDocument === doc.name && isReady && (
                  <div className="absolute -left-1 top-1/2 transform -translate-y-1/2 w-1 h-8 bg-white rounded-full"></div>
                )}
                <svg
                  className={`w-5 h-5 mr-3 ${
                    selectedDocument === doc.name && isReady 
                      ? 'text-white' 
                      : isReady ? 'text-red-500' : isError ? 'text-red-600' : 'text-gray-400'
                  }`}
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z"
                    clipRule="evenodd"
                  />
                </svg>
                <div className="min-w-0 flex-1">
                  <span className={`truncate block font-medium text-sm ${
                    selectedDocument === doc.name && isReady ? 'text-white' : 'text-gray-900'
                  }`}>{doc.name}</span>
                  <span className={`text-xs font-medium ${
                    selectedDocument === doc.name && isReady ? 'text-purple-200' :
                    isReady ? 'text-green-600' : 
                    isError ? 'text-red-600' : 
                    'text-blue-600'
                  }`}>
                    {selectedDocument === doc.name && isReady ? '✓ Currently selected' :
                     doc.status === 'ready' ? 'Ready to chat' : 
                     doc.status === 'processing' ? 'Processing...' :
                     doc.status === 'analyzing' ? 'Analyzing...' :
                     'Error'}
                  </span>
                </div>
              </div>
              {isProcessing && (
                <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}
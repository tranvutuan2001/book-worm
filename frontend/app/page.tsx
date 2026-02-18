'use client';

import { useState, useRef, useEffect } from 'react';
import MessageBubble from '@/components/MessageBubble';
import ChatInput from '@/components/ChatInput';
import ChatHeader from '@/components/ChatHeader';
import DocumentUpload from '@/components/DocumentUpload';
import DocumentList from '@/components/DocumentList';
import ModelManagement from '@/components/ModelManagement';
import ModelSelector from '@/components/ModelSelector';
import { Conversation, Message } from '@/lib/schemas';
import { sendMessage } from '@/lib/api';

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string>();
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null);
  const [showSidebar, setShowSidebar] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [selectedChatModel, setSelectedChatModel] = useState<string | null>(null);
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    // Check if models are selected
    if (!selectedChatModel || !selectedEmbeddingModel) {
      alert('Please select both chat and embedding models before sending a message');
      return;
    }

    // Add user message
    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Prepare conversation with all messages including the new one
      const currentConversationId = conversationId || `conv-${Date.now()}`;
      if (!conversationId) {
        setConversationId(currentConversationId);
      }

      const conversation: Conversation = {
        id: currentConversationId,
        message_list: [...messages, userMessage],
        timestamp: Date.now(),
        document_name: selectedDocument,
        chat_model: selectedChatModel,
        embedding_model: selectedEmbeddingModel,
      };

      // Call the document analysis API
      const response = await sendMessage(conversation);

      // Update conversation ID from response
      setConversationId(response.conversation_id);

      const assistantMessage: Message = {
        id: `msg-${response.timestamp}`,
        role: 'assistant',
        content: response.message,
        timestamp: response.timestamp,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Failed to send message:', error);
      
      // Add error message
      const errorMessage: Message = {
        id: `msg-${Date.now()}-error`,
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleUploadComplete = (filename: string) => {
    setUploadSuccess(`Successfully uploaded: ${filename}`);
    setUploadError(null);
    setTimeout(() => setUploadSuccess(null), 3000);
  };

  const handleUploadError = (error: string) => {
    setUploadError(error);
    setUploadSuccess(null);
    setTimeout(() => setUploadError(null), 5000);
  };

  const handleDocumentSelect = (documentName: string) => {
    setSelectedDocument(documentName === selectedDocument ? null : documentName);
  };

  return (
    <div className="flex h-screen bg-linear-to-br from-purple-50 via-white to-blue-50">
      {/* Left Sidebar - Documents */}
      <div className={`
        fixed inset-y-0 left-0 z-50 w-80 bg-white/95 backdrop-blur-sm border-r border-purple-200 transform transition-transform duration-300 ease-in-out shadow-xl
        ${showSidebar ? 'translate-x-0' : '-translate-x-full'}
        lg:relative lg:translate-x-0
      `}>
        <div className="flex flex-col h-full">
          <div className="flex items-center justify-between p-4 border-b border-purple-200 bg-linear-to-r from-purple-600 to-blue-600">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Documents
            </h2>
            <button
              onClick={() => setShowSidebar(false)}
              className="lg:hidden p-2 rounded-lg hover:bg-white/20 text-white"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* Upload Section */}
            <div>
              <DocumentUpload 
                onUploadComplete={handleUploadComplete}
                onError={handleUploadError}
              />
              
              {/* Upload notifications */}
              {uploadSuccess && (
                <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg text-green-800 text-sm">
                  {uploadSuccess}
                </div>
              )}
              
              {uploadError && (
                <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-red-800 text-sm">
                  {uploadError}
                </div>
              )}
            </div>
            
            {/* Document List */}
            <DocumentList 
              onDocumentSelect={handleDocumentSelect}
              selectedDocument={selectedDocument || undefined}
            />
          </div>
        </div>
      </div>
      
      {/* Overlay for mobile */}
      {showSidebar && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setShowSidebar(false)}
        />
      )}
      
      {/* Main content area */}
      <div className="flex-1 flex flex-col min-w-0">
        <ChatHeader 
          onToggleSidebar={() => setShowSidebar(!showSidebar)}
          selectedDocument={selectedDocument}
        />
      
        {/* Main content with chat and model management */}
        <div className="flex-1 flex overflow-hidden">
          {/* Chat Area */}
          <div className="flex-1 flex flex-col min-w-0">
            <main className="flex-1 overflow-y-auto">
              <div className="max-w-4xl mx-auto px-4 py-6">
                {/* Model Selector - Compact at top */}
                <div className="mb-4">
                  <ModelSelector
                    selectedChatModel={selectedChatModel}
                    selectedEmbeddingModel={selectedEmbeddingModel}
                    onChatModelChange={setSelectedChatModel}
                    onEmbeddingModelChange={setSelectedEmbeddingModel}
                  />
                </div>

                {/* Chat Messages */}
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-center py-12">
                    <div className="w-20 h-20 bg-linear-to-br from-purple-600 to-blue-600 rounded-2xl flex items-center justify-center mb-6 shadow-2xl shadow-purple-400/50 animate-pulse">
                      <svg
                        className="w-10 h-10 text-white"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                        />
                      </svg>
                    </div>
                    <h2 className="text-3xl font-bold bg-linear-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent mb-3">
                      {selectedChatModel && selectedEmbeddingModel 
                        ? 'Ready to chat!' 
                        : 'Select models to get started'}
                    </h2>
                    <p className="text-gray-600 max-w-md text-lg">
                      {selectedChatModel && selectedEmbeddingModel
                        ? 'Ask me anything! I\'m using your selected models to provide intelligent responses.'
                        : 'Please select both a chat model and an embedding model to begin.'}
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {messages.map((message) => (
                      <MessageBubble key={message.id} message={message} />
                    ))}
                    {isLoading && (
                      <div className="flex justify-start mb-4">
                        <div className="max-w-[70%] rounded-2xl px-4 py-3 bg-white/90 backdrop-blur-sm border border-purple-200 shadow-sm">
                          <div className="flex gap-1">
                            <span className="w-2 h-2 bg-purple-600 rounded-full animate-bounce"></span>
                            <span
                              className="w-2 h-2 bg-purple-600 rounded-full animate-bounce"
                              style={{ animationDelay: '0.1s' }}
                            ></span>
                            <span
                              className="w-2 h-2 bg-purple-600 rounded-full animate-bounce"
                              style={{ animationDelay: '0.2s' }}
                            ></span>
                          </div>
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                )}
              </div>
            </main>

            <ChatInput 
              onSendMessage={handleSendMessage} 
              disabled={isLoading || !selectedChatModel || !selectedEmbeddingModel} 
              selectedDocument={selectedDocument}
            />
          </div>

          {/* Right Sidebar - Model Management */}
          <div className={`
            w-96 border-l border-purple-200 bg-white/95 backdrop-blur-sm flex-col overflow-hidden shadow-xl
            hidden xl:flex
          `}>
            <div className="flex items-center justify-between p-4 border-b border-purple-200 bg-linear-to-r from-purple-600 to-blue-600">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                </svg>
                Model Management
              </h2>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4">
              <ModelManagement />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

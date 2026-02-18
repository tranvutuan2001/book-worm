import { useState, KeyboardEvent } from 'react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
  selectedDocument?: string | null;
}

export default function ChatInput({ onSendMessage, disabled, selectedDocument }: ChatInputProps) {
  const [message, setMessage] = useState('');
  const [showAlert, setShowAlert] = useState(false);

  const handleSend = () => {
    if (!selectedDocument) {
      setShowAlert(true);
      setTimeout(() => setShowAlert(false), 3000);
      return;
    }
    
    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage('');
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-purple-200 bg-white/95 backdrop-blur-md">
      {/* Alert for no document selected */}
      {showAlert && (
        <div className="bg-amber-50 border-t border-amber-200 p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex items-center gap-3">
              <svg className="w-5 h-5 text-amber-600" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <div>
                <p className="text-amber-800 font-medium">Please select a book first</p>
                <p className="text-amber-700 text-sm">Choose a document from the sidebar to start chatting about it.</p>
              </div>
            </div>
          </div>
        </div>
      )}
      
      <div className="p-4">
        <div className="flex gap-2 max-w-4xl mx-auto">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder={selectedDocument ? "Type your message..." : "Select a book from the sidebar first..."}
            disabled={disabled}
            rows={1}
            className="flex-1 resize-none rounded-xl border border-purple-200 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-400 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button
            onClick={handleSend}
            disabled={!message.trim() || disabled}
            className="px-6 py-3 bg-purple-600 text-white rounded-xl hover:bg-purple-700 disabled:bg-purple-300 disabled:cursor-not-allowed transition-all font-medium shadow-lg shadow-purple-400/30 hover:shadow-purple-500/40"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

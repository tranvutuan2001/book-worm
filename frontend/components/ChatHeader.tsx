interface ChatHeaderProps {
  onToggleSidebar?: () => void;
  selectedDocument?: string | null;
}

export default function ChatHeader({ onToggleSidebar, selectedDocument }: ChatHeaderProps) {
  return (
    <header className="bg-white/95 backdrop-blur-md border-b border-purple-200 shadow-sm">
      <div className="max-w-4xl mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {onToggleSidebar && (
              <button
                onClick={onToggleSidebar}
                className="lg:hidden p-2 rounded-lg hover:bg-purple-50 text-purple-600"
              >
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              </button>
            )}
            <div className="w-10 h-10 bg-linear-to-br from-purple-600 to-purple-800 rounded-full flex items-center justify-center shadow-lg shadow-purple-500/30">
              <svg
                className="w-6 h-6 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-800">
                BookWorm AI
              </h1>
              <p className="text-sm text-gray-600">
                AI-powered document analysis
              </p>
            </div>
          </div>
          
          {/* Current Book Section */}
          {selectedDocument && (
            <div className="flex-1 flex justify-center">
              <div className="bg-gradient-to-r from-purple-600 to-purple-700 text-white px-6 py-3 rounded-xl shadow-lg">
                <div className="flex items-center gap-3">
                  <svg
                    className="w-5 h-5 text-purple-200"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <div>
                    <p className="text-xs font-medium text-purple-200 uppercase tracking-wide">Currently Reading</p>
                    <p className="font-semibold text-white truncate max-w-64">{selectedDocument}</p>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-purple-600 text-white shadow-md shadow-purple-400/30">
              Online
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}

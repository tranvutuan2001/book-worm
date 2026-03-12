interface EditorHeaderProps {
  title: string;
  onTitleChange: (value: string) => void;
  onExport: () => void;
}

export default function EditorHeader({ title, onTitleChange, onExport }: EditorHeaderProps) {
  return (
    <div className="bg-white border-b border-purple-200 px-6 py-3 flex items-center justify-between shadow-sm shrink-0">
      {/* ── Brand / info ── */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 bg-linear-to-br from-purple-600 to-blue-600 rounded-xl flex items-center justify-center shadow-md shadow-purple-400/30">
          <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <div>
          <h1 className="text-lg font-semibold text-gray-800">PDF Editor</h1>
          <p className="text-xs text-gray-500">Create and export A4 portrait documents</p>
        </div>
      </div>

      {/* ── Controls ── */}
      <div className="flex items-center gap-3">
        <input
          type="text"
          value={title}
          onChange={(e) => onTitleChange(e.target.value)}
          placeholder="Document title…"
          className="border border-purple-200 rounded-lg px-3 py-1.5 text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-400 w-56"
        />
        <button
          onClick={onExport}
          className="flex items-center gap-2 px-4 py-2 bg-linear-to-r from-purple-600 to-blue-600 text-white text-sm font-medium rounded-lg shadow-md shadow-purple-400/30 hover:from-purple-700 hover:to-blue-700 transition-all"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          Export PDF
        </button>
      </div>
    </div>
  );
}

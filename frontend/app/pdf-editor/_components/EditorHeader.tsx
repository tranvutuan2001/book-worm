import { useRef } from 'react';

interface EditorHeaderProps {
  title: string;
  onTitleChange: (value: string) => void;
  onExport: () => void;
  onUploadDefinition: (file: File) => void;
  onDownloadDefinition: () => void;
}

export default function EditorHeader({
  title,
  onTitleChange,
  onExport,
  onUploadDefinition,
  onDownloadDefinition,
}: EditorHeaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

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
      <div className="flex items-center gap-2">
        <input
          type="text"
          value={title}
          onChange={(e) => onTitleChange(e.target.value)}
          placeholder="Document title…"
          className="border border-purple-200 rounded-lg px-3 py-1.5 text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-400 w-56"
        />

        {/* Hidden file input for definition upload */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onUploadDefinition(file);
            // Reset value so the same file can be re-uploaded
            e.target.value = '';
          }}
        />

        {/* Upload definition */}
        <button
          onClick={() => fileInputRef.current?.click()}
          title="Upload definition (.bkwpdf.json)"
          className="flex items-center gap-1.5 px-3 py-2 border border-gray-300 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
          </svg>
          Open
        </button>

        {/* Download definition */}
        <button
          onClick={onDownloadDefinition}
          title="Save document definition as .bkwpdf.json"
          className="flex items-center gap-1.5 px-3 py-2 border border-gray-300 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
          </svg>
          Save
        </button>

        {/* Divider */}
        <div className="w-px h-6 bg-gray-200 mx-1" />

        {/* Export PDF */}
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

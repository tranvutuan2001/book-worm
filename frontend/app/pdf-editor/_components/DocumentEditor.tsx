import React from 'react';

interface DocumentEditorProps {
  fontFamily: string;
  fontSize: string;
  initEditor: (node: HTMLDivElement | null) => void;
  onInput: () => void;
  onKeyUp: () => void;
  onMouseUp: () => void;
}

export default function DocumentEditor({
  fontFamily,
  fontSize,
  initEditor,
  onInput,
  onKeyUp,
  onMouseUp,
}: DocumentEditorProps) {
  return (
    <div className="flex-1 flex flex-col overflow-hidden border-r border-gray-200">
      <div className="bg-gray-100 px-4 py-1.5 text-xs font-medium text-gray-500 uppercase tracking-wide shrink-0">
        Editor
      </div>
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl mx-auto">
          <div
            ref={initEditor}
            contentEditable
            suppressContentEditableWarning
            onInput={onInput}
            onKeyUp={onKeyUp}
            onMouseUp={onMouseUp}
            style={{ fontFamily, fontSize: `${fontSize}pt` }}
            className="min-h-[600px] bg-white rounded-xl shadow-sm border border-gray-200 p-8 text-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-300 leading-relaxed"
            data-placeholder="Start typing your document here…"
          />
        </div>
      </div>

      <style>{`
        [data-placeholder]:empty::before {
          content: attr(data-placeholder);
          color: #aaa;
          pointer-events: none;
        }
      `}</style>
    </div>
  );
}

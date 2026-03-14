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
            className="editor-content min-h-[600px] bg-white rounded-xl shadow-sm border border-gray-200 p-8 text-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-300 leading-relaxed"
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
        .editor-content ul { list-style: disc; margin-left: 1.4em; }
        .editor-content ol { list-style: decimal; margin-left: 1.4em; }
        .editor-content h1 { font-size: 2em; font-weight: 700; margin: .6em 0; }
        .editor-content h2 { font-size: 1.5em; font-weight: 700; margin: .7em 0; }
        .editor-content h3 { font-size: 1.2em; font-weight: 600; margin: .8em 0; }
        .editor-content h4 { font-size: 1em; font-weight: 600; margin: .9em 0; }
        .editor-content p  { margin: .6em 0; }
        .editor-content blockquote {
          border-left: 4px solid #7c3aed;
          padding: .3em .8em;
          margin: .8em 0;
          color: #555;
          font-style: italic;
        }
        .editor-content hr { border: none; border-top: 1px solid #ccc; margin: 1em 0; }
        .editor-content pre,
        .editor-content code {
          font-family: 'Courier New', monospace;
          background: #f5f5f5;
          padding: .15em .4em;
          border-radius: 3px;
          font-size: .88em;
        }
        .editor-content table { border-collapse: collapse; width: 100%; margin: .8em 0; }
        .editor-content td,
        .editor-content th {
          border: 1px solid #ccc;
          padding: .3em .5em;
          font-size: .9em;
        }
        .editor-content th { background: #f3f0ff; font-weight: 600; }
      `}</style>
    </div>
  );
}

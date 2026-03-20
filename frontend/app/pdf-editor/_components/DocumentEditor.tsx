'use client';

import React from 'react';
import { Slate, Editable, type RenderElementProps, type RenderLeafProps } from 'slate-react';
import type { Descendant } from 'slate';
import type { CustomEditor, CustomElement, CustomText } from '../_utils/slate-types';

// ─────────────────────────────────────────────────────────────────────────────
// Element renderer
// ─────────────────────────────────────────────────────────────────────────────

function Element({ attributes, children, element }: RenderElementProps) {
  const el = element as CustomElement;
  const style: React.CSSProperties = el.align ? { textAlign: el.align } : {};

  switch (el.type) {
    case 'heading-one':
      return <h1 {...attributes} style={style}>{children}</h1>;
    case 'heading-two':
      return <h2 {...attributes} style={style}>{children}</h2>;
    case 'heading-three':
      return <h3 {...attributes} style={style}>{children}</h3>;
    case 'heading-four':
      return <h4 {...attributes} style={style}>{children}</h4>;
    case 'blockquote':
      return <blockquote {...attributes} style={style}>{children}</blockquote>;
    case 'code-block':
      return (
        <pre {...attributes} style={style}>
          <code>{children}</code>
        </pre>
      );
    case 'bulleted-list':
      return <ul {...attributes}>{children}</ul>;
    case 'numbered-list':
      return <ol {...attributes}>{children}</ol>;
    case 'list-item':
      return <li {...attributes} style={style}>{children}</li>;
    case 'divider':
      return (
        <div {...attributes} contentEditable={false} style={{ userSelect: 'none' }}>
          <hr />
          {children}
        </div>
      );
    case 'raw-html':
      return (
        <div
          {...attributes}
          contentEditable={false}
          style={{ userSelect: 'none' }}
          dangerouslySetInnerHTML={{ __html: el.rawHtml ?? '' }}
        >
          {children}
        </div>
      );
    default:
      return <p {...attributes} style={style}>{children}</p>;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Leaf renderer
// ─────────────────────────────────────────────────────────────────────────────

function Leaf({ attributes, children, leaf }: RenderLeafProps) {
  const n = leaf as CustomText;

  const styles: React.CSSProperties = {};
  if (n.fontFamily) styles.fontFamily = n.fontFamily;
  if (n.fontSize) styles.fontSize = `${n.fontSize}pt`;
  if (n.color) styles.color = n.color;
  if (n.highlight) styles.backgroundColor = n.highlight;
  if (n.verticalAlign === 'super') { styles.verticalAlign = 'super'; styles.fontSize = '0.75em'; }
  if (n.verticalAlign === 'sub') { styles.verticalAlign = 'sub'; styles.fontSize = '0.75em'; }

  let el: React.ReactNode = (
    <span style={Object.keys(styles).length ? styles : undefined}>{children}</span>
  );
  if (n.strikethrough) el = <s>{el}</s>;
  if (n.underline) el = <u>{el}</u>;
  if (n.italic) el = <em>{el}</em>;
  if (n.bold) el = <strong>{el}</strong>;

  return <span {...attributes}>{el}</span>;
}

// ─────────────────────────────────────────────────────────────────────────────
// DocumentEditor component
// ─────────────────────────────────────────────────────────────────────────────

interface DocumentEditorProps {
  editor: CustomEditor;
  /** Increment to force Slate to remount with new initialValue (e.g. on file load). */
  slateKey: number;
  initialValue: Descendant[];
  fontFamily: string;
  fontSize: string;
  onChange: (value: Descendant[]) => void;
  onKeyDown?: (e: React.KeyboardEvent) => void;
}

export default function DocumentEditor({
  editor,
  slateKey,
  initialValue,
  fontFamily,
  fontSize,
  onChange,
  onKeyDown,
}: DocumentEditorProps) {
  return (
    <div className="flex-1 flex flex-col overflow-hidden border-r border-gray-200">
      <div className="bg-gray-100 px-4 py-1.5 text-xs font-medium text-gray-500 uppercase tracking-wide shrink-0">
        Editor
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl mx-auto">
          <Slate key={slateKey} editor={editor} initialValue={initialValue} onChange={onChange}>
            <Editable
              renderElement={(props) => <Element {...props} />}
              renderLeaf={(props) => <Leaf {...props} />}
              placeholder="Start typing your document here…"
              style={{ fontFamily, fontSize: `${fontSize}pt` }}
              className="editor-content min-h-[600px] bg-white rounded-xl shadow-sm border border-gray-200 p-8 text-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-300 leading-relaxed"
              spellCheck
              onKeyDown={onKeyDown}
            />
          </Slate>
        </div>
      </div>

      <style>{`
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

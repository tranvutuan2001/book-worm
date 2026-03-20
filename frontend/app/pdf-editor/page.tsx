'use client';

import React, { useState, useCallback, useMemo } from 'react';
import { createEditor, type Descendant } from 'slate';
import { withReact } from 'slate-react';
import { withHistory } from 'slate-history';
import EditorHeader from '@/app/pdf-editor/_components/EditorHeader';
import EditorToolbar, { ActiveFormats } from '@/app/pdf-editor/_components/EditorToolbar';
import DocumentEditor from '@/app/pdf-editor/_components/DocumentEditor';
import DocumentPreview from '@/app/pdf-editor/_components/DocumentPreview';
import { exportPdf } from '@/config/exportPdf';
import {
  type PdfDocument,
  getMinifiedJsonSchema,
} from '@/lib/pdf-document-schema';
import { htmlToPdfDocument, pdfDocumentToHtml } from '@/lib/pdf-document-converter';
import { createBlankDocument, safeParsePdfDocument, safeParseMinifiedComponents, serializePdfDocument } from './_utils/serializer';
import {
  toggleMark,
  toggleBlock,
  setBlockFromTag,
  setMark,
  setAlign,
  insertDivider,
  removeAllMarks,
  indentListItem,
  outdentListItem,
  getActiveFormats,
} from './_utils/slate-commands';
import { slateToHtml, htmlToSlate, EMPTY_SLATE_VALUE } from './_utils/slate-html';

const DEFAULT_FORMATS: ActiveFormats = {
  bold: false,
  italic: false,
  underline: false,
  strikeThrough: false,
  justifyLeft: false,
  justifyCenter: false,
  justifyRight: false,
  justifyFull: false,
  insertUnorderedList: false,
  insertOrderedList: false,
};

export default function PdfEditorPage() {
  // Slate editor instance — stable across renders
  const editor = useMemo(() => withHistory(withReact(createEditor())), []);

  const [title, setTitle] = useState('Untitled Document');
  const [fontFamily, setFontFamily] = useState('Georgia, serif');
  const [fontSize, setFontSize] = useState('12');
  const [activeFormats, setActiveFormats] = useState<ActiveFormats>(DEFAULT_FORMATS);

  /** Slate document value — source of truth for the editor content. */
  const [slateValue, setSlateValue] = useState<Descendant[]>(EMPTY_SLATE_VALUE);

  /**
   * Bumped whenever we want to force Slate to remount with a new initialValue
   * (e.g. when loading a file). Passed as `key` to <DocumentEditor>.
   */
  const [slateKey, setSlateKey] = useState(0);

  /** HTML derived from Slate value — used for preview + PDF schema sync. */
  const [previewHtml, setPreviewHtml] = useState('');

  /** Live PdfDocument schema, kept in sync with every content change. */
  const [pdfDoc, setPdfDoc] = useState<PdfDocument>(() => createBlankDocument('Untitled Document'));

  // ── onChange: Slate → preview HTML + PdfDocument schema ──────────────────
  const handleSlateChange = useCallback(
    (value: Descendant[]) => {
      setSlateValue(value);
      const html = slateToHtml(value);
      setPreviewHtml(html);
      setPdfDoc((prev) =>
        htmlToPdfDocument(html, { title, fontFamily, fontSize, existingDoc: prev }),
      );
      setActiveFormats(getActiveFormats(editor));
    },
    [editor, title, fontFamily, fontSize],
  );

  // ── Title change ──────────────────────────────────────────────────────────
  const handleTitleChange = useCallback((value: string) => {
    setTitle(value);
    setPdfDoc((prev) => ({
      ...prev,
      meta: { ...prev.meta, title: value, updatedAt: new Date().toISOString() },
    }));
  }, []);

  // ── Toolbar: execCommand-style dispatch → Slate commands ─────────────────
  const exec = useCallback(
    (command: string, _value?: string) => {
      switch (command) {
        case 'bold':          toggleMark(editor, 'bold'); break;
        case 'italic':        toggleMark(editor, 'italic'); break;
        case 'underline':     toggleMark(editor, 'underline'); break;
        case 'strikeThrough': toggleMark(editor, 'strikethrough'); break;
        case 'justifyLeft':   setAlign(editor, 'left'); break;
        case 'justifyCenter': setAlign(editor, 'center'); break;
        case 'justifyRight':  setAlign(editor, 'right'); break;
        case 'justifyFull':   setAlign(editor, 'justify'); break;
        case 'insertUnorderedList': toggleBlock(editor, 'bulleted-list'); break;
        case 'insertOrderedList':   toggleBlock(editor, 'numbered-list'); break;
        case 'indent':   indentListItem(editor); break;
        case 'outdent':  outdentListItem(editor); break;
        case 'insertHorizontalRule': insertDivider(editor); break;
        case 'removeFormat': removeAllMarks(editor); break;
        case 'undo': editor.undo(); break;
        case 'redo': editor.redo(); break;
        default: break;
      }
      setActiveFormats(getActiveFormats(editor));
    },
    [editor],
  );

  // ── Block format (from heading/paragraph dropdown) ────────────────────────
  const handleBlockFormat = useCallback(
    (tag: string) => {
      setBlockFromTag(editor, tag);
      setActiveFormats(getActiveFormats(editor));
    },
    [editor],
  );

  // ── Font family ───────────────────────────────────────────────────────────
  const handleFontFamilyChange = useCallback(
    (family: string) => {
      setFontFamily(family);
      setMark(editor, 'fontFamily', family);
    },
    [editor],
  );

  // ── Font size ─────────────────────────────────────────────────────────────
  const handleFontSizeChange = useCallback(
    (size: string) => {
      setFontSize(size);
      setMark(editor, 'fontSize', size);
    },
    [editor],
  );

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;
      switch (e.key.toLowerCase()) {
        case 'b': e.preventDefault(); toggleMark(editor, 'bold'); break;
        case 'i': e.preventDefault(); toggleMark(editor, 'italic'); break;
        case 'u': e.preventDefault(); toggleMark(editor, 'underline'); break;
        default: break;
      }
      setActiveFormats(getActiveFormats(editor));
    },
    [editor],
  );

  // ── Export PDF ────────────────────────────────────────────────────────────
  const handleExport = useCallback(() => {
    exportPdf({ title, content: previewHtml, fontFamily, fontSize });
  }, [title, previewHtml, fontFamily, fontSize]);

  // ── Helper: load a PdfDocument into Slate ─────────────────────────────────
  const loadDoc = useCallback((doc: PdfDocument) => {
    const loadedTitle = doc.meta.title ?? 'Untitled Document';
    const loadedFont  = doc.defaultStyles?.fontFamily ?? 'Georgia, serif';
    const loadedSize  = doc.defaultStyles?.fontSize ? String(doc.defaultStyles.fontSize) : '12';

    setTitle(loadedTitle);
    setFontFamily(loadedFont);
    setFontSize(loadedSize);
    setPdfDoc(doc);

    const html = pdfDocumentToHtml(doc);
    const slateNodes = htmlToSlate(html);
    setSlateValue(slateNodes);
    setPreviewHtml(html);
    // Force Slate remount so it picks up the new initialValue
    setSlateKey((k) => k + 1);
  }, []);

  // ── Definition upload ─────────────────────────────────────────────────────
  const handleUploadDefinition = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result;
      if (typeof text !== 'string') return;
      const [doc, err] = safeParsePdfDocument(text);
      if (err || !doc) {
        alert(`Invalid document definition.\n\n${err?.issues.map((i) => i.message).join('\n') ?? 'Unknown error'}`);
        return;
      }
      loadDoc(doc);
    };
    reader.readAsText(file);
  }, [loadDoc]);

  // ── Minified-version upload ───────────────────────────────────────────────
  const handleUploadMinifiedVersion = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result;
      if (typeof text !== 'string') return;
      const [doc, err] = safeParseMinifiedComponents(text);
      if (err || !doc) {
        alert(`Invalid minified document.\n\n${err?.issues.map((i) => i.message).join('\n') ?? 'Unknown error'}`);
        return;
      }
      loadDoc(doc);
    };
    reader.readAsText(file);
  }, [loadDoc]);

  // ── JSON Schema download ──────────────────────────────────────────────────
  const handleDownloadJsonSchema = useCallback(() => {
    const schema = getMinifiedJsonSchema();
    const json = JSON.stringify(schema, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'bkwpdf-document-schema.json';
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  // ── Definition download ───────────────────────────────────────────────────
  const handleDownloadDefinition = useCallback(() => {
    const json = serializePdfDocument(pdfDoc);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const safeName = (pdfDoc.meta.title ?? 'document')
      .replace(/[^a-z0-9_\-. ]/gi, '_')
      .trim()
      .replace(/\s+/g, '_');
    a.href = url;
    a.download = `${safeName}.bkwpdf.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [pdfDoc]);

  return (
    <div className="flex flex-col flex-1 overflow-hidden bg-gray-50">
      <EditorHeader
        title={title}
        onTitleChange={handleTitleChange}
        onExport={handleExport}
        onUploadDefinition={handleUploadDefinition}
        onUploadMinifiedVersion={handleUploadMinifiedVersion}
        onDownloadDefinition={handleDownloadDefinition}
        onDownloadJsonSchema={handleDownloadJsonSchema}
      />

      <EditorToolbar
        activeFormats={activeFormats}
        fontFamily={fontFamily}
        fontSize={fontSize}
        onExec={exec}
        onBlockFormat={handleBlockFormat}
        onFontFamilyChange={handleFontFamilyChange}
        onFontSizeChange={handleFontSizeChange}
      />

      <div className="flex flex-1 overflow-hidden">
        <DocumentEditor
          editor={editor}
          slateKey={slateKey}
          initialValue={slateValue}
          fontFamily={fontFamily}
          fontSize={fontSize}
          onChange={handleSlateChange}
          onKeyDown={handleKeyDown}
        />

        <DocumentPreview
          previewHtml={previewHtml}
          title={title}
          fontFamily={fontFamily}
          fontSize={fontSize}
        />
      </div>
    </div>
  );
}

'use client';

import React, { useRef, useState, useCallback, useEffect } from 'react';
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
import { createBlankDocument, safeParsePdfDocument, serializePdfDocument } from './_utils/serializer';

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
  const editorRef = useRef<HTMLDivElement>(null);

  const [title, setTitle] = useState('Untitled Document');
  const [previewHtml, setPreviewHtml] = useState('');
  const [fontFamily, setFontFamily] = useState('Georgia, serif');
  const [fontSize, setFontSize] = useState('12');
  const [activeFormats, setActiveFormats] = useState<ActiveFormats>(DEFAULT_FORMATS);

  /** Live PdfDocument definition — kept in sync with every editor change. */
  const [pdfDoc, setPdfDoc] = useState<PdfDocument>(() => createBlankDocument('Untitled Document'));

  /* ── Sync preview HTML + schema from the editor DOM ── */
  const syncPreview = useCallback(() => {
    if (!editorRef.current) return;
    const html = editorRef.current.innerHTML;
    setPreviewHtml(html);
    // Update the schema definition in the background (uses current state via
    // functional update to avoid stale closure on pdfDoc)
    setPdfDoc((prev) =>
      htmlToPdfDocument(html, {
        title,
        fontFamily,
        fontSize,
        existingDoc: prev,
      }),
    );
  }, [title, fontFamily, fontSize]);

  /* ── Keep schema in sync when title changes ── */
  const handleTitleChange = useCallback((value: string) => {
    setTitle(value);
    setPdfDoc((prev) => ({
      ...prev,
      meta: { ...prev.meta, title: value, updatedAt: new Date().toISOString() },
    }));
  }, []);

  /* ── Refresh toolbar active-state indicators on every selection change ── */
  const updateActiveFormats = useCallback(() => {
    setActiveFormats({
      bold: document.queryCommandState('bold'),
      italic: document.queryCommandState('italic'),
      underline: document.queryCommandState('underline'),
      strikeThrough: document.queryCommandState('strikeThrough'),
      justifyLeft: document.queryCommandState('justifyLeft'),
      justifyCenter: document.queryCommandState('justifyCenter'),
      justifyRight: document.queryCommandState('justifyRight'),
      justifyFull: document.queryCommandState('justifyFull'),
      insertUnorderedList: document.queryCommandState('insertUnorderedList'),
      insertOrderedList: document.queryCommandState('insertOrderedList'),
    });
  }, []);

  useEffect(() => {
    document.addEventListener('selectionchange', updateActiveFormats);
    return () => document.removeEventListener('selectionchange', updateActiveFormats);
  }, [updateActiveFormats]);

  /* ── execCommand wrapper ── */
  const exec = useCallback(
    (command: string, value?: string) => {
      editorRef.current?.focus();
      document.execCommand(command, false, value);
      syncPreview();
      updateActiveFormats();
    },
    [syncPreview, updateActiveFormats],
  );

  /* ── Block/paragraph style ── */
  const handleBlockFormat = useCallback(
    (tag: string) => {
      exec('formatBlock', tag);
    },
    [exec],
  );

  /* ── Font family ── */
  const handleFontFamilyChange = useCallback(
    (family: string) => {
      setFontFamily(family);
      exec('fontName', family);
    },
    [exec],
  );

  /* ── Font size (execCommand only accepts 1-7; we inject a styled <span>) ── */
  const handleFontSizeChange = useCallback(
    (size: string) => {
      setFontSize(size);
      exec('fontSize', '7');
      editorRef.current?.querySelectorAll('font[size="7"]').forEach((el) => {
        const span = document.createElement('span');
        span.style.fontSize = `${size}pt`;
        span.innerHTML = el.innerHTML;
        el.replaceWith(span);
      });
      syncPreview();
    },
    [exec, syncPreview],
  );

  /* ── Export PDF ── */
  const handleExport = useCallback(() => {
    exportPdf({
      title,
      content: editorRef.current?.innerHTML ?? '',
      fontFamily,
      fontSize,
    });
  }, [title, fontFamily, fontSize]);

  // ── Definition upload ──────────────────────────────────────────────────────
  const handleUploadDefinition = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result;
      if (typeof text !== 'string') return;

      const [doc, err] = safeParsePdfDocument(text);
      if (err || !doc) {
        alert(
          `Invalid document definition.\n\n${err?.issues.map((i) => i.message).join('\n') ?? 'Unknown error'}`,
        );
        return;
      }

      // Restore title and default typography from the loaded document
      const loadedTitle = doc.meta.title ?? 'Untitled Document';
      const loadedFont = doc.defaultStyles?.fontFamily ?? 'Georgia, serif';
      const loadedSize = doc.defaultStyles?.fontSize
        ? String(doc.defaultStyles.fontSize)
        : '12';

      setTitle(loadedTitle);
      setFontFamily(loadedFont);
      setFontSize(loadedSize);
      setPdfDoc(doc);

      // Render the schema back to HTML and inject into the contentEditable editor
      const html = pdfDocumentToHtml(doc);
      if (editorRef.current) {
        editorRef.current.innerHTML = html || '<p><br></p>';
        setPreviewHtml(editorRef.current.innerHTML);
      }
    };
    reader.readAsText(file);
  }, []);

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

  // ── Definition download ────────────────────────────────────────────────────
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

  /* ── Ref callback: initialise the editor with an empty paragraph ── */
  const initEditor = useCallback((node: HTMLDivElement | null) => {
    if (node && !node.innerHTML.trim()) {
      node.innerHTML = '<p><br /></p>';
    }
    (editorRef as React.MutableRefObject<HTMLDivElement | null>).current = node;
  }, []);

  return (
    <div className="flex flex-col flex-1 overflow-hidden bg-gray-50">
      <EditorHeader
        title={title}
        onTitleChange={handleTitleChange}
        onExport={handleExport}
        onUploadDefinition={handleUploadDefinition}
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
          fontFamily={fontFamily}
          fontSize={fontSize}
          initEditor={initEditor}
          onInput={syncPreview}
          onKeyUp={updateActiveFormats}
          onMouseUp={updateActiveFormats}
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

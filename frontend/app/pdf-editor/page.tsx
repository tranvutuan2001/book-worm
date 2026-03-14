'use client';

import React, { useRef, useState, useCallback, useEffect } from 'react';
import EditorHeader from '@/app/pdf-editor/_components/EditorHeader';
import EditorToolbar, { ActiveFormats } from '@/app/pdf-editor/_components/EditorToolbar';
import DocumentEditor from '@/app/pdf-editor/_components/DocumentEditor';
import DocumentPreview from '@/app/pdf-editor/_components/DocumentPreview';
import { exportPdf } from '@/config/exportPdf';

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

  /* ── Sync preview HTML from the editor DOM ── */
  const syncPreview = useCallback(() => {
    if (editorRef.current) {
      setPreviewHtml(editorRef.current.innerHTML);
    }
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
  const exec = useCallback((command: string, value?: string) => {
    editorRef.current?.focus();
    document.execCommand(command, false, value);
    syncPreview();
    updateActiveFormats();
  }, [syncPreview, updateActiveFormats]);

  /* ── Block/paragraph style ── */
  const handleBlockFormat = useCallback((tag: string) => {
    exec('formatBlock', tag);
  }, [exec]);

  /* ── Font family ── */
  const handleFontFamilyChange = useCallback((family: string) => {
    setFontFamily(family);
    exec('fontName', family);
  }, [exec]);

  /* ── Font size (execCommand only accepts 1-7; we inject a styled <span>) ── */
  const handleFontSizeChange = useCallback((size: string) => {
    setFontSize(size);
    exec('fontSize', '7');
    editorRef.current?.querySelectorAll('font[size="7"]').forEach((el) => {
      const span = document.createElement('span');
      span.style.fontSize = `${size}pt`;
      span.innerHTML = el.innerHTML;
      el.replaceWith(span);
    });
    syncPreview();
  }, [exec, syncPreview]);

  /* ── Export ── */
  const handleExport = useCallback(() => {
    exportPdf({
      title,
      content: editorRef.current?.innerHTML ?? '',
      fontFamily,
      fontSize,
    });
  }, [title, fontFamily, fontSize]);

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
        onTitleChange={setTitle}
        onExport={handleExport}
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

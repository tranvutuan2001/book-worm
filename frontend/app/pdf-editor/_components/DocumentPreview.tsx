import React, { useRef, useEffect, useState } from 'react';
import { A4_W, A4_H } from '@/lib/pdf-editor/constants';

interface DocumentPreviewProps {
  previewHtml: string;
  title: string;
  fontFamily: string;
  fontSize: string;
}

export default function DocumentPreview({ previewHtml, title, fontFamily, fontSize }: DocumentPreviewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);

  useEffect(() => {
    const update = () => {
      if (!containerRef.current) return;
      const available = containerRef.current.clientWidth - 40; // 20px padding each side
      setScale(Math.min(1, available / A4_W));
    };
    update();
    const ro = new ResizeObserver(update);
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  return (
    <div
      ref={containerRef}
      className="w-[480px] xl:w-[560px] flex flex-col overflow-hidden bg-gray-200 shrink-0"
    >
      {/* Panel header */}
      <div className="bg-gray-300 px-4 py-1.5 text-xs font-medium text-gray-500 uppercase tracking-wide shrink-0 flex items-center justify-between">
        <span>Preview — A4 Portrait</span>
        <span className="text-gray-400">{Math.round(scale * 100)}%</span>
      </div>

      <div className="flex-1 overflow-y-auto py-6 px-5">
        {/* Scaled A4 page */}
        <div
          style={{
            width: A4_W,
            minHeight: A4_H,
            transform: `scale(${scale})`,
            transformOrigin: 'top left',
            // compensate height after scaling so scrollable area is correct
            marginBottom: `${A4_H * scale - A4_H + 24}px`,
          }}
          className="bg-white shadow-xl"
        >
          {/* Page content — 20mm ≈ 75px padding matches @page margin */}
          <div
            style={{ fontFamily, fontSize: `${fontSize}pt`, padding: '75px' }}
            className="text-gray-900 leading-relaxed"
          >
            <h1 style={{ marginBottom: '1.2em', fontWeight: 700, fontSize: '1.6em' }}>
              {title || 'Untitled Document'}
            </h1>
            <div
              dangerouslySetInnerHTML={{ __html: previewHtml }}
              className="preview-content"
            />
          </div>
        </div>
      </div>

      <style>{`
        .preview-content ul { list-style: disc; margin-left: 1.4em; }
        .preview-content ol { list-style: decimal; margin-left: 1.4em; }
        .preview-content h1 { font-size: 2em; font-weight: 700; margin: .6em 0; }
        .preview-content h2 { font-size: 1.5em; font-weight: 700; margin: .7em 0; }
        .preview-content h3 { font-size: 1.2em; font-weight: 600; margin: .8em 0; }
        .preview-content h4 { font-size: 1em; font-weight: 600; margin: .9em 0; }
        .preview-content p  { margin: .6em 0; }
        .preview-content blockquote {
          border-left: 4px solid #7c3aed;
          padding: .3em .8em;
          margin: .8em 0;
          color: #555;
          font-style: italic;
        }
        .preview-content hr { border: none; border-top: 1px solid #ccc; margin: 1em 0; }
        .preview-content pre,
        .preview-content code {
          font-family: 'Courier New', monospace;
          background: #f5f5f5;
          padding: .15em .4em;
          border-radius: 3px;
          font-size: .88em;
        }
        .preview-content table { border-collapse: collapse; width: 100%; margin: .8em 0; }
        .preview-content td,
        .preview-content th {
          border: 1px solid #ccc;
          padding: .3em .5em;
          font-size: .9em;
        }
        .preview-content th { background: #f3f0ff; font-weight: 600; }
      `}</style>
    </div>
  );
}

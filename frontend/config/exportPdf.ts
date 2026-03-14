import { A4_W, A4_H, PAGE_MARGIN_MM } from './pdf-editor-font';

export interface ExportPdfOptions {
  title: string;
  content: string;
  fontFamily: string;
  fontSize: string;
}

export function exportPdf({ title, content, fontFamily, fontSize }: ExportPdfOptions): void {
  const printWindow = window.open('', '_blank', 'width=900,height=700');
  if (!printWindow) return;

  printWindow.document.write(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>${title}</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    @page {
      size: A4 portrait;
      margin: ${PAGE_MARGIN_MM}mm;
    }

    body {
      font-family: ${fontFamily};
      font-size: ${fontSize}pt;
      color: #1a1a1a;
      background: #fff;
      line-height: 1.6;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }

    @media screen {
      html { background: #e8e8e8; }
      .page {
        width: ${A4_W}px;
        min-height: ${A4_H}px;
        background: #fff;
        margin: 32px auto;
        padding: ${PAGE_MARGIN_MM}mm;
        box-shadow: 0 4px 32px rgba(0,0,0,.18);
      }
    }

    h1 { font-size: 2em;    margin: .67em 0; font-weight: 700; }
    h2 { font-size: 1.5em;  margin: .75em 0; font-weight: 700; }
    h3 { font-size: 1.17em; margin: .83em 0; font-weight: 600; }
    h4 { font-size: 1em;    margin: 1.12em 0; font-weight: 600; }
    p  { margin: .8em 0; }
    ul, ol { margin: .8em 0 .8em 1.5em; }
    li { margin: .3em 0; }
    blockquote {
      border-left: 4px solid #7c3aed;
      padding: .4em 1em;
      margin: 1em 0;
      color: #555;
      font-style: italic;
    }
    hr { border: none; border-top: 1px solid #ccc; margin: 1.2em 0; }
    table { border-collapse: collapse; width: 100%; margin: 1em 0; }
    td, th { border: 1px solid #ccc; padding: .4em .6em; }
    th { background: #f3f0ff; font-weight: 600; }
    img { max-width: 100%; height: auto; }
    a { color: #7c3aed; text-decoration: underline; }
    pre, code {
      font-family: 'Courier New', monospace;
      background: #f5f5f5;
      padding: .1em .3em;
      border-radius: 3px;
    }

    @media print {
      .page { box-shadow: none; margin: 0; padding: 0; width: auto; }
    }
  </style>
</head>
<body>
  <div class="page">
    <h1 style="margin-bottom:1.4em">${title}</h1>
    ${content}
  </div>
  <script>
    window.onload = function() { window.print(); };
  <\/script>
</body>
</html>`);

  printWindow.document.close();
}

import { PAGE_MARGIN_MM } from './pdf-editor-font';

export interface ExportPdfOptions {
  title: string;
  content: string;
  fontFamily: string;
  fontSize: string;
}

/**
 * Exports the document by injecting a temporary print stylesheet and a
 * hidden print-only container into the current page, then calling the
 * browser's native window.print(). Everything is cleaned up after printing.
 */
export function exportPdf({ title, content, fontFamily, fontSize }: ExportPdfOptions): void {
  const PRINT_CONTAINER_ID = '__pdf_print_root__';
  const PRINT_STYLE_ID = '__pdf_print_style__';

  // Remove any leftover elements from a previous call
  document.getElementById(PRINT_CONTAINER_ID)?.remove();
  document.getElementById(PRINT_STYLE_ID)?.remove();

  // ── Print stylesheet ──────────────────────────────────────────────────────
  const style = document.createElement('style');
  style.id = PRINT_STYLE_ID;
  style.textContent = `
    @page {
      size: A4 portrait;
      margin: ${PAGE_MARGIN_MM}mm;
    }

    /* Hide everything on the page when printing… */
    @media print {
      body > *:not(#${PRINT_CONTAINER_ID}) {
        display: none !important;
      }

      #${PRINT_CONTAINER_ID} {
        display: block !important;
        font-family: ${fontFamily};
        font-size: ${fontSize}pt;
        color: #1a1a1a;
        background: #fff;
        line-height: 1.6;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
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
    }
  `;
  document.head.appendChild(style);

  // ── Print container (hidden from screen, visible only when printing) ──────
  const container = document.createElement('div');
  container.id = PRINT_CONTAINER_ID;
  container.style.cssText = 'display:none;';
  container.innerHTML = `<h1 style="margin-bottom:1.4em">${title}</h1>${content}`;
  document.body.appendChild(container);

  // ── Trigger browser print dialog ──────────────────────────────────────────
  window.print();

  // ── Clean up after the dialog closes ─────────────────────────────────────
  container.remove();
  style.remove();
}

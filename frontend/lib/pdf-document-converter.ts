/**
 * Bidirectional converter between the PdfDocument schema and plain HTML.
 *
 * pdfDocumentToHtml  — renders a PdfDocument into an HTML string suitable for
 *                      the contentEditable editor.
 *
 * htmlToPdfDocument  — parses contentEditable innerHTML back into a
 *                      PdfDocument, optionally merging metadata and settings
 *                      from an existing document.
 *
 * NOTE: This module must only run in a browser context (it uses DOMParser).
 */

import type {
  PdfDocument,
  PdfPage,
  PdfSection,
  PdfContentElement,
  PdfTextRun,
} from './pdf-document-schema';

// ─────────────────────────────────────────────────────────────────────────────
// PdfDocument → HTML
// ─────────────────────────────────────────────────────────────────────────────

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderRuns(runs: PdfTextRun[]): string {
  return runs
    .map((run) => {
      let html = escapeHtml(run.text).replace(/\n/g, '<br>');

      // Inline style overrides
      const styles: string[] = [];
      if (run.color) styles.push(`color:${run.color}`);
      if (run.fontSize) styles.push(`font-size:${run.fontSize}pt`);
      if (run.fontFamily) styles.push(`font-family:${run.fontFamily}`);
      if (run.highlight) styles.push(`background-color:${run.highlight}`);
      if (run.verticalAlign === 'super') styles.push('vertical-align:super;font-size:0.75em');
      if (run.verticalAlign === 'sub') styles.push('vertical-align:sub;font-size:0.75em');
      if (styles.length) html = `<span style="${styles.join(';')}">${html}</span>`;

      // Semantic wrappers (applied after style span so they sit outside it)
      if (run.strikethrough) html = `<s>${html}</s>`;
      if (run.underline) html = `<u>${html}</u>`;
      if (run.italic) html = `<em>${html}</em>`;
      if (run.bold) html = `<strong>${html}</strong>`;
      if (run.link) html = `<a href="${escapeHtml(run.link)}">${html}</a>`;

      return html;
    })
    .join('');
}

type ListItem = { runs: PdfTextRun[]; children?: ListItem[] };

function renderListItems(items: ListItem[], ordered: boolean): string {
  const tag = ordered ? 'ol' : 'ul';
  const inner = items
    .map((item) => {
      let li = `<li>${renderRuns(item.runs)}`;
      if (item.children?.length) li += renderListItems(item.children, ordered);
      return li + '</li>';
    })
    .join('');
  return `<${tag}>${inner}</${tag}>`;
}

function renderElement(el: PdfContentElement): string {
  switch (el.type) {
    case 'heading':
      return `<h${el.level}>${renderRuns(el.runs)}</h${el.level}>`;

    case 'paragraph':
      return el.runs.length === 0 || (el.runs.length === 1 && el.runs[0].text === '')
        ? '<p><br></p>'
        : `<p>${renderRuns(el.runs)}</p>`;

    case 'image': {
      const attrs = [`src="${escapeHtml(el.src)}"`];
      if (el.alt) attrs.push(`alt="${escapeHtml(el.alt)}"`);
      if (el.width) attrs.push(`style="width:${el.width}mm"`);
      const img = `<img ${attrs.join(' ')}>`;
      return el.caption
        ? `<figure>${img}<figcaption>${renderRuns(el.caption)}</figcaption></figure>`
        : img;
    }

    case 'table': {
      const rows = el.rows
        .map((row) => {
          const cells = row.cells
            .map((cell) => {
              const cellTag = row.isHeader ? 'th' : 'td';
              const cs = (cell.colSpan ?? 1) > 1 ? ` colspan="${cell.colSpan}"` : '';
              const rs = (cell.rowSpan ?? 1) > 1 ? ` rowspan="${cell.rowSpan}"` : '';
              return `<${cellTag}${cs}${rs}>${renderRuns(cell.runs)}</${cellTag}>`;
            })
            .join('');
          return `<tr>${cells}</tr>`;
        })
        .join('');
      return `<table>${rows}</table>`;
    }

    case 'list':
      return renderListItems(el.items as ListItem[], el.ordered);

    case 'code': {
      const lang = el.language ? ` class="language-${el.language}"` : '';
      const safe = escapeHtml(el.content);
      return `<pre><code${lang}>${safe}</code></pre>`;
    }

    case 'blockquote': {
      const attr = el.attribution
        ? `<cite>${renderRuns(el.attribution)}</cite>`
        : '';
      return `<blockquote>${renderRuns(el.runs)}${attr}</blockquote>`;
    }

    case 'divider':
      return '<hr>';

    case 'spacer':
      return `<div style="height:${el.height}mm"></div>`;

    case 'pageBreak':
      return '<div style="page-break-after:always"></div>';

    default:
      return '';
  }
}

function renderSection(section: PdfSection): string {
  return section.children
    .map((child) =>
      'layout' in child
        ? renderSection(child as PdfSection)
        : renderElement(child as PdfContentElement),
    )
    .join('');
}

/** Renders a PdfDocument into an HTML string for the contentEditable editor. */
export function pdfDocumentToHtml(doc: PdfDocument): string {
  return doc.pages
    .map((page: PdfPage) => page.sections.map(renderSection).join(''))
    .join('<div style="page-break-after:always"></div>');
}

// ─────────────────────────────────────────────────────────────────────────────
// HTML → PdfDocument
// ─────────────────────────────────────────────────────────────────────────────

function parseRuns(container: Node): PdfTextRun[] {
  const runs: PdfTextRun[] = [];

  function walk(node: Node, ctx: Partial<PdfTextRun>): void {
    if (node.nodeType === Node.TEXT_NODE) {
      const text = node.textContent ?? '';
      if (text) runs.push({ ...ctx, text } as PdfTextRun);
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return;

    const el = node as HTMLElement;
    const tag = el.tagName.toLowerCase();
    const next: Partial<PdfTextRun> = { ...ctx };

    if (tag === 'br') { runs.push({ text: '\n' }); return; }
    if (tag === 'strong' || tag === 'b') next.bold = true;
    if (tag === 'em' || tag === 'i') next.italic = true;
    if (tag === 'u') next.underline = true;
    if (tag === 's' || tag === 'del' || tag === 'strike') next.strikethrough = true;
    if (tag === 'sup') next.verticalAlign = 'super';
    if (tag === 'sub') next.verticalAlign = 'sub';
    if (tag === 'a') next.link = (el as HTMLAnchorElement).href || undefined;

    if (tag === 'span' || tag === 'font') {
      const s = el.style;
      if (s.color) next.color = s.color;
      if (s.backgroundColor) next.highlight = s.backgroundColor;
      if (s.fontFamily) next.fontFamily = s.fontFamily;
      const ptMatch = s.fontSize?.match(/([\d.]+)pt/);
      if (ptMatch) next.fontSize = parseFloat(ptMatch[1]);
      if (s.verticalAlign === 'super') next.verticalAlign = 'super';
      if (s.verticalAlign === 'sub') next.verticalAlign = 'sub';
    }

    el.childNodes.forEach((child) => walk(child, next));
  }

  container.childNodes.forEach((child) => walk(child, {}));
  return runs.filter((r) => r.text !== undefined);
}

/** Parses a <ul> or <ol> into a list of PdfListItems (recursive). */
function parseListItems(list: Element): ListItem[] {
  const items: ListItem[] = [];
  list.querySelectorAll(':scope > li').forEach((li) => {
    const nested = li.querySelector(':scope > ul, :scope > ol');
    // Collect text nodes / inline elements excluding the nested list
    const textNodes: ChildNode[] = [];
    li.childNodes.forEach((n) => { if (n !== nested) textNodes.push(n); });
    // Build a temporary fragment to parse runs
    const frag = document.createDocumentFragment();
    textNodes.forEach((n) => frag.appendChild(n.cloneNode(true)));
    const tmpDiv = document.createElement('div');
    tmpDiv.appendChild(frag);
    const runs = parseRuns(tmpDiv);
    const item: ListItem = { runs: runs.length ? runs : [{ text: '' }] };
    if (nested) item.children = parseListItems(nested);
    items.push(item);
  });
  return items;
}

function parseBlockNode(node: Node): PdfContentElement | null {
  if (node.nodeType !== Node.ELEMENT_NODE) {
    // Top-level text nodes → paragraph
    const text = node.textContent?.trim();
    return text ? { type: 'paragraph', runs: [{ text }] } : null;
  }

  const el = node as HTMLElement;
  const tag = el.tagName.toLowerCase();

  // Headings
  if (/^h[1-6]$/.test(tag)) {
    return {
      type: 'heading',
      level: parseInt(tag[1]) as 1 | 2 | 3 | 4 | 5 | 6,
      runs: parseRuns(el),
      includeInToc: true,
    };
  }

  // Paragraph
  if (tag === 'p') {
    return { type: 'paragraph', runs: parseRuns(el) };
  }

  // Lists
  if (tag === 'ul' || tag === 'ol') {
    const items = parseListItems(el);
    return {
      type: 'list',
      ordered: tag === 'ol',
      start: 1,
      items,
    };
  }

  // Blockquote
  if (tag === 'blockquote') {
    const cite = el.querySelector('cite');
    const textNodes: ChildNode[] = [];
    el.childNodes.forEach((n) => { if (n !== cite) textNodes.push(n); });
    const frag = document.createDocumentFragment();
    textNodes.forEach((n) => frag.appendChild(n.cloneNode(true)));
    const tmpDiv = document.createElement('div');
    tmpDiv.appendChild(frag);
    const runs = parseRuns(tmpDiv);
    return {
      type: 'blockquote',
      runs: runs.length ? runs : [{ text: '' }],
      attribution: cite ? parseRuns(cite) : undefined,
    };
  }

  // Code block
  if (tag === 'pre') {
    const codeEl = el.querySelector('code');
    const langMatch = (codeEl?.className ?? '').match(/language-(\S+)/);
    return {
      type: 'code',
      content: (codeEl ?? el).textContent ?? '',
      language: langMatch?.[1],
      lineNumbers: false,
    };
  }

  // Horizontal rule
  if (tag === 'hr') return { type: 'divider', widthPercent: 100 };

  // Image
  if (tag === 'img') {
    const img = el as HTMLImageElement;
    return {
      type: 'image',
      src: img.getAttribute('src') ?? img.src,
      alt: img.alt || undefined,
    };
  }

  // Figure with image + caption
  if (tag === 'figure') {
    const img = el.querySelector('img') as HTMLImageElement | null;
    const caption = el.querySelector('figcaption');
    if (img) {
      return {
        type: 'image',
        src: img.getAttribute('src') ?? img.src,
        alt: img.alt || undefined,
        caption: caption ? parseRuns(caption) : undefined,
      };
    }
  }

  // Table
  if (tag === 'table') {
    const rows = Array.from(el.querySelectorAll('tr')).map((tr) => {
      const isHeader = !!tr.closest('thead') || tr.querySelector('th') !== null;
      const cells = Array.from(tr.querySelectorAll('td, th')).map((cell) => ({
        runs: parseRuns(cell),
        colSpan: parseInt(cell.getAttribute('colspan') ?? '1') || 1,
        rowSpan: parseInt(cell.getAttribute('rowspan') ?? '1') || 1,
      }));
      return { cells, isHeader };
    });
    if (rows.length) return { type: 'table' as const, rows, repeatHeader: true };
  }

  // Spacer / page break divs
  if (tag === 'div') {
    const s = el.style;
    if (s.pageBreakAfter === 'always' || s.breakAfter === 'page') {
      return { type: 'pageBreak' };
    }
    const mmMatch = s.height?.match(/([\d.]+)mm/);
    if (mmMatch) return { type: 'spacer', height: parseFloat(mmMatch[1]) };

    // Recurse: a div might contain block children (e.g. pasted content)
    const children: PdfContentElement[] = [];
    el.childNodes.forEach((child) => {
      const parsed = parseBlockNode(child);
      if (parsed) children.push(parsed);
    });
    if (children.length === 1) return children[0];
    if (children.length > 1) {
      // Wrap multiple children as a paragraph (best-effort)
      return { type: 'paragraph', runs: parseRuns(el) };
    }
  }

  // Default fallback: treat as paragraph
  const runs = parseRuns(el);
  if (runs.length) return { type: 'paragraph', runs };

  return null;
}

export interface HtmlToDocOptions {
  /** New document title. Falls back to existing doc title or 'Untitled Document'. */
  title?: string;
  /** Default font family (stored in defaultStyles). */
  fontFamily?: string;
  /** Default font size in pt (stored in defaultStyles). */
  fontSize?: string;
  /**
   * Existing document whose metadata, settings and extra pages are preserved.
   * Only page[0] content is replaced.
   */
  existingDoc?: PdfDocument;
}

/**
 * Parses contentEditable innerHTML into a `PdfDocument`.
 * All content becomes the children of the first page's first block section.
 */
export function htmlToPdfDocument(html: string, options: HtmlToDocOptions = {}): PdfDocument {
  const { title, fontFamily, fontSize, existingDoc } = options;

  // Parse HTML in a sandboxed document
  const parser = new DOMParser();
  const parsed = parser.parseFromString(`<div id="r">${html}</div>`, 'text/html');
  const root = parsed.getElementById('r')!;

  const elements: PdfContentElement[] = [];
  root.childNodes.forEach((node) => {
    const el = parseBlockNode(node);
    if (el) elements.push(el);
  });
  if (elements.length === 0) {
    elements.push({ type: 'paragraph', runs: [{ text: '' }] });
  }

  const now = new Date().toISOString();

  // Default styles: merge font changes onto existing
  const existingDefaults = existingDoc?.defaultStyles ?? {};
  const defaultStyles = {
    ...existingDefaults,
    ...(fontFamily ? { fontFamily } : {}),
    ...(fontSize ? { fontSize: parseFloat(fontSize) } : {}),
  };

  // Build the first page (content replaced, layout metadata preserved)
  const firstPage: PdfPage = {
    id: existingDoc?.pages[0]?.id ?? crypto.randomUUID(),
    settings: existingDoc?.pages[0]?.settings,
    background: existingDoc?.pages[0]?.background,
    sections: [
      {
        id: existingDoc?.pages[0]?.sections[0]?.id ?? crypto.randomUUID(),
        layout: 'block',
        children: elements,
      },
    ],
  };

  // Extra pages beyond the first are preserved as-is
  const extraPages = existingDoc?.pages.slice(1) ?? [];

  return {
    schemaVersion: '1.0',
    id: existingDoc?.id ?? crypto.randomUUID(),
    meta: {
      language: 'en',
      ...existingDoc?.meta,
      title: title ?? existingDoc?.meta.title ?? 'Untitled Document',
      updatedAt: now,
      ...(existingDoc ? {} : { createdAt: now }),
    },
    pageSettings: existingDoc?.pageSettings ?? {
      size: 'A4',
      orientation: 'portrait',
      margins: { top: 20, right: 20, bottom: 20, left: 20 },
    },
    defaultStyles: Object.keys(defaultStyles).length ? defaultStyles : undefined,
    namedStyles: existingDoc?.namedStyles,
    pages: [firstPage, ...extraPages],
  };
}

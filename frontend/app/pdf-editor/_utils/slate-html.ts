/**
 * Bidirectional converter between Slate's Descendant tree and plain HTML.
 *
 * slateToHtml   — serialises Slate nodes → HTML string (for preview / pdfDoc sync)
 * htmlToSlate   — parses an HTML string → Slate Descendant[] (for document loading)
 *
 * NOTE: htmlToSlate uses DOMParser and must only run in a browser context.
 */

import { Text, type Descendant } from 'slate';
import type { CustomElement, CustomText, BlockType, AlignType } from './slate-types';

// ─────────────────────────────────────────────────────────────────────────────
// Slate → HTML
// ─────────────────────────────────────────────────────────────────────────────

function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function serializeLeaf(leaf: CustomText): string {
  let text = escapeHtml(leaf.text).replace(/\n/g, '<br>');

  const styles: string[] = [];
  if (leaf.fontFamily) styles.push(`font-family:${leaf.fontFamily}`);
  if (leaf.fontSize) styles.push(`font-size:${leaf.fontSize}pt`);
  if (leaf.color) styles.push(`color:${leaf.color}`);
  if (leaf.highlight) styles.push(`background-color:${leaf.highlight}`);
  if (leaf.verticalAlign === 'super') styles.push('vertical-align:super;font-size:0.75em');
  if (leaf.verticalAlign === 'sub') styles.push('vertical-align:sub;font-size:0.75em');
  if (styles.length) text = `<span style="${styles.join(';')}">${text}</span>`;

  if (leaf.strikethrough) text = `<s>${text}</s>`;
  if (leaf.underline) text = `<u>${text}</u>`;
  if (leaf.italic) text = `<em>${text}</em>`;
  if (leaf.bold) text = `<strong>${text}</strong>`;

  return text;
}

function serializeChildren(children: Descendant[]): string {
  return children.map(serializeNode).join('');
}

function serializeNode(node: Descendant): string {
  if (Text.isText(node)) return serializeLeaf(node as CustomText);

  const el = node as CustomElement;
  const inner = serializeChildren(el.children);
  const alignStyle = el.align && el.align !== 'left' ? ` style="text-align:${el.align}"` : '';

  switch (el.type) {
    case 'paragraph':
      return `<p${alignStyle}>${inner || '<br>'}</p>`;
    case 'heading-one':
      return `<h1${alignStyle}>${inner}</h1>`;
    case 'heading-two':
      return `<h2${alignStyle}>${inner}</h2>`;
    case 'heading-three':
      return `<h3${alignStyle}>${inner}</h3>`;
    case 'heading-four':
      return `<h4${alignStyle}>${inner}</h4>`;
    case 'blockquote':
      return `<blockquote>${inner}</blockquote>`;
    case 'code-block':
      return `<pre><code>${inner}</code></pre>`;
    case 'bulleted-list':
      return `<ul>${inner}</ul>`;
    case 'numbered-list':
      return `<ol>${inner}</ol>`;
    case 'list-item':
      return `<li${alignStyle}>${inner}</li>`;
    case 'divider':
      return '<hr>';
    case 'raw-html':
      return el.rawHtml ?? '';
    default:
      return `<p>${inner}</p>`;
  }
}

/** Serialises a Slate document to an HTML string. */
export function slateToHtml(nodes: Descendant[]): string {
  return nodes.map(serializeNode).join('');
}

// ─────────────────────────────────────────────────────────────────────────────
// HTML → Slate
// ─────────────────────────────────────────────────────────────────────────────

function getAlign(el: HTMLElement): AlignType | undefined {
  const a = el.style?.textAlign;
  if (a === 'center' || a === 'right' || a === 'justify') return a as AlignType;
  return undefined;
}

/** Flatten inline DOM nodes to an array of CustomText leaves. */
function deserializeInline(node: Node, marks: Partial<CustomText> = {}): CustomText[] {
  if (node.nodeType === Node.TEXT_NODE) {
    const text = node.textContent ?? '';
    // Slate requires at least one text node per element; skip purely-empty ones
    // only when they aren't meaningful whitespace.
    if (!text) return [];
    return [{ text, ...marks }];
  }
  if (node.nodeType !== Node.ELEMENT_NODE) return [];

  const el = node as HTMLElement;
  const tag = el.tagName.toLowerCase();

  if (tag === 'br') return [{ text: '\n', ...marks }];

  const next: Partial<CustomText> = { ...marks };
  if (tag === 'strong' || tag === 'b') next.bold = true;
  if (tag === 'em' || tag === 'i') next.italic = true;
  if (tag === 'u') next.underline = true;
  if (tag === 's' || tag === 'del' || tag === 'strike') next.strikethrough = true;
  if (tag === 'sup') next.verticalAlign = 'super';
  if (tag === 'sub') next.verticalAlign = 'sub';
  if (tag === 'span' || tag === 'font') {
    const s = el.style;
    if (s.color) next.color = s.color;
    if (s.backgroundColor) next.highlight = s.backgroundColor;
    if (s.fontFamily) next.fontFamily = s.fontFamily;
    const ptMatch = s.fontSize?.match(/([\d.]+)pt/);
    if (ptMatch) next.fontSize = ptMatch[1];
    if (s.verticalAlign === 'super') next.verticalAlign = 'super';
    if (s.verticalAlign === 'sub') next.verticalAlign = 'sub';
  }

  return Array.from(el.childNodes).flatMap((child) => deserializeInline(child, next));
}

function inlineChildren(el: HTMLElement): CustomText[] {
  const leaves = Array.from(el.childNodes).flatMap((n) => deserializeInline(n));
  return leaves.length ? leaves : [{ text: '' }];
}

/** Recursively convert block-level DOM nodes to Slate elements. */
function deserializeBlock(node: Node): Descendant[] {
  if (node.nodeType === Node.TEXT_NODE) {
    const text = (node.textContent ?? '').trim();
    if (!text) return [];
    return [{ type: 'paragraph', children: [{ text }] }];
  }
  if (node.nodeType !== Node.ELEMENT_NODE) return [];

  const el = node as HTMLElement;
  const tag = el.tagName.toLowerCase();

  switch (tag) {
    case 'p':
      return [{ type: 'paragraph', align: getAlign(el), children: inlineChildren(el) }];
    case 'h1':
      return [{ type: 'heading-one', children: inlineChildren(el) }];
    case 'h2':
      return [{ type: 'heading-two', children: inlineChildren(el) }];
    case 'h3':
      return [{ type: 'heading-three', children: inlineChildren(el) }];
    case 'h4':
      return [{ type: 'heading-four', children: inlineChildren(el) }];
    case 'blockquote':
      return [{ type: 'blockquote', children: inlineChildren(el) }];
    case 'pre': {
      // Strip inner <code> wrapper if present
      const codeEl = el.querySelector('code') ?? el;
      return [{ type: 'code-block', children: inlineChildren(codeEl as HTMLElement) }];
    }
    case 'hr':
      return [{ type: 'divider', children: [{ text: '' }] }];
    case 'ul': {
      const items = Array.from(el.children).map((li): CustomElement => ({
        type: 'list-item',
        children: inlineChildren(li as HTMLElement),
      }));
      return [{ type: 'bulleted-list', children: items.length ? items : [{ type: 'list-item', children: [{ text: '' }] }] }];
    }
    case 'ol': {
      const items = Array.from(el.children).map((li): CustomElement => ({
        type: 'list-item',
        children: inlineChildren(li as HTMLElement),
      }));
      return [{ type: 'numbered-list', children: items.length ? items : [{ type: 'list-item', children: [{ text: '' }] }] }];
    }
    // Complex block elements — preserve as opaque raw-html void nodes
    case 'table':
    case 'figure':
      return [{ type: 'raw-html', rawHtml: el.outerHTML, children: [{ text: '' }] }];
    default: {
      // <div> wrappers for charts or page-breaks — keep as raw-html if they
      // carry special data attributes or inline styles; otherwise recurse.
      if (el.tagName.toLowerCase() === 'div' &&
          (el.dataset.chartBlock || el.classList.contains('pdf-chart-wrapper') ||
           el.style.pageBreakAfter || el.style.height)) {
        return [{ type: 'raw-html', rawHtml: el.outerHTML, children: [{ text: '' }] }];
      }
      return Array.from(el.childNodes).flatMap(deserializeBlock);
    }
  }
}

/** Parses an HTML string into a Slate Descendant array. */
export function htmlToSlate(html: string): Descendant[] {
  if (!html) return [{ type: 'paragraph', children: [{ text: '' }] }];

  const parser = new DOMParser();
  const doc = parser.parseFromString(html, 'text/html');
  const nodes = Array.from(doc.body.childNodes).flatMap(deserializeBlock);

  // Slate requires at least one top-level block
  return nodes.length ? nodes : [{ type: 'paragraph', children: [{ text: '' }] }];
}

/** Default empty Slate document. */
export const EMPTY_SLATE_VALUE: Descendant[] = [
  { type: 'paragraph', children: [{ text: '' }] },
];

/** Map an HTML heading/paragraph tag name to the Slate BlockType. */
export function tagToBlockType(tag: string): BlockType {
  switch (tag) {
    case 'h1': return 'heading-one';
    case 'h2': return 'heading-two';
    case 'h3': return 'heading-three';
    case 'h4': return 'heading-four';
    case 'blockquote': return 'blockquote';
    case 'pre': return 'code-block';
    default: return 'paragraph';
  }
}

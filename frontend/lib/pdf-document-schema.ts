/**
 * PDF Document Schema
 *
 * A document is a flat list of blocks rendered top-to-bottom.  The browser /
 * renderer automatically splits the content into pages based on `pageSettings`.
 *
 * To prevent a block from being cut across pages set `breakInside: 'avoid'`.
 * Images, tables, code blocks, and blockquotes carry a renderer-level default
 * of `breakInside: 'avoid'` even when the field is omitted.
 *
 * All measurements are in millimetres (mm) unless otherwise noted.
 */

import { z } from 'zod';

// ─────────────────────────────────────────────────────────────────────────────
// Primitives
// ─────────────────────────────────────────────────────────────────────────────

export const PdfColorSchema = z.string().describe('CSS-compatible color string');
export const PdfMmSchema    = z.number().describe('Measurement in mm');

export const PdfSpacingSchema = z.object({
  top:    PdfMmSchema.default(0),
  right:  PdfMmSchema.default(0),
  bottom: PdfMmSchema.default(0),
  left:   PdfMmSchema.default(0),
});

export const PdfBorderSchema = z.object({
  width: PdfMmSchema.default(0.3),
  style: z.enum(['none', 'solid', 'dashed', 'dotted', 'double']).default('solid'),
  color: PdfColorSchema.default('#cccccc'),
});

export const PdfBorderSidesSchema = z.object({
  top:   PdfBorderSchema.optional(),
  right: PdfBorderSchema.optional(),
  bottom: PdfBorderSchema.optional(),
  left:  PdfBorderSchema.optional(),
  all:   PdfBorderSchema.optional(),
});

// ─────────────────────────────────────────────────────────────────────────────
// Typography
// ─────────────────────────────────────────────────────────────────────────────

export const PdfTypographySchema = z.object({
  fontFamily:     z.string().optional(),
  /** Point size (pt). */
  fontSize:       z.number().positive().optional(),
  fontWeight:     z.union([z.number().int().min(100).max(900), z.enum(['normal', 'bold', 'lighter', 'bolder'])]).optional(),
  fontStyle:      z.enum(['normal', 'italic', 'oblique']).optional(),
  color:          PdfColorSchema.optional(),
  textAlign:      z.enum(['left', 'center', 'right', 'justify']).optional(),
  textDecoration: z.enum(['none', 'underline', 'line-through', 'overline']).optional(),
  textTransform:  z.enum(['none', 'uppercase', 'lowercase', 'capitalize']).optional(),
  /** Unitless multiplier (e.g. 1.5). */
  lineHeight:     z.number().positive().optional(),
  letterSpacing:  z.number().optional(),
  wordSpacing:    z.number().optional(),
});

// ─────────────────────────────────────────────────────────────────────────────
// Inline text runs
// ─────────────────────────────────────────────────────────────────────────────

export const PdfTextRunSchema = z.object({
  text:          z.string(),
  bold:          z.boolean().optional(),
  italic:        z.boolean().optional(),
  underline:     z.boolean().optional(),
  strikethrough: z.boolean().optional(),
  color:         PdfColorSchema.optional(),
  fontSize:      z.number().positive().optional(),
  fontFamily:    z.string().optional(),
  verticalAlign: z.enum(['baseline', 'super', 'sub']).optional(),
  link:          z.string().url().optional(),
  highlight:     PdfColorSchema.optional(),
});

export type PdfTextRun = z.infer<typeof PdfTextRunSchema>;

// ─────────────────────────────────────────────────────────────────────────────
// Page-break control
// ─────────────────────────────────────────────────────────────────────────────

/**
 * `breakInside: 'avoid'` — keep this block whole on a single page.
 * The renderer applies this automatically to images, tables, code, blockquotes.
 */
export const PdfBreakInsideSchema = z.enum(['auto', 'avoid']);
export const PdfBreakEdgeSchema   = z.enum(['auto', 'always', 'avoid']);

export type PdfBreakInside = z.infer<typeof PdfBreakInsideSchema>;
export type PdfBreakEdge   = z.infer<typeof PdfBreakEdgeSchema>;

/** Fields shared by every block. */
const blockBase = {
  margin:      PdfSpacingSchema.optional(),
  padding:     PdfSpacingSchema.optional(),
  /**
   * Prevent this block from being cut across two pages.
   * When omitted the renderer uses 'avoid' for images, tables, code, and
   * blockquotes; 'auto' for all other blocks.
   */
  breakInside: PdfBreakInsideSchema.optional(),
  breakBefore: PdfBreakEdgeSchema.optional(),
  breakAfter:  PdfBreakEdgeSchema.optional(),
};

// ─────────────────────────────────────────────────────────────────────────────
// Block types
// ─────────────────────────────────────────────────────────────────────────────

export const PdfHeadingSchema = z.object({
  ...blockBase,
  type:         z.literal('heading'),
  level:        z.union([z.literal(1), z.literal(2), z.literal(3), z.literal(4), z.literal(5), z.literal(6)]),
  runs:         z.array(PdfTextRunSchema).min(1),
  typography:   PdfTypographySchema.optional(),
  includeInToc: z.boolean().default(true),
});

export const PdfParagraphSchema = z.object({
  ...blockBase,
  type:       z.literal('paragraph'),
  runs:       z.array(PdfTextRunSchema),
  typography: PdfTypographySchema.optional(),
  textIndent: PdfMmSchema.optional(),
});

export const PdfImageSchema = z.object({
  ...blockBase,
  type:         z.literal('image'),
  /** File path, URL, or base-64 data URI. */
  src:          z.string(),
  alt:          z.string().optional(),
  /** Width in mm; omit to fill the available width. */
  width:        PdfMmSchema.optional(),
  /** Height in mm; omit to preserve aspect ratio. */
  height:       PdfMmSchema.optional(),
  objectFit:    z.enum(['fill', 'contain', 'cover', 'none', 'scale-down']).optional(),
  align:        z.enum(['left', 'center', 'right']).optional(),
  caption:      z.array(PdfTextRunSchema).optional(),
  border:       PdfBorderSchema.optional(),
  borderRadius: PdfMmSchema.optional(),
  opacity:      z.number().min(0).max(1).optional(),
});

export const PdfTableCellSchema = z.object({
  runs:          z.array(PdfTextRunSchema),
  typography:    PdfTypographySchema.optional(),
  padding:       PdfSpacingSchema.optional(),
  background:    PdfColorSchema.optional(),
  border:        PdfBorderSidesSchema.optional(),
  colSpan:       z.number().int().positive().default(1),
  rowSpan:       z.number().int().positive().default(1),
  verticalAlign: z.enum(['top', 'middle', 'bottom']).optional(),
});

export const PdfTableRowSchema = z.object({
  cells:      z.array(PdfTableCellSchema),
  background: PdfColorSchema.optional(),
  height:     PdfMmSchema.optional(),
  isHeader:   z.boolean().default(false),
});

export const PdfTableSchema = z.object({
  ...blockBase,
  type:         z.literal('table'),
  rows:         z.array(PdfTableRowSchema).min(1),
  columnWidths: z.array(PdfMmSchema).optional(),
  cellPadding:  PdfSpacingSchema.optional(),
  border:       PdfBorderSchema.optional(),
  caption:      z.array(PdfTextRunSchema).optional(),
  /** Repeat header rows on every page. */
  repeatHeader: z.boolean().default(true),
  widthPercent: z.number().min(0).max(100).optional(),
});

export type PdfListItem = {
  runs:      PdfTextRun[];
  children?: PdfListItem[];
};

export const PdfListItemSchema: z.ZodType<PdfListItem> = z.lazy(() =>
  z.object({
    runs:     z.array(PdfTextRunSchema).min(1),
    children: z.array(PdfListItemSchema).optional(),
  }),
);

export const PdfListSchema = z.object({
  ...blockBase,
  type:           z.literal('list'),
  ordered:        z.boolean().default(false),
  start:          z.number().int().positive().default(1),
  items:          z.array(PdfListItemSchema).min(1),
  typography:     PdfTypographySchema.optional(),
  indentPerLevel: PdfMmSchema.optional(),
  listStyleType:  z.string().optional(),
});

export const PdfCodeSchema = z.object({
  ...blockBase,
  type:        z.literal('code'),
  content:     z.string(),
  language:    z.string().optional(),
  lineNumbers: z.boolean().default(false),
  typography:  PdfTypographySchema.optional(),
  background:  PdfColorSchema.optional(),
  border:      PdfBorderSchema.optional(),
});

export const PdfBlockquoteSchema = z.object({
  ...blockBase,
  type:        z.literal('blockquote'),
  runs:        z.array(PdfTextRunSchema).min(1),
  attribution: z.array(PdfTextRunSchema).optional(),
  typography:  PdfTypographySchema.optional(),
  accent:      PdfColorSchema.optional(),
  background:  PdfColorSchema.optional(),
});

export const PdfDividerSchema = z.object({
  ...blockBase,
  type:         z.literal('divider'),
  border:       PdfBorderSchema.optional(),
  widthPercent: z.number().min(0).max(100).default(100),
});

export const PdfSpacerSchema = z.object({
  type:   z.literal('spacer'),
  /** Height in mm. */
  height: PdfMmSchema,
});

export const PdfPageBreakSchema = z.object({
  type: z.literal('pageBreak'),
});

// ─────────────────────────────────────────────────────────────────────────────
// PdfBlock — the single union used in document.content
// ─────────────────────────────────────────────────────────────────────────────

export const PdfBlockSchema = z.discriminatedUnion('type', [
  PdfHeadingSchema,
  PdfParagraphSchema,
  PdfImageSchema,
  PdfTableSchema,
  PdfListSchema,
  PdfCodeSchema,
  PdfBlockquoteSchema,
  PdfDividerSchema,
  PdfSpacerSchema,
  PdfPageBreakSchema,
]);

export type PdfBlock = z.infer<typeof PdfBlockSchema>;

// ─────────────────────────────────────────────────────────────────────────────
// Page settings
// ─────────────────────────────────────────────────────────────────────────────

export const PdfPageSettingsSchema = z.object({
  size:         z.enum(['A4', 'A3', 'A5', 'Letter', 'Legal', 'Tabloid', 'custom']).default('A4'),
  orientation:  z.enum(['portrait', 'landscape']).default('portrait'),
  customWidth:  PdfMmSchema.optional(),
  customHeight: PdfMmSchema.optional(),
  margins:      PdfSpacingSchema.default({ top: 20, right: 20, bottom: 20, left: 20 }),
  pageNumbers: z.object({
    show:     z.boolean().default(false),
    position: z.enum(['bottom-center', 'bottom-right', 'bottom-left', 'top-center', 'top-right', 'top-left']).default('bottom-center'),
    format:   z.enum(['numeric', 'roman', 'alpha']).default('numeric'),
    startAt:  z.number().int().positive().default(1),
  }).optional(),
  header: z.array(PdfTextRunSchema).optional(),
  footer: z.array(PdfTextRunSchema).optional(),
});

// ─────────────────────────────────────────────────────────────────────────────
// Document
// ─────────────────────────────────────────────────────────────────────────────

export const PdfDocumentMetaSchema = z.object({
  title:       z.string().default('Untitled Document'),
  author:      z.string().optional(),
  subject:     z.string().optional(),
  description: z.string().optional(),
  keywords:    z.array(z.string()).optional(),
  language:    z.string().default('en'),
  createdAt:   z.string().datetime().optional(),
  updatedAt:   z.string().datetime().optional(),
});

/**
 * Root schema for a `.bkwpdf.json` file.
 *
 * PdfDocument
 * ├── meta           — title, author, …
 * ├── pageSettings   — size, orientation, margins, header/footer
 * ├── defaultStyles  — typography defaults for the whole document
 * ├── namedStyles    — reusable typography presets
 * ├── background?    — colour or image applied to every page
 * └── content[]      — flat list of PdfBlock, rendered top-to-bottom
 *
 * The renderer paginates `content` automatically.  Use `breakInside: 'avoid'`
 * on a block to keep it whole on a single page.  Insert `{ type: 'pageBreak' }`
 * for an explicit page break.
 */
export const PdfDocumentSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  meta:          PdfDocumentMetaSchema,
  pageSettings:  PdfPageSettingsSchema,
  defaultStyles: PdfTypographySchema.optional(),
  namedStyles:   z.record(z.string(), PdfTypographySchema).optional(),
  background: z.union([
    PdfColorSchema,
    z.object({ src: z.string(), opacity: z.number().min(0).max(1).optional() }),
  ]).optional(),
  /**
   * Flat list of content blocks, flowed top-to-bottom and paginated
   * automatically.  Images, tables, code, and blockquotes are never cut in the
   * middle by default (`breakInside: 'avoid'` applied by the renderer).
   */
  content: z.array(PdfBlockSchema).min(1),
});

export type PdfDocument     = z.infer<typeof PdfDocumentSchema>;
export type PdfDocumentMeta = z.infer<typeof PdfDocumentMetaSchema>;
export type PdfPageSettings = z.infer<typeof PdfPageSettingsSchema>;

// ─────────────────────────────────────────────────────────────────────────────
// JSON Schema export
// ─────────────────────────────────────────────────────────────────────────────

/** Zod schema for a minified document — a non-empty array of PdfBlock. */
export const PdfBlockArraySchema = z.array(PdfBlockSchema).min(1);

export function getMinifiedJsonSchema(): Record<string, unknown> {
  return z.toJSONSchema(PdfBlockArraySchema) as Record<string, unknown>;
}



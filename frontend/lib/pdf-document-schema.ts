/**
 * PDF Document Schema
 *
 * Defines the full structure of a PDF document that can be saved to / loaded
 * from a local JSON file.  The design mirrors the HTML / CSS split:
 *
 *   • Content nodes  (heading, paragraph, image, table, list, …)
 *     → what is on the page  (like HTML elements)
 *
 *   • Layout containers  (sections with block / flex / grid / absolute layout)
 *     → how content is positioned and arranged  (like CSS)
 *
 *   • Document-level settings  (page size, margins, default typography)
 *     → like a <head> stylesheet applied to the whole document
 *
 * All measurements are in millimetres (mm) unless otherwise noted, so they
 * map directly to physical print dimensions.
 */

import { z } from 'zod';

// ─────────────────────────────────────────────────────────────────────────────
// Primitive / shared primitives
// ─────────────────────────────────────────────────────────────────────────────

/** Any CSS-compatible colour string: hex, rgb(), rgba(), hsl(), named colour. */
export const PdfColorSchema = z.string().describe('CSS-compatible color string');

/** Physical measurement in mm. */
export const PdfMmSchema = z.number().describe('Measurement in mm');

/** TRBL (top / right / bottom / left) box model in mm. */
export const PdfSpacingSchema = z.object({
  top:    PdfMmSchema.default(0),
  right:  PdfMmSchema.default(0),
  bottom: PdfMmSchema.default(0),
  left:   PdfMmSchema.default(0),
});

export const PdfBorderStyleSchema = z.enum([
  'none', 'solid', 'dashed', 'dotted', 'double',
]);

export const PdfBorderSchema = z.object({
  width: PdfMmSchema.default(0.3),
  style: PdfBorderStyleSchema.default('solid'),
  color: PdfColorSchema.default('#cccccc'),
});

/** Explicit per-side borders — any side can be omitted. */
export const PdfBorderSidesSchema = z.object({
  top:    PdfBorderSchema.optional(),
  right:  PdfBorderSchema.optional(),
  bottom: PdfBorderSchema.optional(),
  left:   PdfBorderSchema.optional(),
  all:    PdfBorderSchema.optional(), // shorthand; per-side overrides `all`
});

// ─────────────────────────────────────────────────────────────────────────────
// Typography
// ─────────────────────────────────────────────────────────────────────────────

export const PdfFontWeightSchema = z.union([
  z.number().int().min(100).max(900),
  z.enum(['normal', 'bold', 'lighter', 'bolder']),
]);

export const PdfFontStyleEnumSchema = z.enum(['normal', 'italic', 'oblique']);

export const PdfTextAlignSchema = z.enum(['left', 'center', 'right', 'justify']);

export const PdfTextDecorationSchema = z.enum([
  'none', 'underline', 'line-through', 'overline',
]);

export const PdfTextTransformSchema = z.enum([
  'none', 'uppercase', 'lowercase', 'capitalize',
]);

/**
 * All typographic properties that can be applied either globally (as a default
 * style) or locally on any content element.
 */
export const PdfTypographySchema = z.object({
  fontFamily:      z.string().optional(),
  /** Point size (pt). */
  fontSize:        z.number().positive().optional(),
  fontWeight:      PdfFontWeightSchema.optional(),
  fontStyle:       PdfFontStyleEnumSchema.optional(),
  color:           PdfColorSchema.optional(),
  textAlign:       PdfTextAlignSchema.optional(),
  textDecoration:  PdfTextDecorationSchema.optional(),
  textTransform:   PdfTextTransformSchema.optional(),
  /** Unitless multiplier (e.g. 1.5). */
  lineHeight:      z.number().positive().optional(),
  /** In pt. */
  letterSpacing:   z.number().optional(),
  /** In pt. */
  wordSpacing:     z.number().optional(),
});

// ─────────────────────────────────────────────────────────────────────────────
// Inline text runs  (rich text without raw HTML)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A single run of text that shares the same inline formatting.
 * Multiple runs compose a paragraph, heading, list item, etc.
 *
 * Inspired by the OOXML run model: each run carries its own overrides on top
 * of the parent element's typography.
 */
export const PdfTextRunSchema = z.object({
  /** The literal text content. Use '\n' only for forced line-breaks. */
  text:            z.string(),
  bold:            z.boolean().optional(),
  italic:          z.boolean().optional(),
  underline:       z.boolean().optional(),
  strikethrough:   z.boolean().optional(),
  /** Overrides the parent element's color. */
  color:           PdfColorSchema.optional(),
  /** Override font size (pt). */
  fontSize:        z.number().positive().optional(),
  /** Override font family. */
  fontFamily:      z.string().optional(),
  /** Superscript / subscript. */
  verticalAlign:   z.enum(['baseline', 'super', 'sub']).optional(),
  /** If set, the run is rendered as a hyperlink. */
  link:            z.string().url().optional(),
  /** Inline background highlight. */
  highlight:       PdfColorSchema.optional(),
});

export type PdfTextRun = z.infer<typeof PdfTextRunSchema>;

// ─────────────────────────────────────────────────────────────────────────────
// Content elements
// ─────────────────────────────────────────────────────────────────────────────

/** Heading levels 1–6 (matching HTML). */
export const PdfHeadingLevelSchema = z.union([
  z.literal(1), z.literal(2), z.literal(3),
  z.literal(4), z.literal(5), z.literal(6),
]);

const baseElementFields = {
  /** Optional stable identifier for cross-referencing. */
  id:       z.string().optional(),
  /** Spacing *outside* the element (mm). */
  margin:   PdfSpacingSchema.optional(),
  /** Spacing *inside* the element (mm). */
  padding:  PdfSpacingSchema.optional(),
};

// ── Heading ──────────────────────────────────────────────────────────────────
export const PdfHeadingElementSchema = z.object({
  ...baseElementFields,
  type:       z.literal('heading'),
  level:      PdfHeadingLevelSchema,
  runs:       z.array(PdfTextRunSchema).min(1),
  typography: PdfTypographySchema.optional(),
  /** Numbered outline position, e.g. "2.1.3" – filled automatically if omitted. */
  outlineLabel: z.string().optional(),
  /** Add to the document outline / TOC. */
  includeInToc: z.boolean().default(true),
});

// ── Paragraph ────────────────────────────────────────────────────────────────
export const PdfParagraphElementSchema = z.object({
  ...baseElementFields,
  type:       z.literal('paragraph'),
  runs:       z.array(PdfTextRunSchema),
  typography: PdfTypographySchema.optional(),
  /** First-line indent in mm. */
  textIndent: PdfMmSchema.optional(),
  /** Minimum lines to keep together (orphan / widow control). */
  keepLines:  z.number().int().positive().optional(),
});

// ── Image ─────────────────────────────────────────────────────────────────────
export const PdfObjectFitSchema = z.enum([
  'fill', 'contain', 'cover', 'none', 'scale-down',
]);

export const PdfImageElementSchema = z.object({
  ...baseElementFields,
  type:       z.literal('image'),
  /** Relative file path, absolute URL, or a base-64 data URI. */
  src:        z.string(),
  alt:        z.string().optional(),
  /** Explicit width in mm; omit to derive from container. */
  width:      PdfMmSchema.optional(),
  /** Explicit height in mm; omit to preserve aspect ratio. */
  height:     PdfMmSchema.optional(),
  objectFit:  PdfObjectFitSchema.optional(),
  /** Horizontal alignment when narrower than its container. */
  align:      z.enum(['left', 'center', 'right']).optional(),
  caption:    z.array(PdfTextRunSchema).optional(),
  border:     PdfBorderSchema.optional(),
  /** Border radius in mm. */
  borderRadius: PdfMmSchema.optional(),
  opacity:    z.number().min(0).max(1).optional(),
});

// ── Table ─────────────────────────────────────────────────────────────────────
export const PdfTableCellSchema = z.object({
  runs:       z.array(PdfTextRunSchema),
  typography: PdfTypographySchema.optional(),
  /** Override individual cell padding. */
  padding:    PdfSpacingSchema.optional(),
  background: PdfColorSchema.optional(),
  border:     PdfBorderSidesSchema.optional(),
  /** How many columns this cell spans. */
  colSpan:    z.number().int().positive().default(1),
  /** How many rows this cell spans. */
  rowSpan:    z.number().int().positive().default(1),
  /** Vertical alignment within the cell. */
  verticalAlign: z.enum(['top', 'middle', 'bottom']).optional(),
});

export const PdfTableRowSchema = z.object({
  cells:      z.array(PdfTableCellSchema),
  /** Optional: shade the whole row with a background colour. */
  background: PdfColorSchema.optional(),
  /** Fixed row height in mm; omit for auto. */
  height:     PdfMmSchema.optional(),
  isHeader:   z.boolean().default(false),
});

export const PdfTableElementSchema = z.object({
  ...baseElementFields,
  type:       z.literal('table'),
  rows:       z.array(PdfTableRowSchema).min(1),
  /** Explicit column widths (mm).  Must sum ≤ container width.  Omit for equal distribution. */
  columnWidths: z.array(PdfMmSchema).optional(),
  /** Default cell padding applied to every cell (mm). */
  cellPadding: PdfSpacingSchema.optional(),
  border:     PdfBorderSchema.optional(),
  caption:    z.array(PdfTextRunSchema).optional(),
  /** Repeat header rows on every page break. */
  repeatHeader: z.boolean().default(true),
  /** Width of table as % of container (0–100). */
  widthPercent: z.number().min(0).max(100).optional(),
});

// ── List ──────────────────────────────────────────────────────────────────────
export type PdfListItem = {
  runs:     PdfTextRun[];
  children?: PdfListItem[];
};

export const PdfListItemSchema: z.ZodType<PdfListItem> = z.lazy(() =>
  z.object({
    runs:     z.array(PdfTextRunSchema).min(1),
    /** Nested sub-list. */
    children: z.array(PdfListItemSchema).optional(),
  }),
);

export const PdfListElementSchema = z.object({
  ...baseElementFields,
  type:          z.literal('list'),
  ordered:       z.boolean().default(false),
  /** Custom start number for ordered lists. */
  start:         z.number().int().positive().default(1),
  items:         z.array(PdfListItemSchema).min(1),
  typography:    PdfTypographySchema.optional(),
  /** Indent per nesting level (mm). */
  indentPerLevel: PdfMmSchema.optional(),
  /** Custom list marker: 'disc' | 'circle' | 'square' | 'decimal' | 'alpha' | … */
  listStyleType: z.string().optional(),
});

// ── Code block ────────────────────────────────────────────────────────────────
export const PdfCodeBlockElementSchema = z.object({
  ...baseElementFields,
  type:       z.literal('code'),
  content:    z.string(),
  language:   z.string().optional(),
  /** Show line numbers. */
  lineNumbers: z.boolean().default(false),
  typography: PdfTypographySchema.optional(),
  background: PdfColorSchema.optional(),
  border:     PdfBorderSchema.optional(),
});

// ── Blockquote ────────────────────────────────────────────────────────────────
export const PdfBlockquoteElementSchema = z.object({
  ...baseElementFields,
  type:         z.literal('blockquote'),
  runs:         z.array(PdfTextRunSchema).min(1),
  attribution:  z.array(PdfTextRunSchema).optional(),
  typography:   PdfTypographySchema.optional(),
  accent:       PdfColorSchema.optional(),
  background:   PdfColorSchema.optional(),
});

// ── Horizontal divider ────────────────────────────────────────────────────────
export const PdfDividerElementSchema = z.object({
  ...baseElementFields,
  type:    z.literal('divider'),
  border:  PdfBorderSchema.optional(),
  /** Width as % of the container. */
  widthPercent: z.number().min(0).max(100).default(100),
});

// ── Vertical spacer ───────────────────────────────────────────────────────────
export const PdfSpacerElementSchema = z.object({
  type:   z.literal('spacer'),
  /** Height in mm. */
  height: PdfMmSchema,
});

// ── Page Break ────────────────────────────────────────────────────────────────
export const PdfPageBreakElementSchema = z.object({
  type: z.literal('pageBreak'),
});

// ── Union of all element types ────────────────────────────────────────────────
export const PdfContentElementSchema = z.discriminatedUnion('type', [
  PdfHeadingElementSchema,
  PdfParagraphElementSchema,
  PdfImageElementSchema,
  PdfTableElementSchema,
  PdfListElementSchema,
  PdfCodeBlockElementSchema,
  PdfBlockquoteElementSchema,
  PdfDividerElementSchema,
  PdfSpacerElementSchema,
  PdfPageBreakElementSchema,
]);

export type PdfContentElement = z.infer<typeof PdfContentElementSchema>;

// ─────────────────────────────────────────────────────────────────────────────
// Layout containers  (like CSS block / flex / grid wrappers)
// ─────────────────────────────────────────────────────────────────────────────

export const PdfLayoutTypeSchema = z.enum([
  'block',     // flow layout — children stack top-to-bottom
  'flex',      // one-dimensional flex layout
  'grid',      // two-dimensional grid layout
  'columns',   // newspaper-style multi-column (like CSS column-count)
  'absolute',  // absolute position within the page canvas (mm from top-left)
]);

/** Options that apply only when layout === 'flex'. */
export const PdfFlexOptionsSchema = z.object({
  direction:      z.enum(['row', 'row-reverse', 'column', 'column-reverse']).default('row'),
  wrap:           z.enum(['nowrap', 'wrap', 'wrap-reverse']).default('wrap'),
  /** Gap between items (mm). */
  gap:            PdfMmSchema.default(4),
  justifyContent: z.enum(['flex-start', 'flex-end', 'center', 'space-between', 'space-around', 'space-evenly']).optional(),
  alignItems:     z.enum(['flex-start', 'flex-end', 'center', 'stretch', 'baseline']).optional(),
  alignContent:   z.enum(['flex-start', 'flex-end', 'center', 'stretch', 'space-between', 'space-around']).optional(),
});

/** Options that apply only when layout === 'grid'. */
export const PdfGridOptionsSchema = z.object({
  /**
   * Number of equal-width columns.
   * For custom widths provide `templateColumns` instead.
   */
  columns:          z.number().int().positive().optional(),
  /**
   * CSS-style column template, e.g. "1fr 2fr 1fr" or "60mm 80mm".
   * Overrides `columns`.
   */
  templateColumns:  z.string().optional(),
  templateRows:     z.string().optional(),
  /** Gap between grid cells (mm). */
  columnGap:        PdfMmSchema.default(4),
  rowGap:           PdfMmSchema.default(4),
  alignItems:       z.enum(['start', 'end', 'center', 'stretch']).optional(),
  justifyItems:     z.enum(['start', 'end', 'center', 'stretch']).optional(),
});

/** Options that apply only when layout === 'columns'. */
export const PdfColumnsOptionsSchema = z.object({
  count:      z.number().int().min(2).max(12).default(2),
  /** Gap between columns (mm). */
  gap:        PdfMmSchema.default(5),
  /** Vertical rule between columns. */
  rule:       PdfBorderSchema.optional(),
});

/** Options that apply only when layout === 'absolute'. */
export const PdfAbsolutePositionSchema = z.object({
  /** Distance from the left edge of the page canvas (mm). */
  x:      PdfMmSchema,
  /** Distance from the top edge of the page canvas (mm). */
  y:      PdfMmSchema,
  width:  PdfMmSchema,
  height: PdfMmSchema,
  /** Z-order for stacking overlapping sections. */
  zIndex: z.number().int().default(0),
});

/**
 * A Section is the layout primitive — the equivalent of a `<div>` in HTML.
 * Sections can be nested to build complex responsive grids or side-by-side
 * columns.
 */
export type PdfSection = {
  id?:          string;
  layout:       z.infer<typeof PdfLayoutTypeSchema>;
  flex?:        z.infer<typeof PdfFlexOptionsSchema>;
  grid?:        z.infer<typeof PdfGridOptionsSchema>;
  multiColumn?: z.infer<typeof PdfColumnsOptionsSchema>;
  absolute?:    z.infer<typeof PdfAbsolutePositionSchema>;
  /** Flex item: how much this section grows (only relevant when parent is flex). */
  flexGrow?:    number;
  /** Fixed width in mm; omit to fill the available container width. */
  width?:       number;
  /** Fixed height in mm; omit to be driven by content. */
  height?:      number;
  padding?:     z.infer<typeof PdfSpacingSchema>;
  margin?:      z.infer<typeof PdfSpacingSchema>;
  background?:  string;
  border?:      z.infer<typeof PdfBorderSidesSchema>;
  /** Border radius in mm. */
  borderRadius?: number;
  /** Keep this section on the same page as the next section (no page break between them). */
  keepWithNext?: boolean;
  /** Children can be either content elements or nested sections. */
  children:     Array<PdfContentElement | PdfSection>;
};

// Zod schema for PdfSection (recursive — uses z.lazy for self-reference)
export const PdfSectionSchema: z.ZodType<PdfSection> = z.lazy(() =>
  z.object({
    id:           z.string().optional(),
    layout:       PdfLayoutTypeSchema,
    flex:         PdfFlexOptionsSchema.optional(),
    grid:         PdfGridOptionsSchema.optional(),
    multiColumn:  PdfColumnsOptionsSchema.optional(),
    absolute:     PdfAbsolutePositionSchema.optional(),
    flexGrow:     z.number().optional(),
    width:        PdfMmSchema.optional(),
    height:       PdfMmSchema.optional(),
    padding:      PdfSpacingSchema.optional(),
    margin:       PdfSpacingSchema.optional(),
    background:   PdfColorSchema.optional(),
    border:       PdfBorderSidesSchema.optional(),
    borderRadius: PdfMmSchema.optional(),
    keepWithNext: z.boolean().optional(),
    children:     z.array(z.union([PdfContentElementSchema, PdfSectionSchema])),
  }),
);

// ─────────────────────────────────────────────────────────────────────────────
// Page settings
// ─────────────────────────────────────────────────────────────────────────────

export const PdfPageSizeSchema = z.enum([
  'A4', 'A3', 'A5',
  'Letter', 'Legal', 'Tabloid',
  'custom',
]);

export const PdfOrientationSchema = z.enum(['portrait', 'landscape']);

export const PdfPageSettingsSchema = z.object({
  size:        PdfPageSizeSchema.default('A4'),
  orientation: PdfOrientationSchema.default('portrait'),
  /** Required when size === 'custom'. */
  customWidth:  PdfMmSchema.optional(),
  /** Required when size === 'custom'. */
  customHeight: PdfMmSchema.optional(),
  margins:     PdfSpacingSchema.default({ top: 20, right: 20, bottom: 20, left: 20 }),
  /** Add page numbers. */
  pageNumbers: z.object({
    show:     z.boolean().default(false),
    position: z.enum(['bottom-center', 'bottom-right', 'bottom-left', 'top-center', 'top-right', 'top-left']).default('bottom-center'),
    format:   z.enum(['numeric', 'roman', 'alpha']).default('numeric'),
    /** Starting number. */
    startAt:  z.number().int().positive().default(1),
  }).optional(),
  header:      z.array(PdfTextRunSchema).optional(),
  footer:      z.array(PdfTextRunSchema).optional(),
});

// ─────────────────────────────────────────────────────────────────────────────
// Page
// ─────────────────────────────────────────────────────────────────────────────

export const PdfPageSchema = z.object({
  /** Unique identifier within the document. */
  id:         z.string(),
  /** Per-page overrides of the document-level page settings. */
  settings:   PdfPageSettingsSchema.partial().optional(),
  /** Page-level background colour or image URL. */
  background: z.union([
    PdfColorSchema,
    z.object({ src: z.string(), opacity: z.number().min(0).max(1).optional() }),
  ]).optional(),
  /** The root section(s) that fill this page. */
  sections:   z.array(PdfSectionSchema),
});

export type PdfPage = z.infer<typeof PdfPageSchema>;

// ─────────────────────────────────────────────────────────────────────────────
// Document metadata
// ─────────────────────────────────────────────────────────────────────────────

export const PdfDocumentMetaSchema = z.object({
  title:       z.string().default('Untitled Document'),
  author:      z.string().optional(),
  subject:     z.string().optional(),
  description: z.string().optional(),
  keywords:    z.array(z.string()).optional(),
  language:    z.string().default('en'),
  /** ISO-8601 date string. */
  createdAt:   z.string().datetime().optional(),
  /** ISO-8601 date string. */
  updatedAt:   z.string().datetime().optional(),
});

// ─────────────────────────────────────────────────────────────────────────────
// Top-level Document
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The root schema for a `.bkwpdf.json` file.
 *
 * Structure summary
 * ─────────────────
 * PdfDocument
 * ├── meta            — title, author, keywords, …
 * ├── pageSettings    — page size, orientation, margins (document defaults)
 * ├── defaultStyles   — typography applied to the whole document
 * ├── namedStyles     — reusable named style presets (like CSS classes)
 * └── pages[]
 *     ├── settings?   — per-page overrides
 *     └── sections[]  — recursive layout tree
 *         └── children[]
 *             ├── PdfSection   (nested layout container)
 *             └── PdfContentElement
 *                 ├── heading | paragraph | image | table
 *                 ├── list | code | blockquote | divider
 *                 └── spacer | pageBreak
 */
export const PdfDocumentSchema = z.object({
  /** Schema version — bump when the shape changes in a breaking way. */
  schemaVersion: z.literal('1.0').default('1.0'),
  /** Stable document UUID. */
  id:            z.string().uuid(),
  meta:          PdfDocumentMetaSchema,
  /** Document-wide page settings (can be overridden per page). */
  pageSettings:  PdfPageSettingsSchema,
  /** Document-wide default typography (can be overridden per element). */
  defaultStyles: PdfTypographySchema.optional(),
  /**
   * Named style presets — referenced by name from element `styleRef` fields.
   * Think of these as CSS classes.
   *
   * Example:
   * ```json
   * { "callout": { "fontSize": 11, "color": "#7c3aed", "fontWeight": "bold" } }
   * ```
   */
  namedStyles:   z.record(z.string(), PdfTypographySchema).optional(),
  pages:         z.array(PdfPageSchema).min(1),
});

export type PdfDocument = z.infer<typeof PdfDocumentSchema>;
export type PdfDocumentMeta = z.infer<typeof PdfDocumentMetaSchema>;
export type PdfPageSettings = z.infer<typeof PdfPageSettingsSchema>;

// ─────────────────────────────────────────────────────────────────────────────
// JSON Schema export
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Returns the formal JSON Schema representation of `PdfDocumentSchema`.
 * Useful for editor tooling, external validation, or documentation.
 *
 * Uses Zod v4's built-in `z.toJSONSchema()` — no extra dependency required.
 */
export function getPdfDocumentJsonSchema(): Record<string, unknown> {
  return z.toJSONSchema(PdfDocumentSchema) as Record<string, unknown>;
}



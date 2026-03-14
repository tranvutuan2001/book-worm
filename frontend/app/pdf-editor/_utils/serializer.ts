// ─────────────────────────────────────────────────────────────────────────────
// Serialisation helpers
// ─────────────────────────────────────────────────────────────────────────────

import { PdfDocument, PdfDocumentSchema, PdfPage } from "@/lib/schemas";
import { PdfPageComponentSchema, PdfPageComponent } from "@/lib/pdf-document-schema";
import z from "zod";

/**
 * Serialises a `PdfDocument` to a pretty-printed JSON string suitable for
 * writing to a `.bkwpdf.json` file.
 */
export function serializePdfDocument(doc: PdfDocument): string {
  return JSON.stringify(doc, null, 2);
}

/**
 * Like `parsePdfDocument` but returns `null` instead of throwing.
 * The second element of the tuple is the Zod error (if any).
 */
export function safeParsePdfDocument(
  input: string | unknown,
): [PdfDocument, null] | [null, z.ZodError] {
  try {
    const raw = typeof input === 'string' ? JSON.parse(input) : input;
    const result = PdfDocumentSchema.safeParse(raw);
    if (result.success) return [result.data, null];
    return [null, result.error];
  } catch {
    return [null, new z.ZodError([{
      code: 'custom',
      path: [],
      message: 'Input is not valid JSON',
    }])];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory helpers — create blank document / page / section
// ─────────────────────────────────────────────────────────────────────────────

/** Creates a minimal valid blank document. */
export function createBlankDocument(title = 'Untitled Document'): PdfDocument {
  return PdfDocumentSchema.parse({
    schemaVersion: '1.0',
    id: crypto.randomUUID(),
    meta: { title, createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() },
    pageSettings: {
      size: 'A4',
      orientation: 'portrait',
      margins: { top: 20, right: 20, bottom: 20, left: 20 },
    },
    pages: [createBlankPage()],
  });
}

/** Creates a minimal valid blank page. */
const createBlankPage = (): PdfPage => {
  return {
    components: [
      {
        layout: 'block',
        children: [
          {
            type: 'paragraph',
            runs: [{ text: '' }],
          },
        ],
      },
    ],
  };
}

/**
 * Parses an uploaded minified JSON (an array of `PdfPageComponent`) and wraps
 * it in a full `PdfDocument` with all missing attributes filled in with their
 * schema default values.
 *
 * Returns `[doc, null]` on success, or `[null, error]` on failure.
 */
export function safeParseMinifiedComponents(
  input: string | unknown,
): [PdfDocument, null] | [null, z.ZodError] {
  try {
    const raw = typeof input === 'string' ? JSON.parse(input) : input;
    const componentsResult = z.array(PdfPageComponentSchema).safeParse(raw);
    if (!componentsResult.success) return [null, componentsResult.error];

    const components: PdfPageComponent[] = componentsResult.data;

    // Wrap the components in a full document, letting PdfDocumentSchema.parse
    // fill every field that has a Zod default value.
    const docResult = PdfDocumentSchema.safeParse({
      schemaVersion: '1.0',
      meta: {
        title: 'Untitled Document',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      },
      pageSettings: {
        size: 'A4',
        orientation: 'portrait',
        margins: { top: 20, right: 20, bottom: 20, left: 20 },
      },
      pages: [{ components }],
    });

    if (!docResult.success) return [null, docResult.error];
    return [docResult.data, null];
  } catch {
    return [null, new z.ZodError([{
      code: 'custom',
      path: [],
      message: 'Input is not valid JSON',
    }])];
  }
}
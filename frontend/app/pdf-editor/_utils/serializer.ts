// ─────────────────────────────────────────────────────────────────────────────
// Serialisation helpers
// ─────────────────────────────────────────────────────────────────────────────

import { PdfDocument, PdfDocumentSchema, PdfBlock, PdfBlockArraySchema } from "@/lib/pdf-document-schema";
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
    meta: { title, createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() },
    pageSettings: {
      size: 'A4',
      orientation: 'portrait',
      margins: { top: 20, right: 20, bottom: 20, left: 20 },
    },
    content: [
      { type: 'paragraph', runs: [{ text: '' }] },
    ],
  });
}

/**
 * Parses an uploaded JSON (an array of `PdfBlock`) and wraps it in a full
 * `PdfDocument` with all missing attributes filled in with their schema defaults.
 *
 * Returns `[doc, null]` on success, or `[null, error]` on failure.
 */
export function safeParseMinifiedComponents(
  input: string | unknown,
): [PdfDocument, null] | [null, z.ZodError] {
  try {
    const raw = typeof input === 'string' ? JSON.parse(input) : input;
    const blocksResult = PdfBlockArraySchema.safeParse(raw);
    if (!blocksResult.success) return [null, blocksResult.error];

    const blocks: PdfBlock[] = blocksResult.data;

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
      content: blocks,
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
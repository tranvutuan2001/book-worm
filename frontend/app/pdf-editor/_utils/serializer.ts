// ─────────────────────────────────────────────────────────────────────────────
// Serialisation helpers
// ─────────────────────────────────────────────────────────────────────────────

import { PdfDocument, PdfDocumentSchema, PdfPage } from "@/lib/schemas";
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
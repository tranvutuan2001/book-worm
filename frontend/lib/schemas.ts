import { z } from 'zod';

export const MessageSchema = z.object({
  id: z.string(),
  role: z.enum(['user', 'assistant', 'system']),
  content: z.string(),
  timestamp: z.number(),
});

export const ConversationSchema = z.object({
  id: z.string(),
  message_list: z.array(MessageSchema),
  timestamp: z.number(),
  document_name: z.string().nullable().optional(),
});

// Ask endpoint response
export const AskResponseSchema = z.object({
  message: z.string(),
  conversation_id: z.string(),
  timestamp: z.number(),
});

// Document status enum
export const DocumentStatusSchema = z.enum(['ready', 'processing', 'analyzing', 'error']);

// Document upload response
export const UploadResponseSchema = z.object({
  message: z.string(),
  document_name: z.string(),
  status: DocumentStatusSchema,
});

// Document info with status
export const DocumentInfoSchema = z.object({
  name: z.string(),
  status: DocumentStatusSchema,
  path: z.string(),
});

// Documents list response
export const DocumentsResponseSchema = z.object({
  documents: z.array(DocumentInfoSchema),
});

export type Message = z.infer<typeof MessageSchema>;
export type Conversation = z.infer<typeof ConversationSchema>;
export type AskResponse = z.infer<typeof AskResponseSchema>;
export type DocumentStatus = z.infer<typeof DocumentStatusSchema>;
export type UploadResponse = z.infer<typeof UploadResponseSchema>;
export type DocumentInfo = z.infer<typeof DocumentInfoSchema>;
export type DocumentsResponse = z.infer<typeof DocumentsResponseSchema>;

// ── PDF Document schema ───────────────────────────────────────────────────────
export * from './pdf-document-schema';

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
  chat_model: z.string(),
  embedding_model: z.string(),
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

// LLM Backend Schemas
export const ModelInfoSchema = z.object({
  name: z.string(),
  path: z.string(),
  size: z.string(),
  status: z.enum(['ready_to_use', 'downloading']),
});

export const DownloadableModelInfoSchema = z.object({
  name: z.string(),
  repository: z.string(),
  filename: z.string(),
});

export const LoadedModelInfoSchema = z.object({
  model_name: z.string(),
  model_path: z.string(),
  model_type: z.string(),
  loaded: z.boolean(),
});

export const ModelLoadRequestSchema = z.object({
  model: z.string(),
  model_type: z.enum(['chat', 'embedding']),
});

export const ModelUnloadRequestSchema = z.object({
  model: z.string(),
  model_type: z.enum(['chat', 'embedding']),
});

export const ModelDownloadRequestSchema = z.object({
  repository: z.string(),
});

export type ModelInfo = z.infer<typeof ModelInfoSchema>;
export type DownloadableModelInfo = z.infer<typeof DownloadableModelInfoSchema>;
export type LoadedModelInfo = z.infer<typeof LoadedModelInfoSchema>;
export type ModelLoadRequest = z.infer<typeof ModelLoadRequestSchema>;
export type ModelUnloadRequest = z.infer<typeof ModelUnloadRequestSchema>;
export type ModelDownloadRequest = z.infer<typeof ModelDownloadRequestSchema>;

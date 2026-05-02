import { 
  Conversation, 
  AskResponse, 
  AskResponseSchema, 
  DocumentsResponse, 
  DocumentsResponseSchema, 
  UploadResponse, 
  UploadResponseSchema,
} from '@/lib/schemas';
import { z } from 'zod';

// The frontend uses Next.js rewrites (see next.config.ts) to proxy requests:
//   /api/*  -> backend service
//   /llm/*  -> llm-server service
// This keeps the browser code using relative paths only, so no URL is baked
// into the browser bundle at build time.
const API_BASE_URL = 'http://localhost:8000';

export async function sendMessage(request: Conversation): Promise<AskResponse> {
  const response = await fetch(`${API_BASE_URL}/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`Failed to send message: ${response.statusText}`);
  }

  const data = await response.json();
  return AskResponseSchema.parse(data);
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Failed to upload document: ${response.statusText}`);
  }

  const data = await response.json();
  return UploadResponseSchema.parse(data);
}

export async function listDocuments(): Promise<DocumentsResponse> {
  const response = await fetch(`${API_BASE_URL}/documents`, {
    method: 'GET', 
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to list documents: ${response.statusText}`);
  }

  const data = await response.json();
  return DocumentsResponseSchema.parse(data);
}

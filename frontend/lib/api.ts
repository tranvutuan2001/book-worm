import { 
  Conversation, 
  AskResponse, 
  AskResponseSchema, 
  DocumentsResponse, 
  DocumentsResponseSchema, 
  UploadResponse, 
  UploadResponseSchema,
  ModelInfo,
  ModelInfoSchema,
  DownloadableModelInfo,
  DownloadableModelInfoSchema,
  LoadedModelInfo,
  LoadedModelInfoSchema,
  ModelLoadRequest,
  ModelUnloadRequest,
  ModelDownloadRequest,
} from '@/lib/schemas';
import { z } from 'zod';

const API_BASE_URL = 'http://localhost:8000';
const LLM_BASE_URL = 'http://localhost:8001';

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

// LLM Backend API Functions

export async function listChatModels(): Promise<ModelInfo[]> {
  const response = await fetch(`${LLM_BASE_URL}/v1/models/chat`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to list chat models: ${response.statusText}`);
  }

  const data = await response.json();
  return z.array(ModelInfoSchema).parse(data);
}

export async function listEmbeddingModels(): Promise<ModelInfo[]> {
  const response = await fetch(`${LLM_BASE_URL}/v1/models/embeddings`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to list embedding models: ${response.statusText}`);
  }

  const data = await response.json();
  return z.array(ModelInfoSchema).parse(data);
}

export async function listDownloadableChatModels(): Promise<DownloadableModelInfo[]> {
  const response = await fetch(`${LLM_BASE_URL}/v1/models/chat/downloadable`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to list downloadable chat models: ${response.statusText}`);
  }

  const data = await response.json();
  return z.array(DownloadableModelInfoSchema).parse(data);
}

export async function listDownloadableEmbeddingModels(): Promise<DownloadableModelInfo[]> {
  const response = await fetch(`${LLM_BASE_URL}/v1/models/embeddings/downloadable`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to list downloadable embedding models: ${response.statusText}`);
  }

  const data = await response.json();
  return z.array(DownloadableModelInfoSchema).parse(data);
}

export async function listLoadedModels(): Promise<LoadedModelInfo[]> {
  const response = await fetch(`${LLM_BASE_URL}/v1/models/loaded`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to list loaded models: ${response.statusText}`);
  }

  const data = await response.json();
  return z.array(LoadedModelInfoSchema).parse(data);
}

export async function downloadModel(request: ModelDownloadRequest): Promise<void> {
  const response = await fetch(`${LLM_BASE_URL}/v1/models/download`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to download model: ${response.statusText}`);
  }
}

export async function loadModel(request: ModelLoadRequest): Promise<void> {
  const response = await fetch(`${LLM_BASE_URL}/v1/models/load`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to load model: ${response.statusText}`);
  }
}

export async function unloadModel(request: ModelUnloadRequest): Promise<void> {
  const response = await fetch(`${LLM_BASE_URL}/v1/models/unload`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to unload model: ${response.statusText}`);
  }
}

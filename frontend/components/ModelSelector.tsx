'use client';

import { useState, useEffect } from 'react';
import { listLoadedModels } from '@/lib/api';
import type { LoadedModelInfo } from '@/lib/schemas';

interface ModelSelectorProps {
  selectedChatModel: string | null;
  selectedEmbeddingModel: string | null;
  onChatModelChange: (modelName: string) => void;
  onEmbeddingModelChange: (modelName: string) => void;
}

export default function ModelSelector({
  selectedChatModel,
  selectedEmbeddingModel,
  onChatModelChange,
  onEmbeddingModelChange,
}: ModelSelectorProps) {
  const [loadedChatModels, setLoadedChatModels] = useState<LoadedModelInfo[]>([]);
  const [loadedEmbeddingModels, setLoadedEmbeddingModels] = useState<LoadedModelInfo[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchLoadedModels = async () => {
    setLoading(true);
    try {
      const models = await listLoadedModels();
      const chatModels = models.filter((m) => m.model_type === 'chat');
      const embeddingModels = models.filter((m) => m.model_type === 'embedding');
      
      setLoadedChatModels(chatModels);
      setLoadedEmbeddingModels(embeddingModels);

      // Auto-select first model if none selected
      if (!selectedChatModel && chatModels.length > 0) {
        onChatModelChange(chatModels[0].model_name);
      }
      if (!selectedEmbeddingModel && embeddingModels.length > 0) {
        onEmbeddingModelChange(embeddingModels[0].model_name);
      }
    } catch (err) {
      console.error('Failed to fetch loaded models:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLoadedModels();
    // Refresh every 10 seconds
    const interval = setInterval(fetchLoadedModels, 10000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isReady = selectedChatModel && selectedEmbeddingModel;

  return (
    <div className="bg-white/90 backdrop-blur-sm rounded-xl shadow-lg border border-purple-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-base font-bold text-gray-800 flex items-center gap-2">
          <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Active Models
        </h3>
        <button
          onClick={fetchLoadedModels}
          disabled={loading}
          className="text-xs px-2 py-1 bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 disabled:bg-gray-100 font-medium"
        >
          {loading ? '↻' : '↻ Refresh'}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {/* Chat Model Selector */}
        <div>
          <label className="flex items-center gap-1 text-xs font-semibold text-gray-700 mb-1.5">
            <span className="w-2 h-2 rounded-full bg-purple-600"></span>
            Chat Model
          </label>
          {loadedChatModels.length === 0 ? (
            <div className="p-2 bg-yellow-50 border border-yellow-200 rounded-lg text-xs text-yellow-800">
              ⚠️ No chat models loaded
            </div>
          ) : (
            <select
              value={selectedChatModel || ''}
              onChange={(e) => onChatModelChange(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-purple-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 bg-white"
            >
              <option value="">Select chat model</option>
              {loadedChatModels.map((model) => (
                <option key={model.model_name} value={model.model_name}>
                  {model.model_name}
                </option>
              ))}
            </select>
          )}
        </div>

        {/* Embedding Model Selector */}
        <div>
          <label className="flex items-center gap-1 text-xs font-semibold text-gray-700 mb-1.5">
            <span className="w-2 h-2 rounded-full bg-blue-600"></span>
            Embedding Model
          </label>
          {loadedEmbeddingModels.length === 0 ? (
            <div className="p-2 bg-yellow-50 border border-yellow-200 rounded-lg text-xs text-yellow-800">
              ⚠️ No embedding models loaded
            </div>
          ) : (
            <select
              value={selectedEmbeddingModel || ''}
              onChange={(e) => onEmbeddingModelChange(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-blue-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
            >
              <option value="">Select embedding model</option>
              {loadedEmbeddingModels.map((model) => (
                <option key={model.model_name} value={model.model_name}>
                  {model.model_name}
                </option>
              ))}
            </select>
          )}
        </div>
      </div>

      {/* Status Indicator */}
      <div className="mt-3 pt-3 border-t border-purple-200">
        {isReady ? (
          <div className="flex items-center text-green-600 bg-green-50 rounded-lg px-3 py-2">
            <svg
              className="w-5 h-5 mr-2"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                clipRule="evenodd"
              />
            </svg>
            <span className="text-xs font-semibold">✓ System Ready</span>
          </div>
        ) : (
          <div className="flex items-center text-yellow-600 bg-yellow-50 rounded-lg px-3 py-2">
            <svg
              className="w-5 h-5 mr-2"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
            <span className="text-xs font-semibold">
              Please select both models
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

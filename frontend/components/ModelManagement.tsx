'use client';

import { useState, useEffect, useRef } from 'react';
import {
  listChatModels,
  listEmbeddingModels,
  listDownloadableChatModels,
  listDownloadableEmbeddingModels,
  listLoadedModels,
  downloadModel,
  loadModel,
  unloadModel,
} from '@/lib/api';
import type {
  ModelInfo,
  DownloadableModelInfo,
  LoadedModelInfo,
} from '@/lib/schemas';

export default function ModelManagement() {
  const [allChatModels, setAllChatModels] = useState<ModelInfo[]>([]);
  const [allEmbeddingModels, setAllEmbeddingModels] = useState<ModelInfo[]>([]);
  const [downloadableChatModels, setDownloadableChatModels] = useState<DownloadableModelInfo[]>([]);
  const [downloadableEmbeddingModels, setDownloadableEmbeddingModels] = useState<DownloadableModelInfo[]>([]);
  const [loadedModels, setLoadedModels] = useState<LoadedModelInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'available' | 'downloadable' | 'loaded'>('loaded');
  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Filter models by status
  const chatModels = allChatModels.filter((m) => m.status === 'ready_to_use');
  const embeddingModels = allEmbeddingModels.filter((m) => m.status === 'ready_to_use');
  const downloadingChatModels = allChatModels.filter((m) => m.status === 'downloading');
  const downloadingEmbeddingModels = allEmbeddingModels.filter((m) => m.status === 'downloading');
  const hasDownloadingModels = downloadingChatModels.length > 0 || downloadingEmbeddingModels.length > 0;

  const fetchAllModels = async () => {
    setLoading(true);
    setError(null);
    try {
      const [chat, embedding, dlChat, dlEmbedding, loaded] = await Promise.all([
        listChatModels(),
        listEmbeddingModels(),
        listDownloadableChatModels(),
        listDownloadableEmbeddingModels(),
        listLoadedModels(),
      ]);
      setAllChatModels(chat);
      setAllEmbeddingModels(embedding);
      setDownloadableChatModels(dlChat);
      setDownloadableEmbeddingModels(dlEmbedding);
      setLoadedModels(loaded);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch models');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAllModels();
  }, []);

  // Auto-refresh when models are downloading
  useEffect(() => {
    if (hasDownloadingModels) {
      refreshIntervalRef.current = setInterval(() => {
        fetchAllModels();
      }, 3000); // Refresh every 3 seconds
    } else {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
        refreshIntervalRef.current = null;
      }
    }

    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, [hasDownloadingModels]);

  const handleDownload = async (repository: string, modelName: string) => {
    // Check if model is already available or downloading
    const isAvailable = [...allChatModels, ...allEmbeddingModels].some(
      (m) => m.name === modelName && (m.status === 'ready_to_use' || m.status === 'downloading')
    );
    
    if (isAvailable) {
      setError('This model is already available or currently downloading');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      await downloadModel({ repository });
      await fetchAllModels();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to download model');
    } finally {
      setLoading(false);
    }
  };

  const handleLoad = async (modelName: string, modelType: 'chat' | 'embedding') => {
    try {
      setLoading(true);
      setError(null);
      await loadModel({ model: modelName, model_type: modelType });
      await fetchAllModels();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model');
    } finally {
      setLoading(false);
    }
  };

  const handleUnload = async (modelName: string, modelType: 'chat' | 'embedding') => {
    try {
      setLoading(true);
      setError(null);
      await unloadModel({ model: modelName, model_type: modelType });
      await fetchAllModels();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to unload model');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <button
          onClick={fetchAllModels}
          disabled={loading}
          className="w-full px-3 py-2 bg-linear-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:from-purple-700 hover:to-blue-700 disabled:from-gray-400 disabled:to-gray-500 font-medium text-sm shadow-lg"
        >
          {loading ? 'Refreshing...' : 'Refresh Models'}
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-lg text-sm border border-red-200">
          {error}
        </div>
      )}

      {/* Tabs */}
      <div className="flex border-b mb-4 gap-1">
        <button
          onClick={() => setActiveTab('loaded')}
          className={`flex-1 px-2 py-2 font-medium text-xs transition-colors ${
            activeTab === 'loaded'
              ? 'border-b-2 border-purple-600 text-purple-600'
              : 'text-gray-600 hover:text-gray-800'
          }`}
        >
          Loaded ({loadedModels.length})
        </button>
        <button
          onClick={() => setActiveTab('available')}
          className={`flex-1 px-2 py-2 font-medium text-xs transition-colors ${
            activeTab === 'available'
              ? 'border-b-2 border-purple-600 text-purple-600'
              : 'text-gray-600 hover:text-gray-800'
          }`}
        >
          Available ({chatModels.length + embeddingModels.length})
        </button>
        <button
          onClick={() => setActiveTab('downloadable')}
          className={`flex-1 px-2 py-2 font-medium text-xs transition-colors ${
            activeTab === 'downloadable'
              ? 'border-b-2 border-purple-600 text-purple-600'
              : 'text-gray-600 hover:text-gray-800'
          }`}
        >
          Download ({downloadableChatModels.length + downloadableEmbeddingModels.length + downloadingChatModels.length + downloadingEmbeddingModels.length})
        </button>
      </div>

      {/* Loaded Models Tab */}
      {activeTab === 'loaded' && (
        <div className="space-y-2">
          {loadedModels.length === 0 ? (
            <p className="text-gray-500 text-center py-6 text-sm">No models loaded</p>
          ) : (
            loadedModels.map((model) => (
              <div
                key={model.model_name}
                className="border border-green-300 rounded-lg p-3 bg-linear-to-br from-green-50 to-emerald-50 shadow-sm"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-sm text-gray-800 truncate">{model.model_name}</h3>
                    <p className="text-xs text-gray-600 mt-1">
                      <span className="inline-flex items-center px-2 py-0.5 rounded bg-green-100 text-green-800 font-medium">
                        {model.model_type}
                      </span>
                    </p>
                  </div>
                  <button
                    onClick={() => handleUnload(model.model_name, model.model_type as 'chat' | 'embedding')}
                    disabled={loading}
                    className="px-3 py-1.5 bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:bg-gray-400 text-xs font-medium shadow-sm"
                  >
                    Unload
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* Available Models Tab */}
      {activeTab === 'available' && (
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-semibold mb-2 text-gray-800 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-purple-600"></span>
              Chat Models
            </h3>
            {chatModels.length === 0 ? (
              <p className="text-gray-500 text-xs">No chat models available</p>
            ) : (
              <div className="space-y-2">
                {chatModels.map((model) => {
                  const isLoaded = loadedModels.some(
                    (m) => m.model_name === model.name && m.model_type === 'chat'
                  );
                  return (
                    <div
                      key={model.name}
                      className={`border rounded-lg p-3 shadow-sm ${
                        isLoaded ? 'bg-linear-to-br from-green-50 to-emerald-50 border-green-300' : 'bg-white border-gray-200'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <h4 className="font-semibold text-xs text-gray-800 truncate">{model.name}</h4>
                          <p className="text-xs text-gray-500 mt-1">{model.size}</p>
                        </div>
                        {isLoaded ? (
                          <span className="px-2 py-1 bg-green-500 text-white rounded text-xs font-medium">
                            Loaded
                          </span>
                        ) : (
                          <button
                            onClick={() => handleLoad(model.name, 'chat')}
                            disabled={loading}
                            className="px-3 py-1 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400 text-xs font-medium shadow-sm"
                          >
                            Load
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div>
            <h3 className="text-sm font-semibold mb-2 text-gray-800 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-600"></span>
              Embedding Models
            </h3>
            {embeddingModels.length === 0 ? (
              <p className="text-gray-500 text-xs">No embedding models available</p>
            ) : (
              <div className="space-y-2">
                {embeddingModels.map((model) => {
                  const isLoaded = loadedModels.some(
                    (m) => m.model_name === model.name && m.model_type === 'embedding'
                  );
                  return (
                    <div
                      key={model.name}
                      className={`border rounded-lg p-3 shadow-sm ${
                        isLoaded ? 'bg-linear-to-br from-green-50 to-emerald-50 border-green-300' : 'bg-white border-gray-200'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <h4 className="font-semibold text-xs text-gray-800 truncate">{model.name}</h4>
                          <p className="text-xs text-gray-500 mt-1">{model.size}</p>
                        </div>
                        {isLoaded ? (
                          <span className="px-2 py-1 bg-green-500 text-white rounded text-xs font-medium">
                            Loaded
                          </span>
                        ) : (
                          <button
                            onClick={() => handleLoad(model.name, 'embedding')}
                            disabled={loading}
                            className="px-3 py-1 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 text-xs font-medium shadow-sm"
                          >
                            Load
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Downloadable Models Tab */}
      {activeTab === 'downloadable' && (
        <div className="space-y-4">
          {/* Downloading Models Section */}
          {(downloadingChatModels.length > 0 || downloadingEmbeddingModels.length > 0) && (
            <div>
              <h3 className="text-sm font-semibold mb-2 text-gray-800 flex items-center gap-2">
                <svg className="animate-spin h-4 w-4 text-orange-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Downloading
              </h3>
              <div className="space-y-2 mb-4">
                {downloadingChatModels.map((model) => (
                  <div
                    key={model.name}
                    className="border border-orange-300 rounded-lg p-3 bg-orange-50 shadow-sm"
                  >
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <div className="flex-1 min-w-0">
                        <h4 className="font-semibold text-xs text-gray-800 truncate">{model.name}</h4>
                        <p className="text-xs text-gray-600 mt-1">{model.size}</p>
                      </div>
                      <span className="px-2 py-1 bg-orange-500 text-white rounded text-xs font-medium flex items-center gap-1">
                        <svg className="animate-spin h-3 w-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        {model.status === 'ready_to_use' ? '100' : '0'}%
                      </span>
                    </div>
                  </div>
                ))}
                {downloadingEmbeddingModels.map((model) => (
                  <div
                    key={model.name}
                    className="border border-orange-300 rounded-lg p-3 bg-orange-50 shadow-sm"
                  >
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <div className="flex-1 min-w-0">
                        <h4 className="font-semibold text-xs text-gray-800 truncate">{model.name}</h4>
                        <p className="text-xs text-gray-600 mt-1">{model.size}</p>
                      </div>
                      <span className="px-2 py-1 bg-orange-500 text-white rounded text-xs font-medium flex items-center gap-1">
                        <svg className="animate-spin h-3 w-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        {model.status === 'ready_to_use' ? '100' : '0'}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div>
            <h3 className="text-sm font-semibold mb-2 text-gray-800 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-purple-600"></span>
              Chat Models
            </h3>
            {downloadableChatModels.length === 0 ? (
              <p className="text-gray-500 text-xs">No downloadable chat models</p>
            ) : (
              <div className="space-y-2">
                {downloadableChatModels.map((model) => {
                  const isAvailable = allChatModels.some(
                    (m) => m.name === model.name && (m.status === 'ready_to_use' || m.status === 'downloading')
                  );
                  return (
                    <div
                      key={model.repository}
                      className="border border-gray-200 rounded-lg p-3 bg-white shadow-sm"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <h4 className="font-semibold text-xs text-gray-800 truncate">{model.name}</h4>
                          <p className="text-xs text-gray-500 mt-1 truncate">{model.filename}</p>
                        </div>
                        <button
                          onClick={() => handleDownload(model.repository, model.name)}
                          disabled={loading || isAvailable}
                          className="px-3 py-1 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400 text-xs font-medium shadow-sm whitespace-nowrap"
                        >
                          {isAvailable ? 'Available' : 'Download'}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div>
            <h3 className="text-sm font-semibold mb-2 text-gray-800 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-600"></span>
              Embedding Models
            </h3>
            {downloadableEmbeddingModels.length === 0 ? (
              <p className="text-gray-500 text-xs">No downloadable embedding models</p>
            ) : (
              <div className="space-y-2">
                {downloadableEmbeddingModels.map((model) => {
                  const isAvailable = allEmbeddingModels.some(
                    (m) => m.name === model.name && (m.status === 'ready_to_use' || m.status === 'downloading')
                  );
                  return (
                    <div
                      key={model.repository}
                      className="border border-gray-200 rounded-lg p-3 bg-white shadow-sm"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <h4 className="font-semibold text-xs text-gray-800 truncate">{model.name}</h4>
                          <p className="text-xs text-gray-500 mt-1 truncate">{model.filename}</p>
                        </div>
                        <button
                          onClick={() => handleDownload(model.repository, model.name)}
                          disabled={loading || isAvailable}
                          className="px-3 py-1 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 text-xs font-medium shadow-sm whitespace-nowrap"
                        >
                          {isAvailable ? 'Available' : 'Download'}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

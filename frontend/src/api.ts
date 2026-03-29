import axios from 'axios';
import type { QueryResponse, IngestionLog } from './types';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const queryMedicalAssistant = async (question: string, chat_history?: { role: string; content: string }[]): Promise<QueryResponse> => {
  const response = await api.post<QueryResponse>('/query', { question, chat_history });
  return response.data;
};

export const ingestDocuments = async (file: File, onProgress: (log: IngestionLog) => void): Promise<void> => {
  // First, we need to upload the file to the 'data' directory.
  // Since the current FastAPI backend doesn't have a file upload endpoint (Streamlit was doing it manually),
  // I should add a file upload endpoint to FastAPI as well to make it work with React.
  
  // For now, I'll assume the backend will be updated to handle file uploads.
  const formData = new FormData();
  formData.append('file', file);
  
  await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });

  // Then trigger ingestion
  const response = await fetch(`${API_BASE_URL}/ingest`, {
    method: 'POST',
  });

  if (!response.body) return;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    
    // Split by newline and parse each line
    const lines = chunk.split('\n').filter(l => l.trim());
    for (const line of lines) {
      try {
        const logData = JSON.parse(line) as IngestionLog;
        onProgress(logData);
      } catch (e) {
        console.warn('Failed to parse log line:', line, e);
      }
    }
  }
};

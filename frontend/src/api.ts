import axios from 'axios';
import type { QueryResponse } from './types';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const queryMedicalAssistant = async (question: string): Promise<QueryResponse> => {
  const response = await api.post<QueryResponse>('/query', { question });
  return response.data;
};

export const ingestDocuments = async (file: File, onProgress: (msg: string) => void): Promise<void> => {
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
    onProgress(chunk);
  }
};

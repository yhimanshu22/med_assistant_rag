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

export const queryMedicalAssistantStream = async (
  question: string,
  chat_history: { role: string; content: string }[] | undefined,
  onEvent: (evt: { type: string; text?: string; sources?: any; confidence?: number; metrics?: any; total_time?: string; message?: string }) => void
): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, chat_history }),
  });

  if (!response.body) return;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        onEvent(JSON.parse(trimmed));
      } catch {
        // ignore parse errors on partial chunks
      }
    }
  }
};

export const ingestDocuments = async (file: File, onProgress: (log: IngestionLog) => void): Promise<void> => {
  // First, we need to upload the file to the 'data' directory.
  // The backend exposes `/upload` to accept PDFs, then `/ingest` to index them.
  const formData = new FormData();
  formData.append('file', file);
  
  await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });

  // Then trigger ingestion asynchronously and poll for status/logs.
  const startResp = await fetch(`${API_BASE_URL}/ingest/async`, { method: 'POST' });
  const startData = (await startResp.json()) as { job_id: string };
  const jobId = startData.job_id;

  let lastLogCount = 0;
  while (true) {
    const statusResp = await fetch(`${API_BASE_URL}/ingest/${jobId}`);
    const statusData = (await statusResp.json()) as { status: string; logs: string[]; error?: string };

    const newLogs = statusData.logs.slice(lastLogCount);
    lastLogCount = statusData.logs.length;

    for (const line of newLogs) {
      try {
        const logData = JSON.parse(line) as IngestionLog;
        onProgress(logData);
      } catch {
        // ignore non-json log lines
      }
    }

    if (statusData.status === 'completed') return;
    if (statusData.status === 'failed') throw new Error(statusData.error || 'Ingestion failed');

    await new Promise((r) => setTimeout(r, 500));
  }
};

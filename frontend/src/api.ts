import axios from 'axios';
import type { QueryResponse, IngestionLog } from './types';

const API_BASE_URL = 'http://localhost:8000';
const TOKEN_KEY = 'medassist_token';

const getStoredToken = (): string | null => localStorage.getItem(TOKEN_KEY);

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  const token = getStoredToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

const authHeaders = (token?: string): HeadersInit => {
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  const resolved = token || getStoredToken();
  if (resolved) {
    headers.Authorization = `Bearer ${resolved}`;
  }
  return headers;
};

const parseApiError = async (response: Response): Promise<string> => {
  try {
    const data = await response.json();
    if (typeof data.detail === 'string') return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail.map((item: { msg?: string }) => item.msg || 'Validation error').join(', ');
    }
  } catch {
    // ignore JSON parse errors
  }
  return `Request failed (${response.status})`;
};

export interface AuthUser {
  email: string;
}

export interface AuthResult {
  access_token: string;
  token_type: string;
  user: AuthUser;
}

export const signup = async (email: string, password: string): Promise<AuthResult> => {
  const response = await fetch(`${API_BASE_URL}/auth/signup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  if (!response.ok) throw new Error(await parseApiError(response));
  return response.json();
};

export const login = async (email: string, password: string): Promise<AuthResult> => {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  if (!response.ok) throw new Error(await parseApiError(response));
  return response.json();
};

export const logout = async (token?: string): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/auth/logout`, {
    method: 'POST',
    headers: authHeaders(token),
  });
  if (!response.ok) throw new Error(await parseApiError(response));
};

export const getMe = async (token?: string): Promise<AuthUser> => {
  const response = await fetch(`${API_BASE_URL}/auth/me`, {
    headers: authHeaders(token),
  });
  if (!response.ok) throw new Error(await parseApiError(response));
  return response.json();
};

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
    headers: authHeaders(),
    body: JSON.stringify({ question, chat_history }),
  });

  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }

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
  const formData = new FormData();
  formData.append('file', file);

  const token = getStoredToken();
  await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });

  const startResp = await fetch(`${API_BASE_URL}/ingest/async`, {
    method: 'POST',
    headers: authHeaders(),
  });
  if (!startResp.ok) throw new Error(await parseApiError(startResp));

  const startData = (await startResp.json()) as { job_id: string };
  const jobId = startData.job_id;

  let lastLogCount = 0;
  while (true) {
    const statusResp = await fetch(`${API_BASE_URL}/ingest/${jobId}`, {
      headers: authHeaders(),
    });
    if (!statusResp.ok) throw new Error(await parseApiError(statusResp));

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

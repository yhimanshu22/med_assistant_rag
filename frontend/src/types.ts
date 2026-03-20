export interface DocumentMetadata {
  source: string;
  [key: string]: any;
}

export interface DocumentSource {
  page_content: string;
  metadata: DocumentMetadata;
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: DocumentSource[];
  timestamp: number;
  total_time?: string;
}

export interface QueryRequest {
  question: string;
}

export interface QueryResponse {
  question: string;
  answer: string;
  source_documents: DocumentSource[];
  total_time: string;
}

export interface IngestionLog {
  step: 'scanning' | 'processing' | 'splitting' | 'embedding' | 'ingesting' | 'complete';
  message: string;
  file?: string;
  status?: 'info' | 'warning' | 'error';
  total_time?: string;
}



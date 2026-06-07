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
  confidence?: number;
  evaluationEnabled?: boolean;
  metrics?: {
    faithfulness: number;
    relevance: number;
  };
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  timestamp: number;
}

export interface QueryRequest {
  question: string;
  chat_history?: { role: string; content: string }[];
  enable_evaluation?: boolean;
}

export interface QueryResponse {
  question: string;
  answer: string;
  source_documents: DocumentSource[];
  total_time: string;
  confidence: number;
  evaluation_enabled: boolean;
  metrics: {
    faithfulness: number;
    relevance: number;
  };
}

export interface LatencySummary {
  count: number;
  avg_ms: number;
  p50_ms: number;
  p95_ms: number;
}

export interface EvalScoreSummary {
  count: number;
  avg: number;
}

export interface MetricsErrorEntry {
  at: number;
  event: string;
  error: string;
  path?: string | null;
  request_id?: string | null;
}

export interface MetricsSnapshot {
  queries_total: number;
  errors_total: number;
  cache_hits: number;
  conversational_queries: number;
  retrieval_hit_rate: number;
  retrieval_hits: number;
  retrieval_misses: number;
  latency_ms: LatencySummary;
  stages_ms: Record<'rewrite' | 'retrieve' | 'llm' | 'eval', LatencySummary>;
  evaluation_scores: {
    faithfulness: EvalScoreSummary;
    relevance: EvalScoreSummary;
    confidence: EvalScoreSummary;
  };
  recent_errors: MetricsErrorEntry[];
}

export interface IngestionLog {
  step: 'scanning' | 'processing' | 'splitting' | 'embedding' | 'ingesting' | 'complete';
  message: string;
  file?: string;
  status?: 'info' | 'warning' | 'error';
  total_time?: string;
}



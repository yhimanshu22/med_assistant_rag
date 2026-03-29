import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, 
  Upload, 
  Trash2, 
  Stethoscope, 
  FileText, 
  Loader2, 
  ChevronRight,
  User,
  Bot
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Message, IngestionLog } from './types';
import { queryMedicalAssistant, ingestDocuments } from './api';
import { 
  CheckCircle2, 
  AlertCircle, 
  Search, 
  Cpu, 
  Database, 
  Layers, 
  FileCheck,
  Timer,
  ShieldCheck
} from 'lucide-react';
import { Routes, Route, useNavigate } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestionLogs, setIngestionLogs] = useState<IngestionLog[]>([]);
  const [expandedSources, setExpandedSources] = useState<Record<number, boolean>>({});
  
  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    const savedMessages = localStorage.getItem('medassist_chat_history');
    if (savedMessages) {
      try {
        setMessages(JSON.parse(savedMessages));
      } catch (e) {
        console.error('Failed to parse saved messages:', e);
      }
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
    if (messages.length > 0) {
      localStorage.setItem('medassist_chat_history', JSON.stringify(messages));
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: Date.now(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    const chatHistory = messages.map(msg => ({
      role: msg.role,
      content: msg.content
    }));

    try {
      const response = await queryMedicalAssistant(input, chatHistory);
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.answer,
        sources: response.source_documents,
        timestamp: Date.now(),
        total_time: response.total_time,
        confidence: response.confidence, // Add confidence
        metrics: response.metrics,       // Add metrics
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error fetching response:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error while processing your request. Please ensure the backend server is running.',
        timestamp: Date.now(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsIngesting(true);
    setIngestionLogs([{ step: 'scanning', message: 'Starting ingestion...' }]);

    try {
      await ingestDocuments(file, (log) => {
        setIngestionLogs(prev => [...prev, log]);
      });
    } catch (error) {
      console.error('Ingestion failed:', error);
      setIngestionLogs(prev => [...prev, { 
        step: 'processing', 
        message: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        status: 'error' 
      }]);
    } finally {
      setIsIngesting(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const toggleSources = (index: number) => {
    setExpandedSources(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const clearChat = () => {
    setMessages([]);
    setExpandedSources({});
    localStorage.removeItem('medassist_chat_history');
  };

  const navigate = useNavigate();

  return (
    <AnimatePresence mode="wait">
      <Routes>
        <Route path="/" element={
          <motion.div
            key="landing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          >
            <LandingPage onStart={() => navigate('/chat')} />
          </motion.div>
        } />
        
        <Route path="/chat" element={
          <motion.div
            key="chat"
            className="app-container"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {/* Sidebar */}
            <aside className="sidebar">
              <div className="sidebar-header">
                <Stethoscope size={32} color="var(--primary)" strokeWidth={2.5} />
                <h1 onClick={() => navigate('/')} style={{ cursor: 'pointer' }}>MedAssist RAG</h1>
              </div>

            <div className="sidebar-section">
              <h2>Document Management</h2>
              <div 
                className="upload-area"
                onClick={() => fileInputRef.current?.click()}
              >
                {isIngesting ? (
                  <Loader2 className="animate-spin" size={24} color="var(--primary)" style={{ margin: '0 auto 0.5rem' }} />
                ) : (
                  <Upload size={24} color="var(--primary)" style={{ margin: '0 auto 0.5rem' }} />
                )}
                <p style={{ fontSize: '0.875rem', fontWeight: 500 }}>
                  {isIngesting ? 'Ingesting...' : 'Upload Medical PDF'}
                </p>
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileUpload} 
                  accept=".pdf" 
                  style={{ display: 'none' }}
                  disabled={isIngesting}
                />
              </div>
              
              {ingestionLogs.length > 0 && (
                <div className="ingestion-log">
                  {ingestionLogs.map((log, i) => {
                    const isLast = i === ingestionLogs.length - 1;
                    let Icon = Search;
                    if (log.step === 'processing') Icon = FileCheck;
                    if (log.step === 'splitting') Icon = Layers;
                    if (log.step === 'embedding') Icon = Cpu;
                    if (log.step === 'ingesting') Icon = Database;
                    if (log.step === 'complete') Icon = CheckCircle2;
                    
                    return (
                      <div key={i} className={`ingestion-item ${isLast && isIngesting ? 'active' : ''}`}>
                        <div className="ingestion-item-icon">
                          {log.status === 'error' ? (
                            <AlertCircle size={16} color="#ef4444" />
                          ) : (log.status === 'warning' ? (
                            <AlertCircle size={16} color="#f59e0b" />
                          ) : (
                            <Icon size={16} color={log.step === 'complete' ? '#10b981' : 'var(--primary)'} />
                          ))}
                        </div>
                        <div className="ingestion-item-content">
                          <div className="ingestion-item-message">
                            {log.message}
                            {log.step === 'complete' && log.total_time && (
                              <span style={{ marginLeft: '8px', color: '#10b981', fontWeight: 600 }}>
                                ({log.total_time})
                              </span>
                            )}
                          </div>
                          {log.file && <div className="ingestion-item-file">{log.file}</div>}
                        </div>
                      </div>
                    );
                  })}
                  
                  {!isIngesting && ingestionLogs.some(l => l.step === 'complete') && (
                    <div className="ingestion-status">
                      <span>Process complete</span>
                      <CheckCircle2 size={14} color="#10b981" />
                    </div>
                  )}
                </div>
              )}
            </div>

            <div style={{ marginTop: 'auto' }}>
              <button 
                className="btn-clear" 
                onClick={clearChat}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.5rem',
                  padding: '0.75rem',
                  borderRadius: 'var(--radius)',
                  border: '1px solid var(--border)',
                  background: 'transparent',
                  color: '#dc2626',
                  fontWeight: 500
                }}
              >
                <Trash2 size={18} />
                Clear Chat
              </button>
            </div>
          </aside>

          {/* Main Content */}
          <main className="main-content">
            <div className="chat-history">
              {messages.length === 0 ? (
                <div className="welcome-screen">
                  <Stethoscope size={64} color="var(--border)" style={{ marginBottom: '1.5rem' }} />
                  <h2 style={{ color: 'var(--text-main)', marginBottom: '0.5rem' }}>Medical Assistant</h2>
                  <p>Ask a question about your medical documents to get started.</p>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <div key={idx} className={`message ${msg.role}`}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px', alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
                      {msg.role === 'user' ? <User size={14} /> : <Bot size={14} />}
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {msg.role === 'user' ? 'You' : 'Assistant'}
                      </span>
                    </div>
                    <div className="message-bubble">
                      {msg.role === 'assistant' ? (
                        <div className="markdown-content">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {msg.content}
                          </ReactMarkdown>
                        </div>
                      ) : (
                        msg.content
                      )}
                    </div>
                    
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="sources-container">
                        <button 
                          onClick={() => toggleSources(idx)}
                          style={{
                            background: 'transparent',
                            border: 'none',
                            color: 'var(--primary)',
                            fontSize: '0.75rem',
                            fontWeight: 600,
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                            padding: '4px 0'
                          }}
                        >
                          <FileText size={14} />
                          {expandedSources[idx] ? 'Hide Sources' : `View ${msg.sources.length} Sources`}
                          <ChevronRight size={14} style={{ transform: expandedSources[idx] ? 'rotate(90deg)' : 'none', transition: 'transform 0.2s' }} />
                        </button>
                        
                        <AnimatePresence>
                          {expandedSources[idx] && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: 'auto', opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              style={{ overflow: 'hidden' }}
                            >
                              {msg.sources.map((source, sIdx) => (
                                <div key={sIdx} className="source-item">
                                  <div className="source-header">Source {sIdx + 1}: {source.metadata.source}</div>
                                  <div className="source-content">{source.page_content}</div>
                                </div>
                              ))}
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    )}
                    
                    {msg.confidence !== undefined && (
                      <div className="trust-indicator-container">
                        <div className="trust-score-badge" style={{
                          background: msg.confidence > 0.8 ? '#ecfdf5' : (msg.confidence > 0.5 ? '#fffbeb' : '#fef2f2'),
                          color: msg.confidence > 0.8 ? '#065f46' : (msg.confidence > 0.5 ? '#92400e' : '#991b1b'),
                          border: `1px solid ${msg.confidence > 0.8 ? '#10b981' : (msg.confidence > 0.5 ? '#f59e0b' : '#ef4444')}`
                        }}>
                          <ShieldCheck size={14} />
                          <span className="trust-label">Trust Score: {Math.round(msg.confidence * 100)}%</span>
                          <div className="trust-tooltip">
                            <div className="tooltip-item">
                              <span>Faithfulness:</span>
                              <strong>{Math.round((msg.metrics?.faithfulness || 0) * 100)}%</strong>
                            </div>
                            <div className="tooltip-item">
                              <span>Relevance:</span>
                              <strong>{Math.round((msg.metrics?.relevance || 0) * 100)}%</strong>
                            </div>
                          </div>
                        </div>
                        {msg.total_time && (
                          <div className="response-meta">
                            <Timer size={12} />
                            <span>{msg.total_time}</span>
                          </div>
                        )}
                        
                        <div className="evaluation-metrics-row">
                          <div className="metric-chip faithfulness" title="Faithfulness (Groundedness)">
                            <FileCheck size={12} />
                            <span>Faith: {Math.round((msg.metrics?.faithfulness || 0) * 100)}%</span>
                          </div>
                          <div className="metric-chip relevance" title="Answer Relevance">
                            <Search size={12} />
                            <span>Rel: {Math.round((msg.metrics?.relevance || 0) * 100)}%</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))
              )}
              {isLoading && (
                <div className="message assistant">
                  <div className="message-bubble" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Loader2 className="animate-spin" size={18} />
                    Thinking...
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div className="input-area">
              <form 
                onSubmit={(e) => { e.preventDefault(); handleSend(); }}
                className="input-container"
              >
                <input 
                  type="text" 
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="e.g., What are the symptoms of Influenza?"
                  disabled={isLoading}
                />
                <button 
                  type="submit" 
                  className="send-button"
                  disabled={isLoading || !input.trim()}
                >
                  <Send size={18} />
                </button>
              </form>
            </div>
          </main>
        </motion.div>
      } />
    </Routes>
  </AnimatePresence>
  );
};

export default App;

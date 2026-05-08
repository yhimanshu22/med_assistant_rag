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
import type { Message, IngestionLog, Conversation } from './types';
import { queryMedicalAssistantStream, ingestDocuments } from './api';
import { 
  Plus,
  MessageSquare,
  Search, 
  Cpu, 
  Database, 
  Layers, 
  FileCheck,
  Timer,
  ShieldCheck,
  CheckCircle2, 
  AlertCircle
} from 'lucide-react';
import { Routes, Route, useNavigate } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

const App: React.FC = () => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  
  const activeConversation = conversations.find(c => c.id === activeConversationId);
  const messages = activeConversation?.messages || [];

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

  const startNewChat = () => {
    const newId = Date.now().toString();
    const newConv: Conversation = {
      id: newId,
      title: 'New Chat',
      messages: [],
      timestamp: Date.now()
    };
    setConversations(prev => [newConv, ...prev]);
    setActiveConversationId(newId);
    setExpandedSources({});
  };

  const updateActiveConversation = (updateFn: (msgs: Message[]) => Message[]) => {
    let currentId = activeConversationId;
    
    if (!currentId) {
      const newId = Date.now().toString();
      const newConv: Conversation = {
        id: newId,
        title: 'New Chat',
        messages: updateFn([]),
        timestamp: Date.now()
      };
      setConversations(prev => [newConv, ...prev]);
      setActiveConversationId(newId);
      return;
    }

    setConversations(prev => prev.map(c => {
      if (c.id === currentId) {
        const newMessages = updateFn(c.messages);
        let title = c.title;
        if (title === 'New Chat' && newMessages.length > 0 && newMessages[0].role === 'user') {
          title = newMessages[0].content.slice(0, 30) + (newMessages[0].content.length > 30 ? '...' : '');
        }
        return { ...c, messages: newMessages, title, timestamp: Date.now() };
      }
      return c;
    }));
  };

  useEffect(() => {
    const savedConversations = localStorage.getItem('medassist_conversations');
    const savedActiveId = localStorage.getItem('medassist_active_id');
    
    if (savedConversations) {
      try {
        const parsed = JSON.parse(savedConversations);
        setConversations(parsed);
        if (savedActiveId) setActiveConversationId(savedActiveId);
        else if (parsed.length > 0) setActiveConversationId(parsed[0].id);
      } catch (e) {
        console.error('Failed to parse saved conversations:', e);
      }
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
    if (conversations.length > 0) {
      localStorage.setItem('medassist_conversations', JSON.stringify(conversations));
      if (activeConversationId) {
        localStorage.setItem('medassist_active_id', activeConversationId);
      }
    }
  }, [conversations, activeConversationId]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: Date.now(),
    };

    updateActiveConversation(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    const chatHistory = messages.map(msg => ({
      role: msg.role,
      content: msg.content
    }));

    try {
      const assistantMessage: Message = {
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
      };
      updateActiveConversation(prev => [...prev, assistantMessage]);

      let finalSources: any[] | undefined;
      let finalConfidence: number | undefined;
      let finalMetrics: any | undefined;
      let finalTotalTime: string | undefined;

      await queryMedicalAssistantStream(input, chatHistory, (evt) => {
        if (evt.type === 'meta') {
          finalSources = evt.sources;
          finalConfidence = evt.confidence;
          finalMetrics = evt.metrics;
          updateActiveConversation(prev => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === 'assistant') {
              last.sources = finalSources as any;
              last.confidence = finalConfidence;
              last.metrics = finalMetrics;
            }
            return next;
          });
        } else if (evt.type === 'delta' && evt.text) {
          updateActiveConversation(prev => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === 'assistant') {
              last.content = (last.content || '') + evt.text;
            }
            return next;
          });
        } else if (evt.type === 'done') {
          finalTotalTime = evt.total_time;
          updateActiveConversation(prev => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === 'assistant') {
              last.total_time = finalTotalTime;
            }
            return next;
          });
        } else if (evt.type === 'error') {
          updateActiveConversation(prev => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === 'assistant') {
              last.content = evt.message || 'Streaming error';
            }
            return next;
          });
        }
      });
    } catch (error) {
      console.error('Error fetching response:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error while processing your request. Please ensure the backend server is running.',
        timestamp: Date.now(),
      };
      updateActiveConversation(prev => [...prev, errorMessage]);
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
    if (activeConversationId) {
      setConversations(prev => prev.filter(c => c.id !== activeConversationId));
      const remaining = conversations.filter(c => c.id !== activeConversationId);
      if (remaining.length > 0) setActiveConversationId(remaining[0].id);
      else setActiveConversationId(null);
    }
    setExpandedSources({});
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
                <h1 onClick={() => navigate('/')} style={{ cursor: 'pointer' }}>MedAssist</h1>
              </div>

              <button className="btn-new-chat" onClick={startNewChat}>
                <Plus size={18} />
                New Conversation
              </button>

              <div className="sidebar-section history-section">
                <h2>Recent Conversations</h2>
                <div className="conversation-list">
                  {conversations.map(conv => (
                    <div 
                      key={conv.id} 
                      className={`conversation-item ${conv.id === activeConversationId ? 'active' : ''}`}
                      onClick={() => setActiveConversationId(conv.id)}
                    >
                      <MessageSquare size={16} />
                      <span className="conversation-title">{conv.title}</span>
                    </div>
                  ))}
                  {conversations.length === 0 && (
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center', padding: '1rem' }}>
                      No recent chats
                    </p>
                  )}
                </div>
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

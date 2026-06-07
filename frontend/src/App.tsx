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
import { queryMedicalAssistantStream, ingestDocuments, getHealth } from './api';
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
  ShieldOff,
  CheckCircle2, 
  AlertCircle,
  LogOut,
  Square,
  ChevronDown
} from 'lucide-react';
import { Routes, Route, useNavigate } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import LoginPage from './components/LoginPage';
import SignupPage from './components/SignupPage';
import ProtectedRoute from './components/ProtectedRoute';
import { useAuth } from './context/AuthContext';
import MarkdownAnswer from './components/MarkdownAnswer';
import ModelMetricsDrawer from './components/ModelMetricsDrawer';
import './App.css';

const WELCOME_PROMPTS = [
  'What is aplastic anemia?',
  'What are the symptoms of Influenza?',
  'Explain panhypoplasia of the marrow',
];

const getTrustClass = (score: number) => {
  if (score > 0.8) return 'high';
  if (score > 0.5) return 'mid';
  return 'low';
};

const ZERO_METRICS = { faithfulness: 0, relevance: 0 };

const isMessageEvalOn = (msg: Message) => msg.evaluationEnabled === true;

const getDisplayConfidence = (msg: Message) =>
  isMessageEvalOn(msg) ? Math.round((msg.confidence || 0) * 100) : 0;

const getDisplayMetric = (msg: Message, key: 'faithfulness' | 'relevance') =>
  (isMessageEvalOn(msg) ? Math.round((msg.metrics?.[key] || 0) * 100) : 0);

const normalizeStoredMessage = (msg: Message): Message => {
  if (msg.role !== 'assistant' || isMessageEvalOn(msg)) {
    return msg;
  }
  return {
    ...msg,
    evaluationEnabled: false,
    confidence: 0,
    metrics: ZERO_METRICS,
  };
};

const normalizeStoredConversations = (convs: Conversation[]): Conversation[] =>
  convs.map((c) => ({
    ...c,
    messages: c.messages.map(normalizeStoredMessage),
  }));

const ChatApp: React.FC = () => {
  const { email, logout } = useAuth();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  
  const activeConversation = conversations.find(c => c.id === activeConversationId);
  const messages = activeConversation?.messages || [];

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [metricsRefreshKey, setMetricsRefreshKey] = useState(0);
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestionLogs, setIngestionLogs] = useState<IngestionLog[]>([]);
  const [expandedSources, setExpandedSources] = useState<Record<number, boolean>>({});
  
  const chatEndRef = useRef<HTMLDivElement>(null);
  const chatHistoryRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const streamBufferRef = useRef('');
  const followAnswerScrollRef = useRef(false);
  const prevActiveIdRef = useRef<string | null>(null);
  const [showScrollDown, setShowScrollDown] = useState(false);
  const [evaluationAvailable, setEvaluationAvailable] = useState(true);
  const [userEvaluationEnabled, setUserEvaluationEnabled] = useState(
    () => localStorage.getItem('medassist_user_evaluation') === 'true',
  );

  const handleEvaluationToggle = (enabled: boolean) => {
    setUserEvaluationEnabled(enabled);
    localStorage.setItem('medassist_user_evaluation', String(enabled));
  };

  useEffect(() => {
    getHealth()
      .then((health) => {
        const available = health.evaluation_available;
        setEvaluationAvailable(available);
        if (!available) {
          handleEvaluationToggle(false);
        }
      })
      .catch(() => setEvaluationAvailable(false));
  }, []);

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
        const parsed = JSON.parse(savedConversations) as Conversation[];
        setConversations(normalizeStoredConversations(parsed));
        if (savedActiveId) setActiveConversationId(savedActiveId);
        else if (parsed.length > 0) setActiveConversationId(parsed[0].id);
      } catch (e) {
        console.error('Failed to parse saved conversations:', e);
      }
    }
  }, []);

  const handleChatScroll = () => {
    const el = chatHistoryRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceFromBottom > 120) {
      followAnswerScrollRef.current = false;
      setShowScrollDown(true);
    } else {
      setShowScrollDown(false);
    }
  };

  const scrollToBottom = (behavior: ScrollBehavior = 'smooth') => {
    const el = chatHistoryRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior });
    setShowScrollDown(false);
  };

  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem('medassist_conversations', JSON.stringify(conversations));
      if (activeConversationId) {
        localStorage.setItem('medassist_active_id', activeConversationId);
      }
    }
  }, [conversations, activeConversationId]);

  useEffect(() => {
    if (activeConversationId !== prevActiveIdRef.current) {
      prevActiveIdRef.current = activeConversationId;
      followAnswerScrollRef.current = false;
      requestAnimationFrame(() => scrollToBottom('auto'));
      return;
    }

    if (followAnswerScrollRef.current) {
      requestAnimationFrame(() => scrollToBottom(isLoading ? 'auto' : 'smooth'));
      return;
    }

    const el = chatHistoryRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceFromBottom <= 120) {
      requestAnimationFrame(() => scrollToBottom(isLoading ? 'auto' : 'smooth'));
    } else {
      setShowScrollDown(true);
    }
  }, [conversations, activeConversationId, isLoading]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: Date.now(),
    };

    updateActiveConversation(prev => [...prev, userMessage]);
    const question = input;
    setInput('');
    setIsLoading(true);
    streamBufferRef.current = '';
    followAnswerScrollRef.current = true;

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

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
      requestAnimationFrame(() => {
        requestAnimationFrame(() => scrollToBottom('smooth'));
      });

      let finalSources: any[] | undefined;
      let finalConfidence: number | undefined;
      let finalMetrics: any | undefined;
      let finalEvaluationEnabled: boolean | undefined;
      let finalTotalTime: string | undefined;

      const requestEvaluation = evaluationAvailable && userEvaluationEnabled;

      await queryMedicalAssistantStream(question, chatHistory, (evt) => {
        if (evt.type === 'meta') {
          finalSources = evt.sources;
          finalConfidence = evt.confidence;
          finalMetrics = evt.metrics;
          finalEvaluationEnabled = evt.evaluation_enabled ?? false;
          if (!finalEvaluationEnabled) {
            finalConfidence = 0;
            finalMetrics = ZERO_METRICS;
          }
          updateActiveConversation(prev => {
            const next = [...prev];
            const lastIdx = next.length - 1;
            const last = next[lastIdx];
            if (last?.role === 'assistant') {
              next[lastIdx] = {
                ...last,
                sources: finalSources as any,
                confidence: finalConfidence,
                metrics: finalMetrics,
                evaluationEnabled: finalEvaluationEnabled,
              };
            }
            return next;
          });
        } else if (evt.type === 'delta' && evt.text) {
          streamBufferRef.current = evt.text.startsWith(streamBufferRef.current)
            ? evt.text
            : streamBufferRef.current + evt.text;
          const snapshot = streamBufferRef.current;
          updateActiveConversation(prev => {
            const next = [...prev];
            const lastIdx = next.length - 1;
            const last = next[lastIdx];
            if (last?.role === 'assistant') {
              next[lastIdx] = { ...last, content: snapshot };
            }
            return next;
          });
        } else if (evt.type === 'done') {
          finalTotalTime = evt.total_time;
          updateActiveConversation(prev => {
            const next = [...prev];
            const lastIdx = next.length - 1;
            const last = next[lastIdx];
            if (last?.role === 'assistant') {
              next[lastIdx] = { ...last, total_time: finalTotalTime };
            }
            return next;
          });
        } else if (evt.type === 'error') {
          updateActiveConversation(prev => {
            const next = [...prev];
            const lastIdx = next.length - 1;
            const last = next[lastIdx];
            if (last?.role === 'assistant') {
              next[lastIdx] = { ...last, content: evt.message || 'Streaming error' };
            }
            return next;
          });
        }
      }, controller.signal, requestEvaluation);
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        return;
      }
      console.error('Error fetching response:', error);
      updateActiveConversation(prev => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === 'assistant' && !last.content?.trim()) {
          last.content = 'Sorry, I encountered an error while processing your request. Please ensure the backend server is running.';
          return next;
        }
        return [...prev, {
          role: 'assistant',
          content: 'Sorry, I encountered an error while processing your request. Please ensure the backend server is running.',
          timestamp: Date.now(),
        }];
      });
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
      setIsLoading(false);
      setMetricsRefreshKey((k) => k + 1);
    }
  };

  const handleStop = () => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsLoading(false);
    updateActiveConversation(prev => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last?.role === 'assistant') {
        if (!last.content?.trim()) {
          last.content = '*Response stopped.*';
        } else if (!last.content.includes('Response stopped')) {
          last.content += '\n\n---\n*Response stopped.*';
        }
      }
      return next;
    });
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

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  return (
    <motion.div
      key="chat"
      className="app-container"
      style={{ height: '100%', minHeight: 0 }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
            <aside className="sidebar">
              <div className="sidebar-top">
                <div className="sidebar-header">
                  <div className="sidebar-logo-icon">
                    <Stethoscope size={20} strokeWidth={2.5} />
                  </div>
                  <div>
                    <h1 onClick={() => navigate('/')}>MedAssist</h1>
                    {email && <p className="user-email">{email}</p>}
                  </div>
                </div>

                <button className="btn-new-chat" onClick={startNewChat}>
                  <Plus size={18} />
                  New Conversation
                </button>
              </div>

              <div className="sidebar-body">
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
                      <p className="empty-history">No recent chats</p>
                    )}
                  </div>
                </div>

                <div className="sidebar-section">
                  <h2>Response Settings</h2>
                  <label className={`eval-toggle ${!evaluationAvailable ? 'disabled' : ''}`}>
                    <input
                      type="checkbox"
                      checked={userEvaluationEnabled}
                      disabled={!evaluationAvailable || isLoading}
                      onChange={(e) => handleEvaluationToggle(e.target.checked)}
                    />
                    <span className="eval-toggle-slider" />
                    <span className="eval-toggle-label">Ragas evaluation</span>
                  </label>
                  <p className="eval-toggle-hint">
                    {!evaluationAvailable
                      ? 'Disabled by server admin.'
                      : userEvaluationEnabled
                        ? 'Trust scores on — responses will be slower.'
                        : 'Trust scores off — faster responses (0%).'}
                  </p>
                </div>

                <div className="sidebar-section">
                  <h2>Document Management</h2>
                  <div
                    className="upload-area"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    {isIngesting ? (
                      <Loader2 className="animate-spin" size={22} color="var(--primary)" />
                    ) : (
                      <Upload size={22} color="var(--primary)" />
                    )}
                    <p>{isIngesting ? 'Ingesting...' : 'Upload Medical PDF'}</p>
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
              </div>

              <ModelMetricsDrawer refreshKey={metricsRefreshKey} />

              <div className="sidebar-footer">
                <button className="sidebar-footer-btn" onClick={handleLogout}>
                  <LogOut size={16} />
                  Logout
                </button>
                <button className="sidebar-footer-btn danger" onClick={clearChat}>
                  <Trash2 size={16} />
                  Clear Chat
                </button>
              </div>
            </aside>

          <main className="main-content">
            <div className="chat-header">
              <div>
                <div className="chat-header-title">
                  {activeConversation?.title || 'New Conversation'}
                </div>
                <div className="chat-header-sub">Evidence-based answers from your documents</div>
              </div>
              <div className="chat-header-badges">
                <div className="chat-header-badge active">
                  <ShieldCheck size={12} />
                  RAG Active
                </div>
                <div className={`chat-header-badge ${
                  !evaluationAvailable
                    ? 'eval-unavailable'
                    : userEvaluationEnabled
                      ? 'eval-on'
                      : 'eval-off'
                }`}>
                  {!evaluationAvailable ? (
                    <ShieldOff size={12} />
                  ) : userEvaluationEnabled ? (
                    <ShieldCheck size={12} />
                  ) : (
                    <ShieldOff size={12} />
                  )}
                  {!evaluationAvailable
                    ? 'Eval Unavailable'
                    : userEvaluationEnabled
                      ? 'Evaluation On'
                      : 'Evaluation Off'}
                </div>
              </div>
            </div>

            <div className="chat-pane">
            <div
              className="chat-history"
              ref={chatHistoryRef}
              onScroll={handleChatScroll}
            >
              <div className="chat-messages">
              {messages.length === 0 ? (
                <div className="welcome-screen">
                  <div className="welcome-icon">
                    <Stethoscope size={36} strokeWidth={2} />
                  </div>
                  <h2>How can I help you today?</h2>
                  <p>Ask anything about your uploaded medical documents. Answers include citations and trust scores.</p>
                  <div className="welcome-prompts">
                    {WELCOME_PROMPTS.map((prompt) => (
                      <button
                        key={prompt}
                        className="welcome-prompt-btn"
                        onClick={() => setInput(prompt)}
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <div key={idx} className={`message ${msg.role}`}>
                    <div className="message-avatar">
                      {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                    </div>
                    <div className="message-body">
                      <span className="message-label">
                        {msg.role === 'user' ? 'You' : 'MedAssist'}
                      </span>
                      <div className="message-bubble">
                        {msg.role === 'assistant' ? (
                          !msg.content?.trim() && isLoading && idx === messages.length - 1 ? (
                            <div className="thinking-bubble">
                              <div className="thinking-dots">
                                <span /><span /><span />
                              </div>
                              Analyzing documents...
                            </div>
                          ) : (
                            <MarkdownAnswer content={msg.content} />
                          )
                        ) : (
                          msg.content
                        )}
                      </div>

                    {msg.sources && msg.sources.length > 0 && (
                      <div className="sources-container">
                        <button
                          className={`sources-toggle ${expandedSources[idx] ? 'open' : ''}`}
                          onClick={() => toggleSources(idx)}
                        >
                          <FileText size={14} />
                          {expandedSources[idx] ? 'Hide Sources' : `View ${msg.sources.length} Sources`}
                          <ChevronRight size={14} />
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
                    
                    {msg.role === 'assistant' && msg.confidence !== undefined && msg.content?.trim() && (
                      <div className="trust-indicator-container">
                        {(() => {
                          const evalOn = isMessageEvalOn(msg);
                          const trustPct = getDisplayConfidence(msg);
                          const faithPct = getDisplayMetric(msg, 'faithfulness');
                          const relPct = getDisplayMetric(msg, 'relevance');
                          return (
                            <>
                        <div className={`trust-score-badge ${evalOn ? getTrustClass(msg.confidence || 0) : 'eval-off'}`}>
                          {evalOn ? <ShieldCheck size={13} /> : <ShieldOff size={13} />}
                          <span>
                            {evalOn
                              ? `Trust ${trustPct}%`
                              : 'Trust 0% · Evaluation off'}
                          </span>
                          <div className="trust-tooltip">
                            {evalOn ? (
                              <>
                                <div className="tooltip-item">
                                  <span>Faithfulness:</span>
                                  <strong>{faithPct}%</strong>
                                </div>
                                <div className="tooltip-item">
                                  <span>Relevance:</span>
                                  <strong>{relPct}%</strong>
                                </div>
                              </>
                            ) : (
                              <span>Ragas trust scoring is disabled for faster responses.</span>
                            )}
                          </div>
                        </div>
                        {!evalOn && (
                          <div className="eval-status-chip" title="Ragas faithfulness/relevance scoring is disabled">
                            <ShieldOff size={12} />
                            <span>Evaluation off</span>
                          </div>
                        )}
                        {msg.total_time && (
                          <div className="response-meta">
                            <Timer size={12} />
                            <span>{msg.total_time}</span>
                          </div>
                        )}

                        <div className={`evaluation-metrics-row ${!evalOn ? 'disabled' : ''}`}>
                          <div className="metric-chip faithfulness" title="Faithfulness (Groundedness)">
                            <FileCheck size={12} />
                            <span>Faith: {faithPct}%</span>
                          </div>
                          <div className="metric-chip relevance" title="Answer Relevance">
                            <Search size={12} />
                            <span>Rel: {relPct}%</span>
                          </div>
                        </div>
                            </>
                          );
                        })()}
                      </div>
                    )}
                    </div>
                  </div>
                ))
              )}
              <div ref={chatEndRef} />
              </div>

            </div>

              {showScrollDown && (
                <button
                  type="button"
                  className="scroll-down-btn"
                  onClick={() => scrollToBottom()}
                  title="Scroll to latest message"
                >
                  <ChevronDown size={18} />
                </button>
              )}
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
                  placeholder="Ask a question about your medical documents..."
                  disabled={isLoading}
                />
                {isLoading ? (
                  <button
                    type="button"
                    className="stop-button"
                    onClick={handleStop}
                    title="Stop generating"
                  >
                    <Square size={15} fill="currentColor" />
                  </button>
                ) : (
                  <button
                    type="submit"
                    className="send-button"
                    disabled={!input.trim()}
                  >
                    <Send size={17} />
                  </button>
                )}
              </form>
              <p className="input-hint">Responses are grounded in your uploaded PDFs · Not a substitute for professional medical advice</p>
            </div>
          </main>
    </motion.div>
  );
};

const App: React.FC = () => {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();

  return (
    <AnimatePresence mode="wait">
      <Routes>
        <Route path="/" element={
          <motion.div
            key="landing"
            className="route-shell"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          >
            <LandingPage onStart={() => navigate(isAuthenticated ? '/chat' : '/login')} />
          </motion.div>
        } />

        <Route path="/login" element={<div className="route-shell"><LoginPage /></div>} />
        <Route path="/signup" element={<div className="route-shell"><SignupPage /></div>} />

        <Route path="/chat" element={
          <div className="route-shell route-shell--fill">
            <ProtectedRoute>
              <ChatApp />
            </ProtectedRoute>
          </div>
        } />
      </Routes>
    </AnimatePresence>
  );
};

export default App;

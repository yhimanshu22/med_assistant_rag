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
import type { Message } from './types';
import { queryMedicalAssistant, ingestDocuments } from './api';
import './App.css';

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestionLogs, setIngestionLogs] = useState<string[]>([]);
  const [expandedSources, setExpandedSources] = useState<Record<number, boolean>>({});
  
  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
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

    try {
      const response = await queryMedicalAssistant(input);
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.answer,
        sources: response.source_documents,
        timestamp: Date.now(),
        total_time: response.total_time,
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
    setIngestionLogs(["Starting ingestion..."]);

    try {
      await ingestDocuments(file, (msg) => {
        setIngestionLogs(prev => [...prev, msg]);
      });
      setIngestionLogs(prev => [...prev, "Ingestion complete!"]);
    } catch (error) {
      console.error('Ingestion failed:', error);
      setIngestionLogs(prev => [...prev, `Error: ${error instanceof Error ? error.message : 'Unknown error'}`]);
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
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <Stethoscope size={32} color="var(--primary)" strokeWidth={2.5} />
          <h1>MedAssist RAG</h1>
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
              {ingestionLogs.map((log, i) => (
                <div key={i}>{log}</div>
              ))}
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
                    <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
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
                
                {msg.total_time && (
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px', alignSelf: 'flex-start' }}>
                    Speed: {msg.total_time}
                  </span>
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
    </div>
  );
};

export default App;

import React from 'react';
import { Link } from 'react-router-dom';
import LandingNav from './LandingNav';
import { motion } from 'framer-motion';
import {
  Stethoscope,
  ShieldCheck,
  ArrowRight,
  Database,
  FileText,
  Search,
  Upload,
  Bot,
  User,
  Sparkles,
} from 'lucide-react';
import './LandingPage.css';
import demoPreview from '../assets/demo-preview.png';

interface LandingPageProps {
  onStart: () => void;
}

const FEATURES = [
  {
    icon: Database,
    title: 'Local vector store',
    desc: 'Documents are indexed in ChromaDB on your machine. Your data never leaves your environment.',
  },
  {
    icon: Search,
    title: 'Hybrid retrieval',
    desc: 'Dense embeddings plus BM25 keyword search surface the most relevant medical passages.',
  },
  {
    icon: ShieldCheck,
    title: 'Trust scoring',
    desc: 'Every answer is evaluated for faithfulness and relevance so you can gauge reliability.',
  },
  {
    icon: Upload,
    title: 'PDF ingestion',
    desc: 'Upload medical PDFs, chunk them automatically, and query them in natural language.',
  },
];

const STEPS = [
  { num: '1', title: 'Upload PDFs', desc: 'Add clinical guidelines, drug manuals, or study documents.' },
  { num: '2', title: 'Index & embed', desc: 'The system chunks and embeds your files into a searchable index.' },
  { num: '3', title: 'Ask questions', desc: 'Chat with evidence-based answers backed by source citations.' },
];

const PROMPTS = [
  'What is aplastic anemia?',
  'What are the symptoms of Influenza?',
  'List rare blood disorders',
];

const LandingPage: React.FC<LandingPageProps> = ({ onStart }) => {
  return (
    <div className="landing-page">
      <LandingNav />

      <header className="hero">
        <div className="hero-inner">
          <motion.div
            className="hero-copy"
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="hero-eyebrow">
              <Sparkles size={14} />
              Medical RAG assistant
            </div>
            <h1>
              Query your medical documents with{' '}
              <span className="hero-highlight">cited answers</span>
            </h1>
            <p>
              Upload PDFs, ask clinical questions, and get structured responses
              grounded in your own documents — with sources and trust scores.
            </p>
            <div className="hero-cta">
              <button className="btn-primary" onClick={onStart}>
                Get started <ArrowRight size={18} />
              </button>
              <Link to="/signup" className="btn-ghost">Create free account</Link>
            </div>
            <ul className="hero-points">
              <li><FileText size={15} /> PDF upload & indexing</li>
              <li><Bot size={15} /> Streaming chat responses</li>
              <li><ShieldCheck size={15} /> Faithfulness scoring</li>
            </ul>
          </motion.div>

          <motion.div
            className="hero-preview"
            initial={{ opacity: 0, y: 32 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.15 }}
          >
            <div className="preview-window">
              <div className="preview-chrome">
                <span className="chrome-dot red" />
                <span className="chrome-dot yellow" />
                <span className="chrome-dot green" />
                <span className="chrome-title">MedAssist</span>
              </div>
              <div className="preview-body">
                <div className="preview-sidebar">
                  <div className="preview-sidebar-title">Conversations</div>
                  <div className="preview-conv active">What is aplastic anemia?</div>
                  <div className="preview-conv">Influenza symptoms</div>
                  <div className="preview-upload">
                    <Upload size={14} />
                    Upload PDF
                  </div>
                </div>
                <div className="preview-chat">
                  <div className="preview-msg user">
                    <User size={12} />
                    <span>What is aplastic anemia?</span>
                  </div>
                  <div className="preview-msg assistant">
                    <Bot size={12} />
                    <div className="preview-answer">
                      <strong>Definition</strong>
                      <p>Aplastic anemia is a condition characterized by panhypoplasia of the marrow...</p>
                      <div className="preview-trust">
                        <ShieldCheck size={12} />
                        Trust score 87%
                      </div>
                    </div>
                  </div>
                  <div className="preview-input">
                    <span>Ask about your documents...</span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </header>

      <section className="demo-section">
        <div className="section-wrap">
          <div className="section-label">Live preview</div>
          <h2>Built for clinical Q&amp;A</h2>
          <p className="section-desc">
            Real interface — structured markdown answers, expandable sources, and response metrics.
          </p>

          <motion.div
            className="demo-browser"
            initial={{ opacity: 0, y: 32 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <div className="demo-browser-bar">
              <span className="chrome-dot red" />
              <span className="chrome-dot yellow" />
              <span className="chrome-dot green" />
              <span className="chrome-url">medassist / chat</span>
            </div>
            <button className="demo-screenshot-btn" onClick={onStart} aria-label="Open MedAssist">
              <img
                src={demoPreview}
                alt="MedAssist answering a question about aplastic anemia"
                className="demo-screenshot"
              />
            </button>
          </motion.div>

          <div className="demo-prompts">
            <span>Try asking:</span>
            {PROMPTS.map((prompt) => (
              <button key={prompt} className="prompt-chip" onClick={onStart}>
                {prompt}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="features-section">
        <div className="section-wrap">
          <div className="section-label">Capabilities</div>
          <h2>Everything you need for document Q&amp;A</h2>
          <div className="features-grid">
            {FEATURES.map(({ icon: Icon, title, desc }) => (
              <div key={title} className="feature-card">
                <div className="feature-icon">
                  <Icon size={22} />
                </div>
                <h3>{title}</h3>
                <p>{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="steps-section">
        <div className="section-wrap">
          <div className="section-label">How it works</div>
          <h2>Three steps to get answers</h2>
          <div className="steps-grid">
            {STEPS.map(({ num, title, desc }) => (
              <div key={num} className="step-card">
                <div className="step-num">{num}</div>
                <h3>{title}</h3>
                <p>{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="cta-section">
        <div className="cta-card">
          <h2>Ready to consult your documents?</h2>
          <p>Create an account and start asking questions in minutes.</p>
          <div className="cta-actions">
            <button className="btn-primary" onClick={onStart}>Launch MedAssist</button>
            <Link to="/login" className="btn-ghost-light">I already have an account</Link>
          </div>
        </div>
      </section>

      <footer className="landing-footer">
        <div className="footer-inner">
          <div className="footer-brand">
            <Stethoscope size={20} />
            <span>MedAssist</span>
          </div>
          <p>© 2026 Medical Assistant RAG</p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;

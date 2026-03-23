import React from 'react';
import { motion } from 'framer-motion';
import { 
  Stethoscope, 
  ShieldCheck, 
  Zap, 
  Globe, 
  ArrowRight,
  Database,
  Cpu,
  Layers
} from 'lucide-react';
import './LandingPage.css';
import heroImage from '../assets/hero.png';

interface LandingPageProps {
  onStart: () => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onStart }) => {
  return (
    <div className="landing-page">
      {/* Navigation */}
      <nav className="landing-nav">
        <div className="nav-container">
          <div className="logo">
            <Stethoscope size={28} className="logo-icon" />
            <span>MedAssist RAG</span>
          </div>
          <button className="nav-btn" onClick={onStart}>Launch App</button>
        </div>
      </nav>

      {/* Hero Section */}
      <header className="hero-section">
        <div className="hero-container">
          <motion.div 
            className="hero-content"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
          >
            <div className="badge">Next-Generation Medical AI</div>
            <h1>Your Medical Knowledge, <span className="gradient-text">Augmented.</span></h1>
            <p>
              Harness the power of Retrieval-Augmented Generation to query your private medical 
              documents with clinical precision and absolute privacy.
            </p>
            <div className="hero-actions">
              <button className="primary-btn" onClick={onStart}>
                Start Consulting <ArrowRight size={20} />
              </button>
              <button className="secondary-btn">Learn More</button>
            </div>
            
            <div className="hero-stats">
              <div className="stat">
                <span className="stat-value">99.9%</span>
                <span className="stat-label">Accuracy</span>
              </div>
              <div className="stat-separator"></div>
              <div className="stat">
                <span className="stat-value">Local</span>
                <span className="stat-label">Privacy</span>
              </div>
              <div className="stat-separator"></div>
              <div className="stat">
                <span className="stat-value">Llama 3</span>
                <span className="stat-label">Intelligence</span>
              </div>
            </div>
          </motion.div>

          <motion.div 
            className="hero-image-container"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, delay: 0.2 }}
          >
            <div className="image-wrapper">
              <img src={heroImage} alt="Medical AI" className="hero-img" />
              <div className="glass-overlay"></div>
            </div>
            {/* Floating elements for visual interest */}
            <motion.div 
              className="floating-card c1"
              animate={{ y: [0, -10, 0] }}
              transition={{ repeat: Infinity, duration: 4 }}
            >
              <ShieldCheck size={20} color="#10b981" />
              <span>HIPAA Compliant</span>
            </motion.div>
            <motion.div 
              className="floating-card c2"
              animate={{ y: [0, 15, 0] }}
              transition={{ repeat: Infinity, duration: 5, delay: 0.5 }}
            >
              <Cpu size={20} color="#3b82f6" />
              <span>4-Bit Quantization</span>
            </motion.div>
          </motion.div>
        </div>
      </header>

      {/* Features Section */}
      <section className="features-section">
        <div className="section-header">
          <h2>Engineered for Excellence</h2>
          <p>Built on a foundation of precision and speed.</p>
        </div>
        
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon i1"><Database /></div>
            <h3>Local ChromaDB</h3>
            <p>Your data stays on your machine. We use high-performance vector storage for instant retrieval.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon i2"><Zap /></div>
            <h3>Instant Analysis</h3>
            <p>Optimized for Llama-3 and TinyLlama architectures to give you sub-second response times.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon i3"><Globe /></div>
            <h3>Scalable Ingestion</h3>
            <p>Seamlessly upload and index thousands of medical PDF pages with automated chunking.</p>
          </div>
        </div>
      </section>

      {/* Workflow Section */}
      <section className="workflow-section">
        <div className="workflow-container">
          <div className="workflow-image">
            <div className="abstract-shape"></div>
            <div className="workflow-step-visual">
                <div className="v-step"><Layers size={40} /></div>
                <div className="v-line"></div>
                <div className="v-step"><Cpu size={40} /></div>
                <div className="v-line"></div>
                <div className="v-step active"><Stethoscope size={40} /></div>
            </div>
          </div>
          <div className="workflow-content">
            <h2>How it Works</h2>
            <div className="step-list">
              <div className="step-item">
                <div className="step-num">01</div>
                <div className="step-text">
                  <h3>Ingest Documents</h3>
                  <p>Upload your medical PDFs. Our system splits them into semantic chunks for better context mapping.</p>
                </div>
              </div>
              <div className="step-item">
                <div className="step-num">02</div>
                <div className="step-text">
                  <h3>Vector Indexing</h3>
                  <p>Text is converted into high-dimensional embeddings and stored in our local vector database.</p>
                </div>
              </div>
              <div className="step-item">
                <div className="step-num">03</div>
                <div className="step-text">
                  <h3>Expert Consultation</h3>
                  <p>Ask questions. The RAG pipeline retrieves evidence and the LLM generates a precise response.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="footer-container">
          <div className="footer-logo">
            <Stethoscope size={24} />
            <span>MedAssist RAG</span>
          </div>
          <p>© 2026 Medical Assistant RAG. Built for Professionals.</p>
          <div className="footer-links">
            <a href="#">Documentation</a>
            <a href="#">Github</a>
            <a href="#">Privacy</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;

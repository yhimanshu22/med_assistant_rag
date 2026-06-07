import React from 'react';
import {
  Heart,
  Search,
  Shield,
  Sparkles,
  Target,
} from 'lucide-react';
import MarketingLayout from './MarketingLayout';

const VALUES = [
  {
    icon: Target,
    title: 'Evidence over speculation',
    desc: 'Answers are retrieved from your documents first, then synthesized — reducing hallucination risk.',
  },
  {
    icon: Shield,
    title: 'Local-first privacy',
    desc: 'Vector indexes and PDFs stay on infrastructure you control. Built for sensitive clinical reference material.',
  },
  {
    icon: Search,
    title: 'Hybrid retrieval',
    desc: 'Dense embeddings plus BM25 keyword search combine semantic understanding with exact term matching.',
  },
  {
    icon: Sparkles,
    title: 'Transparent quality',
    desc: 'Source citations and optional Ragas faithfulness scoring let users judge answer reliability.',
  },
];

const AboutPage: React.FC = () => {
  return (
    <MarketingLayout>
      <section className="marketing-hero">
        <div className="marketing-eyebrow">
          <Heart size={14} />
          About MedAssist
        </div>
        <h1>Medical document Q&amp;A, built for trust</h1>
        <p>
          MedAssist is a retrieval-augmented generation (RAG) assistant that helps
          clinicians and researchers query medical PDFs with cited, structured answers.
        </p>
      </section>

      <div className="marketing-content">
        <div className="marketing-prose">
          <p>
            Clinical teams spend hours searching lengthy guidelines, drug manuals, and
            reference texts. MedAssist indexes that content locally, chunks it for search,
            and uses a language model to produce readable answers — always pointing back
            to the passages that support each claim.
          </p>
          <p>
            The product started as a practical RAG stack: ChromaDB for vectors, hybrid
            BM25 + embedding retrieval, TinyLlama for generation on CPU-friendly hardware,
            and optional Ragas evaluation for response quality. It is designed to evolve
            toward enterprise deployment, partner integrations, and a public API.
          </p>
        </div>

        <div className="marketing-grid">
          {VALUES.map(({ icon: Icon, title, desc }) => (
            <div key={title} className="marketing-card">
              <div className="marketing-card-icon">
                <Icon size={20} />
              </div>
              <h3>{title}</h3>
              <p>{desc}</p>
            </div>
          ))}
        </div>

        <div className="marketing-cta">
          <h2>Not a substitute for clinical judgment</h2>
          <p>
            MedAssist supports reference lookup and education. Always verify critical
            decisions against authoritative sources and professional guidelines.
          </p>
        </div>
      </div>
    </MarketingLayout>
  );
};

export default AboutPage;

import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  BookOpen,
  FileSearch,
  ShieldCheck,
  Stethoscope,
  Upload,
  ArrowRight,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import MarketingLayout from './MarketingLayout';

const BENEFITS = [
  {
    icon: FileSearch,
    title: 'Answers with citations',
    desc: 'Every response links back to passages in your uploaded guidelines, formularies, and protocols.',
  },
  {
    icon: ShieldCheck,
    title: 'Trust scoring',
    desc: 'Faithfulness and relevance metrics help you gauge how well an answer matches the source material.',
  },
  {
    icon: Upload,
    title: 'Your documents, your index',
    desc: 'Upload PDFs and query them locally. Clinical content stays in your environment.',
  },
  {
    icon: BookOpen,
    title: 'Structured responses',
    desc: 'Definitions, differentials, and treatment notes formatted for quick scanning at the point of care.',
  },
];

const CliniciansPage: React.FC = () => {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();

  const handleStart = () => {
    navigate(isAuthenticated ? '/chat' : '/signup');
  };

  return (
    <MarketingLayout>
      <section className="marketing-hero">
        <div className="marketing-eyebrow">
          <Stethoscope size={14} />
          For Clinicians
        </div>
        <h1>Clinical answers grounded in your documents</h1>
        <p>
          MedAssist helps physicians, nurses, and care teams query institutional
          medical content with cited, evidence-based responses — not generic web summaries.
        </p>
      </section>

      <div className="marketing-content">
        <div className="marketing-grid">
          {BENEFITS.map(({ icon: Icon, title, desc }) => (
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
          <h2>Start querying your clinical library</h2>
          <p>Upload a PDF and ask your first question in minutes.</p>
          <button type="button" className="btn-primary" onClick={handleStart}>
            Get started <ArrowRight size={18} />
          </button>
        </div>
      </div>
    </MarketingLayout>
  );
};

export default CliniciansPage;

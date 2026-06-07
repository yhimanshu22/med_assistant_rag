import React from 'react';
import { Stethoscope } from 'lucide-react';
import LandingNav from './LandingNav';
import './LandingPage.css';
import './MarketingPage.css';

interface MarketingLayoutProps {
  children: React.ReactNode;
}

const MarketingLayout: React.FC<MarketingLayoutProps> = ({ children }) => {
  return (
    <div className="landing-page">
      <LandingNav />
      <main className="marketing-main">{children}</main>
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

export default MarketingLayout;

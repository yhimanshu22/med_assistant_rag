import React from 'react';
import { Link } from 'react-router-dom';
import { Stethoscope } from 'lucide-react';
import { ACCOUNT_LINKS, PRODUCT_LINKS } from './landingLinks';
import './LandingPage.css';

const LandingFooter: React.FC = () => {
  return (
    <footer className="landing-footer">
      <div className="footer-inner">
        <div className="footer-grid">
          <div className="footer-brand-col">
            <Link to="/" className="footer-brand">
              <Stethoscope size={22} />
              <span>MedAssist</span>
            </Link>
            <p className="footer-tagline">
              Evidence-based answers from your medical documents — with citations and trust scores.
            </p>
          </div>

          <div className="footer-links-col">
            <h4>Product</h4>
            <ul>
              {PRODUCT_LINKS.map(({ label, path }) => (
                <li key={path}>
                  <Link to={path}>{label}</Link>
                </li>
              ))}
            </ul>
          </div>

          <div className="footer-links-col">
            <h4>Account</h4>
            <ul>
              {ACCOUNT_LINKS.map(({ label, path }) => (
                <li key={path}>
                  <Link to={path}>{label}</Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="footer-bottom">
          <p>© 2026 Medical Assistant RAG</p>
          <p className="footer-disclaimer">
            Not a substitute for professional medical advice.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default LandingFooter;

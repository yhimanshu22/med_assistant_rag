import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Stethoscope } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { PRODUCT_LINKS } from './landingLinks';
import './LandingPage.css';

const LandingNav: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { isAuthenticated } = useAuth();

  const handleStart = () => {
    navigate(isAuthenticated ? '/chat' : '/login');
  };

  return (
    <nav className="landing-nav">
      <div className="nav-container">
        <Link to="/" className="logo">
          <Stethoscope size={26} strokeWidth={2.5} />
          <span>MedAssist</span>
        </Link>

        <div className="nav-links">
          {PRODUCT_LINKS.map(({ label, path }) => (
            <Link
              key={path}
              to={path}
              className={`nav-link-item${location.pathname === path ? ' active' : ''}`}
            >
              {label}
            </Link>
          ))}
        </div>

        <div className="nav-actions">
          <Link to="/login" className="nav-link">Login</Link>
          <Link to="/signup" className="nav-link-primary">Sign up</Link>
          <button type="button" className="nav-btn" onClick={handleStart}>Open app</button>
        </div>
      </div>
    </nav>
  );
};

export default LandingNav;

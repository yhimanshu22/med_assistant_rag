import React from 'react';
import LandingNav from './LandingNav';
import LandingFooter from './LandingFooter';
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
      <LandingFooter />
    </div>
  );
};

export default MarketingLayout;

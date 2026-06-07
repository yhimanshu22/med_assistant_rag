import React from 'react';
import {
  Building2,
  Handshake,
  Layers,
  Lock,
  Server,
} from 'lucide-react';
import MarketingLayout from './MarketingLayout';

const OFFERINGS = [
  {
    icon: Building2,
    title: 'Health systems & hospitals',
    desc: 'Deploy MedAssist alongside internal knowledge bases so clinicians can search institutional guidelines at scale.',
  },
  {
    icon: Layers,
    title: 'White-label & embedded UX',
    desc: 'Integrate the chat experience into portals, intranets, or specialty workflows with your branding.',
  },
  {
    icon: Server,
    title: 'On-premise deployment',
    desc: 'Run vector search and inference within your network boundary for strict data residency requirements.',
  },
  {
    icon: Lock,
    title: 'Privacy-first architecture',
    desc: 'Documents are indexed locally. No patient data is required — ideal for reference and educational content.',
  },
];

const PartnersPage: React.FC = () => {
  return (
    <MarketingLayout>
      <section className="marketing-hero">
        <div className="marketing-eyebrow">
          <Handshake size={14} />
          For Partners
        </div>
        <h1>Bring MedAssist to your organization</h1>
        <p>
          We work with digital health vendors, hospital IT teams, and medical
          publishers to deliver document-grounded Q&amp;A inside existing clinical workflows.
        </p>
      </section>

      <div className="marketing-content">
        <div className="marketing-grid">
          {OFFERINGS.map(({ icon: Icon, title, desc }) => (
            <div key={title} className="marketing-card">
              <div className="marketing-card-icon">
                <Icon size={20} />
              </div>
              <h3>{title}</h3>
              <p>{desc}</p>
            </div>
          ))}
        </div>

        <h2 className="marketing-section-title">Partnership models</h2>
        <div className="marketing-prose">
          <p>
            Whether you need a pilot for a single department or a system-wide rollout,
            we can tailor ingestion pipelines, access controls, and hosting to your requirements.
          </p>
          <p>
            Interested in co-developing integrations or distributing MedAssist to your customers?
            Reach out to discuss licensing, support tiers, and implementation timelines.
          </p>
        </div>

        <div className="marketing-cta">
          <h2>Let&apos;s explore a partnership</h2>
          <p>Contact us to schedule a demo or discuss deployment options.</p>
          <a href="mailto:partners@medassist.app" className="btn-primary">
            partners@medassist.app
          </a>
        </div>
      </div>
    </MarketingLayout>
  );
};

export default PartnersPage;

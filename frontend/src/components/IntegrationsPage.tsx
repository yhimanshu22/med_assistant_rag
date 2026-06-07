import React from 'react';
import {
  Code2,
  Plug,
  Webhook,
  Clock,
  CheckCircle2,
} from 'lucide-react';
import MarketingLayout from './MarketingLayout';

const PLANNED = [
  'REST API for document ingestion and semantic search',
  'Streaming query endpoint compatible with chat UIs',
  'Webhook callbacks for ingestion job completion',
  'API keys and rate limiting for production use',
  'OpenAPI specification and SDK examples',
];

const IntegrationsPage: React.FC = () => {
  return (
    <MarketingLayout>
      <section className="marketing-hero">
        <div className="marketing-eyebrow coming-soon">
          <Clock size={14} />
          Coming soon
        </div>
        <h1>Integrations &amp; API</h1>
        <p>
          A public API is in development so you can embed MedAssist retrieval and
          generation into EHRs, portals, and custom clinical tools.
        </p>
      </section>

      <div className="marketing-content">
        <div className="marketing-grid">
          <div className="marketing-card">
            <div className="marketing-card-icon">
              <Code2 size={20} />
            </div>
            <h3>Developer API</h3>
            <p>
              Programmatic access to upload PDFs, run hybrid retrieval, and stream
              answers with source metadata — currently available only via the web app.
            </p>
          </div>
          <div className="marketing-card">
            <div className="marketing-card-icon">
              <Plug size={20} />
            </div>
            <h3>EHR &amp; portal connectors</h3>
            <p>
              Planned connectors for common health IT stacks so clinicians can query
              institutional content without leaving their workflow.
            </p>
          </div>
          <div className="marketing-card">
            <div className="marketing-card-icon">
              <Webhook size={20} />
            </div>
            <h3>Event-driven pipelines</h3>
            <p>
              Notify downstream systems when new documents are indexed or when
              evaluation scores fall below configured thresholds.
            </p>
          </div>
          <div className="marketing-card">
            <div className="marketing-card-icon">
              <CheckCircle2 size={20} />
            </div>
            <h3>What works today</h3>
            <p>
              The backend exposes health, metrics, and authenticated chat endpoints
              used by this app. A documented public API layer is the next milestone.
            </p>
          </div>
        </div>

        <h2 className="marketing-section-title">Planned API surface</h2>
        <ul className="marketing-list">
          {PLANNED.map((item) => (
            <li key={item}>
              <CheckCircle2 size={16} />
              {item}
            </li>
          ))}
        </ul>

        <div className="api-preview">
          <pre>
            <span className="comment"># Preview — not yet publicly available</span>{'\n'}
            <span className="method">POST</span> <span className="path">/api/v1/query</span>{'\n'}
            {'{'}{'\n'}
            {'  '}&quot;question&quot;: &quot;What are influenza symptoms?&quot;,{'\n'}
            {'  '}&quot;stream&quot;: true{'\n'}
            {'}'}
          </pre>
        </div>

        <div className="marketing-cta">
          <h2>Want early API access?</h2>
          <p>Join the waitlist and we&apos;ll notify you when the integration API ships.</p>
          <a href="mailto:api@medassist.app" className="btn-primary">
            api@medassist.app
          </a>
        </div>
      </div>
    </MarketingLayout>
  );
};

export default IntegrationsPage;

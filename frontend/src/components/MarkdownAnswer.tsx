import type { Components } from 'react-markdown';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const markdownComponents: Components = {
  em: ({ children }) => <em className="md-italic">{children}</em>,
  strong: ({ children }) => <strong className="md-bold">{children}</strong>,
  p: ({ children }) => <p className="md-paragraph">{children}</p>,
};

type MarkdownAnswerProps = {
  content: string;
};

export default function MarkdownAnswer({ content }: MarkdownAnswerProps) {
  const parts = content.split(/(==[^=\n]+==)/g);

  return (
    <div className="markdown-content">
      {parts.map((part, index) => {
        const highlight = part.match(/^==([^=\n]+)==$/);
        if (highlight) {
          return (
            <mark key={index} className="md-highlight">
              {highlight[1]}
            </mark>
          );
        }
        if (!part) return null;
        return (
          <ReactMarkdown
            key={index}
            remarkPlugins={[remarkGfm]}
            components={markdownComponents}
          >
            {part}
          </ReactMarkdown>
        );
      })}
    </div>
  );
}

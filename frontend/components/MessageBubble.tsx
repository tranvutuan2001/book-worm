import { Message } from '@/lib/schemas';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface MessageBubbleProps {
  message: Message;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[70%] rounded-2xl px-4 py-3 ${
          isUser
            ? 'bg-purple-600 text-white shadow-lg shadow-purple-300/50'
            : 'bg-white/90 backdrop-blur-sm text-gray-800 border border-purple-200 shadow-sm'
        }`}
      >
        <div className={`text-sm leading-relaxed prose prose-sm max-w-none prose-headings:mt-3 prose-headings:mb-2 prose-p:my-2 prose-pre:my-2 prose-ul:my-2 prose-ol:my-2 ${isUser ? 'text-white prose-invert' : 'text-gray-800'}`}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              // Style code blocks
              code: ({ className, children, ...props }) => {
                const isInline = !className;
                return isInline ? (
                  <code
                    className={`${isUser ? 'bg-purple-700' : 'bg-purple-100'} px-1.5 py-0.5 rounded text-xs`}
                    {...props}
                  >
                    {children}
                  </code>
                ) : (
                  <code
                    className={`block ${isUser ? 'bg-purple-700' : 'bg-purple-50'} p-2 rounded-lg text-xs overflow-x-auto`}
                    {...props}
                  >
                    {children}
                  </code>
                );
              },
              // Style links
              a: ({ children, ...props }) => (
                <a
                  className={`${isUser ? 'text-purple-200 hover:text-purple-100' : 'text-purple-600 hover:text-purple-700'} underline`}
                  target="_blank"
                  rel="noopener noreferrer"
                  {...props}
                >
                  {children}
                </a>
              ),
              // Style lists
              ul: ({ children, ...props }) => (
                <ul className="list-disc list-inside space-y-1" {...props}>
                  {children}
                </ul>
              ),
              ol: ({ children, ...props }) => (
                <ol className="list-decimal list-inside space-y-1" {...props}>
                  {children}
                </ol>
              ),
              // Style headings
              h1: ({ children, ...props }) => (
                <h1 className="text-lg font-bold" {...props}>{children}</h1>
              ),
              h2: ({ children, ...props }) => (
                <h2 className="text-base font-bold" {...props}>{children}</h2>
              ),
              h3: ({ children, ...props }) => (
                <h3 className="text-sm font-bold" {...props}>{children}</h3>
              ),
              // Style blockquotes
              blockquote: ({ children, ...props }) => (
                <blockquote
                  className={`border-l-4 ${isUser ? 'border-purple-400' : 'border-purple-300'} pl-3 italic my-2`}
                  {...props}
                >
                  {children}
                </blockquote>
              ),
              // Style tables
              table: ({ children, ...props }) => (
                <div className="overflow-x-auto my-2">
                  <table className="min-w-full border-collapse" {...props}>
                    {children}
                  </table>
                </div>
              ),
              th: ({ children, ...props }) => (
                <th
                  className={`border ${isUser ? 'border-purple-500 bg-purple-700' : 'border-purple-300 bg-purple-100'} px-3 py-1 text-left font-semibold`}
                  {...props}
                >
                  {children}
                </th>
              ),
              td: ({ children, ...props }) => (
                <td
                  className={`border ${isUser ? 'border-purple-500' : 'border-purple-300'} px-3 py-1`}
                  {...props}
                >
                  {children}
                </td>
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>
        <span className="text-xs opacity-70 mt-1 block">
          {new Date(message.timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
      </div>
    </div>
  );
}

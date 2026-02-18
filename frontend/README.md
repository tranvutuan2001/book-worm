# AI Chat Application

A modern, minimalist chat application built with Next.js 14, TypeScript, Tailwind CSS, and Zod for schema validation. This app provides a clean interface for chatting with LLM services.

## Features

- 🎨 **Modern UI**: Clean, simple interface with white and light purple theme
- 💬 **Real-time Chat**: Interactive chat interface with message history
- ✅ **Type-Safe**: Built with TypeScript and Zod for runtime validation
- 🎯 **Responsive Design**: Works seamlessly on desktop and mobile devices
- ⚡ **Next.js 14**: Built on the latest Next.js with App Router
- 🎭 **Tailwind CSS**: Utility-first styling for rapid development

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Validation**: Zod
- **Package Manager**: npm

## Getting Started

First, run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Project Structure

```
├── app/
│   ├── api/
│   │   └── chat/
│   │       └── route.ts          # API endpoint for chat (placeholder)
│   ├── globals.css                # Global styles with purple theme
│   ├── layout.tsx                 # Root layout
│   └── page.tsx                   # Main chat page
├── components/
│   ├── ChatHeader.tsx             # Header component
│   ├── ChatInput.tsx              # Message input component
│   └── MessageBubble.tsx          # Message display component
├── lib/
│   ├── api.ts                     # API client functions
│   └── schemas.ts                 # Zod schemas for validation
└── package.json
```

## Backend Integration

The application includes placeholder API endpoints that you can connect to your LLM backend:

### API Endpoint: `/api/chat`

Located in `app/api/chat/route.ts`, this endpoint currently returns an echo response. To connect to your backend:

1. Replace the placeholder logic in `app/api/chat/route.ts`
2. Update the API call to point to your LLM service
3. Modify the response handling as needed

Example integration:

```typescript
// In app/api/chat/route.ts
const response = await fetch('YOUR_BACKEND_URL/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${process.env.API_KEY}`,
  },
  body: JSON.stringify({
    message: validatedData.message,
    conversationId: validatedData.conversationId,
  }),
});

const data = await response.json();
```

## Customization

### Theme Colors

The color scheme is defined in `app/globals.css`. You can customize the purple shades by modifying the CSS variables.

### Components

All UI components are in the `components/` directory and can be easily customized:

- `ChatHeader.tsx` - Modify the header text and styling
- `ChatInput.tsx` - Adjust input behavior and appearance
- `MessageBubble.tsx` - Change message bubble styling

## Validation Schemas

The app uses Zod for type-safe validation. Schemas are defined in `lib/schemas.ts`:

- `MessageSchema` - Validates message objects
- `ChatRequestSchema` - Validates API requests
- `ChatResponseSchema` - Validates API responses

## Build for Production

```bash
npm run build
npm start
```

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme).

Check out the [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.


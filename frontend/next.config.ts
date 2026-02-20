import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  output: 'standalone',
  // Proxy /api/* and /llm/* to the respective backend services.
  // In development these default to localhost; in production the values
  // come from environment variables injected by the Kubernetes deployment.
  // This keeps the browser code simple (relative paths only) and avoids
  // baking URLs into the Next.js browser bundle at build time.
  async rewrites() {
    const backendUrl = process.env.NODE_ENV === 'production'? 'backend.book-worm.svc.cluster.local' : 'http://localhost:8000';
    const llmServerUrl = process.env.NODE_ENV === 'production'? 'llm-server.book-worm.svc.cluster.local' : 'http://localhost:8001';
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/:path*`,
      },
      {
        source: '/llm/:path*',
        destination: `${llmServerUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;

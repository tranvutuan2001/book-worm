# Use BuildKit features when available
# Multi-stage Dockerfile for building and running a Next.js app in production
ARG NODE_VERSION=24.11.1

########################
# Dependencies stage
########################
FROM node:${NODE_VERSION}-alpine AS deps
WORKDIR /app
COPY package*.json ./
# Install all deps (dev + prod) needed for build
RUN npm ci --prefer-offline --no-audit

########################
# Builder stage
########################
FROM node:${NODE_VERSION}-alpine AS builder
WORKDIR /app
# Copy installed deps from previous stage to leverage caching
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NODE_ENV=production
# Build the Next.js app
RUN npm run build
# Remove devDependencies to keep final image small
RUN npm prune --production

########################
# Runtime stage
########################
FROM node:${NODE_VERSION}-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV HOSTNAME="0.0.0.0"
ENV PORT=3000

# Create a non-root user for security
RUN addgroup -S app && adduser -S app -G app

# Copy only the standalone artifacts from builder (no node_modules or npm needed)
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# Ensure correct permissions
RUN chown -R app:app /app
USER app

EXPOSE 3000

# Simple healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 CMD wget -qO- http://localhost:3000 || exit 1

# Run the standalone Next.js server
CMD ["node", "server.js"]

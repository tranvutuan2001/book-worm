# Use BuildKit features when available
# Multi-stage Dockerfile for building and running a Next.js app in production
ARG NODE_VERSION=24.11.1

########################
# Dependencies stage
########################
FROM node:${NODE_VERSION}-alpine AS deps
RUN apk add --no-cache python3 make g++ git
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
# Ensure `next.config.js` exists so runtime COPY won't fail if optional
RUN if [ ! -f ./next.config.js ]; then printf 'module.exports = {};\n' > next.config.js; fi
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
ENV PORT=3000

# Create a non-root user for security
RUN addgroup -S app && adduser -S app -G app

# Copy only the production artifacts from builder
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/next.config.js ./

# Ensure correct permissions
RUN chown -R app:app /app
USER app

EXPOSE 3000

# Simple healthcheck; requires wget (available in alpine via busybox's wget)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 CMD wget -qO- http://localhost:3000 || exit 1

# Run the production Next.js server; expects `start` script in package.json
CMD ["npm","run","start"]

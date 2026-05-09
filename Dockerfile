# Stage 1: Build Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /build

# Copy frontend package files
COPY src/lida/web/frontend/package.json src/lida/web/frontend/package-lock.json ./

# Install dependencies
RUN npm ci

# Copy frontend source code
COPY src/lida/web/frontend ./

# Build the frontend
RUN npm run build


# Stage 2: Serve Frontend with Nginx
FROM nginx:alpine AS frontend
COPY --from=frontend-builder /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]


# Stage 3: Build Backend
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS backend

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock pyproject.toml /app/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY . /app

# Note: We do NOT copy frontend dist here anymore.
# The backend operates purely as an API server.

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

ENTRYPOINT ["lida", "ui", "--host", "0.0.0.0", "--port", "8080"]
FROM node:22-alpine AS frontend
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim AS application
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VOYAGER_HOST=0.0.0.0 \
    VOYAGER_PORT=8000 \
    VOYAGER_FRONTEND_DIR=/app/web/dist \
    VOYAGER_REPLAY_ROOTS=/app/benchmarks/replays/stage6_curated_v1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY voyager/ voyager/
COPY benchmarks/replays/stage6_curated_v1/ benchmarks/replays/stage6_curated_v1/
COPY --from=frontend /app/web/dist web/dist
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["voyager-web"]

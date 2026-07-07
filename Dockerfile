# Universal container for the Rakshak agent API.
# Works on any free container host: Hugging Face Spaces (Docker), Render (Docker),
# Fly.io, Google Cloud Run, Koyeb, Railway, etc.
FROM python:3.12-slim

WORKDIR /app
COPY . /app

# No runtime deps (stdlib only) — this stays a no-op but documents intent.
RUN pip install --no-cache-dir -r requirements.txt

# Hosts inject $PORT; default 8000. The server binds 0.0.0.0 automatically.
ENV PORT=8000
EXPOSE 8000

# Set DEEPSEEK_API_KEY as a secret/env var in the host dashboard (never bake it in).
CMD ["python", "server.py"]

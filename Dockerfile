# Hugging Face Spaces (Docker SDK) / Render / Fly / any container host.
#
# One long-lived process, which is what this app needs: the model is loaded
# once and cached, and utils/ood.py attaches a forward hook that must survive
# between requests. A serverless platform would reload PyTorch on every call.

FROM python:3.11-slim

# libgomp1 is required by the CPU torch wheel (OpenMP runtime).
# No libGL/libglib needed because we install opencv-python-headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000. Creating that user here
# means the app can write uploads/ and results/ instead of hitting EACCES.
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Dependencies first: this layer is cached across code edits, so a change to
# app.py does not re-download ~200 MB of torch.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY --chown=appuser:appuser . .

# Writable at runtime, and pre-created so the first request never races.
RUN mkdir -p uploads results && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache \
    TORCH_HOME=/app/.cache/torch \
    PORT=7860 \
    TOMATO_PRELOAD=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

EXPOSE 7860

# Single worker on purpose. Each worker would hold its own copy of the model
# (~100 MB resident), and the free CPU tier has neither the RAM nor the cores
# to serve two. Threads handle concurrent requests; inference is the bottleneck.
# The long timeout covers first-request model load on a cold container.
CMD gunicorn --bind 0.0.0.0:${PORT:-7860} \
             --workers 1 --threads 4 \
             --timeout 180 \
             --access-logfile - \
             app:app

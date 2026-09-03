FROM python:3.10-slim

# System deps for OpenCV and building native extensions (psutil)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 gcc python3-dev \
        ffmpeg tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first to avoid pulling CUDA packages.
# The pure-python deps come from PyPI; only the torch/torchvision wheels come
# from the CPU index (with --no-deps, so PyPI's CUDA-flavoured torch metadata is
# never consulted). Using --index-url for everything would hide PyPI entirely and
# break any dependency that needs a build backend fetched at install time.
RUN pip install --no-cache-dir \
        filelock typing-extensions sympy networkx jinja2 fsspec requests && \
    pip install --no-cache-dir --no-deps \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.1.2 torchvision==0.16.2

# Install remaining dependencies, skipping torch/torchvision (already installed as CPU-only)
# yt-dlp is deliberately unpinned: YouTube changes break older versions, and a
# stale pin silently kills the coastal camera feeds.
RUN pip install --no-cache-dir yt-dlp

COPY requirements.txt .
RUN grep -iv '^torch' requirements.txt > requirements-notorch.txt && \
    pip install --no-cache-dir -r requirements-notorch.txt

# Copy application code and model files
COPY . .

EXPOSE 8001

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]

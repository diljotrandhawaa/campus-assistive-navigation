FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libopus-dev \
    libvpx-dev \
    pkg-config \
    libsrtp2-dev \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /app

# Upgrade pip and build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy package files first for better layer caching
COPY pyproject.toml MANIFEST.in ./
COPY requirements.txt .

# Copy application source
COPY src/ ./src/

# Install the package in editable mode (includes all dependencies)
RUN pip install --no-cache-dir -e .

# Copy scripts
COPY scripts/generate_cert.sh .
RUN chmod +x generate_cert.sh

# Copy images for documentation/assets if needed
COPY images/ ./images/

# Generate self-signed SSL certificates (can be overridden by volume mount)
RUN ./generate_cert.sh

# Expose port
EXPOSE 8090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('https://localhost:8090', timeout=5, context=__import__('ssl')._create_unverified_context())" || exit 1

# Run the server
CMD ["python", "-m", "live_vlm_webui.server", "--host", "0.0.0.0", "--port", "8090", "--ssl-cert", "cert.pem", "--ssl-key", "key.pem"]


#!/bin/bash
# Test script for Mac Docker build
# Run this on Mac to verify Docker container works

set -e

echo "=========================================="
echo "Testing Mac Docker Build"
echo "=========================================="

# Check if running on Mac
if [[ "$(uname)" != "Darwin" ]]; then
    echo "⚠️  WARNING: This script is intended for Mac"
    echo "    Detected: $(uname)"
    echo "    Continuing anyway..."
fi

# Check Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found"
    echo "   Install Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✅ Docker found: $(docker --version)"

# Check architecture
echo "🖥️  Architecture: $(uname -m)"

# Build the image
echo ""
echo "Building Docker image..."
docker build -f Dockerfile.mac -t live-vlm-webui:mac-test .

echo ""
echo "✅ Build successful!"
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Start Ollama on your Mac (natively, for fast inference):"
echo "   ollama serve &"
echo "   ollama pull llama3.2-vision:11b"
echo ""
echo "2. Run the container:"
echo "   docker run -d \\"
echo "     --name live-vlm-webui-mac \\"
echo "     -p 8090:8090 \\"
echo "     -e VLM_API_BASE=http://host.docker.internal:11434/v1 \\"
echo "     -e VLM_MODEL=llama3.2-vision:11b \\"
echo "     live-vlm-webui:mac-test"
echo ""
echo "3. Access at: https://localhost:8090"
echo ""
echo "4. Stop container:"
echo "   docker stop live-vlm-webui-mac"
echo "   docker rm live-vlm-webui-mac"
echo ""
echo "⚠️  REMINDER: Native Python is 10-20x faster than Docker on Mac!"
echo "    See docs/cursor/MAC_SETUP.md for native installation."
echo ""


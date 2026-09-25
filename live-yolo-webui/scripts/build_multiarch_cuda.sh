#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build multi-architecture CUDA Docker images
# Supports: x86_64 PC/DGX and ARM64 DGX Spark/Jetson Thor

set -e

echo "=========================================="
echo "Multi-Architecture CUDA Build"
echo "=========================================="
echo ""
echo "Target Platforms:"
echo "  - linux/amd64 (x86_64 PC, DGX x86, A100/H100)"
echo "  - linux/arm64 (DGX Spark, Jetson Thor, ARM servers)"
echo ""
echo "Base Image: nvidia/cuda:12.4.1-runtime-ubuntu22.04"
echo "  (Multi-arch manifest, pulls correct version per platform)"
echo ""

# Check if buildx is available
if ! docker buildx version &> /dev/null; then
    echo "❌ docker buildx not found"
    echo "   Install Docker Desktop or Docker Buildx plugin"
    exit 1
fi

# Create/use multiplatform builder
if ! docker buildx ls | grep -q "multiplatform"; then
    echo "📦 Creating multiplatform builder..."
    docker buildx create --name multiplatform --use --driver docker-container
else
    echo "✅ Using existing multiplatform builder"
    docker buildx use multiplatform
fi

# Inspect builder
echo ""
echo "Builder info:"
docker buildx inspect --bootstrap

echo ""
echo "=========================================="
echo "Build Options"
echo "=========================================="
echo ""
echo "1. Build for both platforms (requires registry push)"
echo "2. Build for linux/amd64 only (x86_64)"
echo "3. Build for linux/arm64 only (DGX Spark/Thor)"
echo ""
read -p "Select option (1/2/3): " BUILD_OPTION

case $BUILD_OPTION in
    1)
        # Multi-arch build (must push to registry)
        echo ""
        read -p "Enter registry URL (e.g., ghcr.io/user/live-vlm-webui): " REGISTRY_URL

        if [ -z "$REGISTRY_URL" ]; then
            echo "❌ Registry URL required for multi-arch build"
            exit 1
        fi

        echo ""
        echo "Building multi-arch image..."
        echo "Registry: $REGISTRY_URL:latest"
        echo "Platforms: linux/amd64, linux/arm64"
        echo ""

        docker buildx build \
          --platform linux/amd64,linux/arm64 \
          -t "$REGISTRY_URL:latest" \
          -t "$REGISTRY_URL:$(date +%Y%m%d)" \
          --push \
          .

        echo ""
        echo "=========================================="
        echo "✅ Multi-arch build complete!"
        echo "=========================================="
        echo ""
        echo "Image pushed: $REGISTRY_URL:latest"
        echo "Platforms: linux/amd64, linux/arm64"
        echo ""
        echo "Pull on any platform with:"
        echo "  docker pull $REGISTRY_URL:latest"
        echo ""
        echo "Docker will automatically select the correct architecture!"
        echo ""
        ;;

    2)
        # x86_64 only
        echo ""
        echo "Building for linux/amd64 (x86_64)..."
        echo ""

        docker buildx build \
          --platform linux/amd64 \
          -t live-vlm-webui:x86 \
          --load \
          .

        echo ""
        echo "=========================================="
        echo "✅ x86_64 build complete!"
        echo "=========================================="
        echo "Image: live-vlm-webui:x86"
        echo "Platform: linux/amd64"
        echo ""
        ;;

    3)
        # ARM64 only
        echo ""
        echo "Building for linux/arm64 (DGX Spark/Jetson Thor)..."
        echo ""

        docker buildx build \
          --platform linux/arm64 \
          -t live-vlm-webui:arm64 \
          --load \
          .

        echo ""
        echo "=========================================="
        echo "✅ ARM64 build complete!"
        echo "=========================================="
        echo "Image: live-vlm-webui:arm64"
        echo "Platform: linux/arm64"
        echo ""
        echo "This image will run on:"
        echo "  - NVIDIA DGX Spark"
        echo "  - NVIDIA Jetson Thor"
        echo "  - ARM64 servers with NVIDIA GPUs"
        echo ""
        ;;

    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo "=========================================="
echo "Platform Compatibility"
echo "=========================================="
echo ""
echo "✅ x86_64 (linux/amd64):"
echo "   - Desktop/Workstation PCs"
echo "   - NVIDIA DGX x86 systems"
echo "   - Cloud GPU instances (AWS, GCP, Azure)"
echo ""
echo "✅ ARM64 (linux/arm64):"
echo "   - NVIDIA DGX Spark (SBSA)"
echo "   - NVIDIA Jetson Thor (SBSA)"
echo "   - AWS Graviton + NVIDIA GPU"
echo "   - ARM64 servers with NVIDIA GPUs"
echo ""
echo "Both use the same Dockerfile!"
echo "CUDA base image is multi-arch by default."
echo ""

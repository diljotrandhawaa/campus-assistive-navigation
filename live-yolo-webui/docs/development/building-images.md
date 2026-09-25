# Building Platform-Specific Docker Images

This guide is for developers who want to build and publish their own Docker images.

## Prerequisites

- Docker installed
- NVIDIA Container Toolkit (for GPU support)
- GitHub account (for GHCR publishing)

## Building Locally

### x86_64 PC Image

```bash
docker build -t live-vlm-webui:x86 .
```

### Jetson Orin Image

```bash
docker build -f Dockerfile.jetson-orin -t live-vlm-webui:jetson-orin .
```

### Jetson Thor Image

```bash
docker build -f Dockerfile.jetson-thor -t live-vlm-webui:jetson-thor .
```

## Publishing to GitHub Container Registry

### 1. Authenticate with GHCR

```bash
# Create a Personal Access Token with 'write:packages' scope at:
# https://github.com/settings/tokens

echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin
```

### 2. Tag Your Images

```bash
# x86_64 image
docker tag live-vlm-webui:x86 ghcr.io/YOUR_USERNAME/live-vlm-webui:latest-x86
docker tag live-vlm-webui:x86 ghcr.io/YOUR_USERNAME/live-vlm-webui:v1.0.0-x86

# Jetson Orin image
docker tag live-vlm-webui:jetson-orin ghcr.io/YOUR_USERNAME/live-vlm-webui:latest-jetson-orin
docker tag live-vlm-webui:jetson-orin ghcr.io/YOUR_USERNAME/live-vlm-webui:v1.0.0-jetson-orin

# Jetson Thor image
docker tag live-vlm-webui:jetson-thor ghcr.io/YOUR_USERNAME/live-vlm-webui:latest-jetson-thor
docker tag live-vlm-webui:jetson-thor ghcr.io/YOUR_USERNAME/live-vlm-webui:v1.0.0-jetson-thor
```

### 3. Push to Registry

```bash
# Push x86 images
docker push ghcr.io/YOUR_USERNAME/live-vlm-webui:latest-x86
docker push ghcr.io/YOUR_USERNAME/live-vlm-webui:v1.0.0-x86

# Push Jetson Orin images
docker push ghcr.io/YOUR_USERNAME/live-vlm-webui:latest-jetson-orin
docker push ghcr.io/YOUR_USERNAME/live-vlm-webui:v1.0.0-jetson-orin

# Push Jetson Thor images
docker push ghcr.io/YOUR_USERNAME/live-vlm-webui:latest-jetson-thor
docker push ghcr.io/YOUR_USERNAME/live-vlm-webui:v1.0.0-jetson-thor
```

## Using GitHub Actions (Recommended)

The repository includes `.github/workflows/docker-publish.yml` which automatically builds images on:
- **Push to main**: Builds `latest-*` tags
- **Git tag** (e.g., `v1.0.0`): Builds versioned releases

### Workflow Structure

```yaml
name: Build and Publish Docker Images

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  build-x86:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      - name: Login to GHCR
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:latest-x86
```

### Trigger a Build

```bash
# Method 1: Push to main
git push origin main

# Method 2: Create a version tag
git tag v1.0.0
git push origin v1.0.0
```

## Multi-Architecture Builds

To build ARM64 images on x86_64 machines, use QEMU:

```bash
# Install QEMU
sudo apt-get install qemu-user-static

# Enable multi-architecture builds
docker buildx create --use --name multi-arch-builder
docker buildx inspect --bootstrap

# Build for ARM64 on x86_64
docker buildx build \
  --platform linux/arm64 \
  -f Dockerfile.jetson-orin \
  -t live-vlm-webui:jetson-orin \
  --load \
  .
```

## Image Tagging Strategy

We use the following tagging convention:

| Tag Format | Purpose | Example |
|------------|---------|---------|
| `latest-x86` | Latest x86_64 build | `ghcr.io/nvidia-ai-iot/live-vlm-webui:latest-x86` |
| `latest-jetson-orin` | Latest Jetson Orin build | `ghcr.io/nvidia-ai-iot/live-vlm-webui:latest-jetson-orin` |
| `latest-jetson-thor` | Latest Jetson Thor build | `ghcr.io/nvidia-ai-iot/live-vlm-webui:latest-jetson-thor` |
| `v1.0.0-x86` | Versioned x86_64 release | `ghcr.io/nvidia-ai-iot/live-vlm-webui:v1.0.0-x86` |
| `v1.0.0-jetson-orin` | Versioned Jetson Orin release | `ghcr.io/nvidia-ai-iot/live-vlm-webui:v1.0.0-jetson-orin` |
| `v1.0.0-jetson-thor` | Versioned Jetson Thor release | `ghcr.io/nvidia-ai-iot/live-vlm-webui:v1.0.0-jetson-thor` |

## Testing Your Images

After building, test locally:

```bash
# Test x86 image
docker run --rm --gpus all -p 8090:8090 live-vlm-webui:x86

# Test Jetson Orin image (on Jetson device)
docker run --rm --runtime nvidia --network host live-vlm-webui:jetson-orin

# Test Jetson Thor image (on Thor device)
docker run --rm --gpus all --network host live-vlm-webui:jetson-thor
```

## Troubleshooting

### Build fails with "No space left on device"

```bash
# Clean up Docker
docker system prune -af
docker volume prune -f
```

### ARM64 build is slow

This is expected when building ARM64 on x86_64 (QEMU emulation). Consider:
- Building on native ARM64 hardware (Jetson device)
- Using GitHub Actions (free ARM64 runners)
- Pre-downloading base images: `docker pull nvcr.io/nvidia/l4t-base:r36.2.0`

### Permission denied when pushing to GHCR

Ensure your GitHub token has `write:packages` permission:
1. Go to https://github.com/settings/tokens
2. Create token with `write:packages` scope
3. Re-login: `echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin`


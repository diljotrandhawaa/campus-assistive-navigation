# Docker Compose Setup Details

This guide provides detailed information about the unified docker-compose configuration and the `start_docker_compose.sh` launcher script.

## Overview

The project uses a **unified `docker-compose.yml`** that includes all VLM backends (Ollama, NVIDIA NIM, vLLM, etc.). You can select which backend to use via Docker Compose profiles.

## Quick Start with start_docker_compose.sh

The easiest way to launch the stack is using the auto-detection script:

```bash
# Auto-detect platform, use Ollama
./start_docker_compose.sh

# Explicit backend selection
./start_docker_compose.sh ollama
./start_docker_compose.sh nim

# With model specification (Ollama)
./start_docker_compose.sh ollama llama3.2-vision:11b

# With NIM model variant
export NIM_IMAGE=nvcr.io/nim/nvidia/llama-3.1-nemotron-nano-vl-8b-v1:1
export NIM_MODEL_NAME=nvidia/llama-3.1-nemotron-nano-vl-8b-v1
./start_docker_compose.sh nim
```

**What the script does:**
- ✅ Checks Docker and Docker Compose installation
- ✅ Detects if you have V1 (`docker-compose`) or V2 (`docker compose`)
- ✅ Warns V1 users to upgrade with specific command
- ✅ Auto-detects your platform (PC/DGX Spark/Jetson Orin/Jetson Thor)
- ✅ Selects the appropriate profile
- ✅ Stops existing services gracefully
- ✅ Launches the stack

### Docker Compose Version Detection

The script automatically detects which version you have:

**V2 (docker compose) - Recommended:**
```
✅ Docker Compose: docker compose
```

**V1 (docker-compose) - Legacy:**
```
⚠️  Using legacy docker-compose (V1)
   Recommend upgrading to V2: sudo apt install docker-compose-plugin

✅ Docker Compose: docker-compose
```

**To upgrade from V1 to V2:**
```bash
sudo apt update
sudo apt install docker-compose-plugin

# After install, the script will automatically use V2
```

**Benefits of V2:**
- 2-3x faster startup
- Better error messages
- Native Docker CLI integration
- Active development

---

## docker-compose.yml - Unified Stack

The single `docker-compose.yml` includes all backends with profile-based selection.

### Available Profiles

**Backend-centric naming:** `{backend}` or `{backend}-{platform}`

| Backend | PC/DGX Spark | Jetson Orin | Jetson Thor |
|---------|--------------|-------------|-------------|
| **Ollama** | `ollama` | `ollama-jetson-orin` | `ollama-jetson-thor` |
| **NIM** | `nim` | `nim-jetson-orin` ⚠️ | `nim-jetson-thor` |
| **vLLM** | `vllm` (future) | `vllm-jetson-orin` (future) | `vllm-jetson-thor` (future) |

⚠️ `nim-jetson-orin` only works with multi-arch NIM models (e.g., cosmos-reason1-7b)

### Manual Usage (without script)

```bash
# Ollama - PC (x86_64) or DGX Spark (ARM64)
docker compose --profile ollama up

# Ollama - Jetson Orin
docker compose --profile ollama-jetson-orin up

# Ollama - Jetson Thor
docker compose --profile ollama-jetson-thor up

# After starting, pull a vision model:
docker exec ollama ollama pull llama3.2-vision:11b
```

### Services

**1. Ollama Service**
- Image: `ollama/ollama:latest`
- Port: 11434
- GPU: Enabled via `deploy.resources.reservations`
- Volumes: `~/.ollama` for model storage

**2. Live VLM WebUI Services (3 profiles)**
- **ollama**: For PC (x86_64) / DGX Spark (ARM64)
  - Image: `ghcr.io/nvidia-ai-iot/live-vlm-webui:latest` (multi-arch)
  - GPU: `--gpus all`
- **ollama-jetson-orin**: For Jetson Orin
  - Image: `ghcr.io/nvidia-ai-iot/live-vlm-webui:latest-jetson-orin`
  - Runtime: `nvidia`
  - Extra: `--privileged`, jtop socket mount
- **ollama-jetson-thor**: For Jetson Thor
  - Image: `ghcr.io/nvidia-ai-iot/live-vlm-webui:latest-jetson-thor`
  - GPU: `--gpus all`
  - Extra: `--privileged`, jtop socket mount

### Configuration

All services use `network_mode: host` for easy communication with localhost services.

---

## NVIDIA NIM Backend

The unified `docker-compose.yml` includes NVIDIA NIM services for production-grade vision-language inference.

### Prerequisites

1. **NGC API Key**: Get from https://org.ngc.nvidia.com/setup/api-key
2. **NVIDIA Driver**: 565+ (CUDA 12.9+ support) - or 570+ recommended
3. **GPU VRAM**: 16GB+ recommended for cosmos-reason1-7b
4. **Disk Space**: ~10-15GB for model download
5. **Docker Login**: `docker login nvcr.io` (username: `$oauthtoken`, password: NGC API key)

### Quick Start (with script)

```bash
# Set NGC API Key
export NGC_API_KEY=your-key-here

# Launch with auto-detection
./start_docker_compose.sh nim

# Or with specific NIM model
export NIM_IMAGE=nvcr.io/nim/nvidia/llama-3.1-nemotron-nano-vl-8b-v1:1
export NIM_MODEL_NAME=nvidia/llama-3.1-nemotron-nano-vl-8b-v1
./start_docker_compose.sh nim
```

### Manual Usage (without script)

```bash
# Set NGC API Key
export NGC_API_KEY=your-key-here

# PC (x86_64) or DGX Spark (ARM64)
docker compose --profile nim up

# Jetson Orin (only with multi-arch NIM models)
docker compose --profile nim-jetson-orin up

# Jetson Thor
docker compose --profile nim-jetson-thor up
```

### NIM Services

**1. NIM Service (Flexible Model Selection)**
- Default Image: `nvcr.io/nim/nvidia/cosmos-reason1-7b:1.4.1` (multi-arch: x86_64, ARM64)
- Port: 8000
- GPU: Full GPU access via `runtime: nvidia`
- Volumes: `~/.cache/nim` for model cache
- Shared Memory: 32GB (`shm_size`)
- Health Check: `GET /v1/models` (30s interval, 120s startup grace period)

**Model Selection via Environment Variables:**
```bash
# Default: Cosmos-Reason1-7B (multi-arch)
export NIM_IMAGE=nvcr.io/nim/nvidia/cosmos-reason1-7b:1.4.1
export NIM_MODEL_NAME=nvidia/cosmos-reason1-7b

# Nemotron Nano 8B (x86_64 only)
export NIM_IMAGE=nvcr.io/nim/nvidia/llama-3.1-nemotron-nano-vl-8b-v1:1
export NIM_MODEL_NAME=nvidia/llama-3.1-nemotron-nano-vl-8b-v1

# Llama 3.2 90B Vision (x86_64 only, requires more VRAM)
export NIM_IMAGE=nvcr.io/nim/meta/llama-3.2-90b-vision-instruct:1.1.1
export NIM_MODEL_NAME=meta/llama-3.2-90b-vision-instruct
```

**2. Live VLM WebUI Services**
- Backend-centric profiles: `nim` (PC/DGX Spark), `nim-jetson-orin`, `nim-jetson-thor`
- Pre-configured to connect to NIM on port 8000
- Model name passed via `${NIM_MODEL_NAME}` environment variable
- Explicitly passes VLM settings via `command` args

### First Run

The first time you run NIM:
1. Model downloads (~10-15GB) - takes 10-30 minutes depending on connection
2. NIM initialization - takes 2-5 minutes
3. Health check - waits for NIM to be ready
4. WebUI starts and connects to NIM

**Monitor progress:**
```bash
# Watch NIM logs (container name is dynamic based on NIM_CONTAINER_NAME)
docker logs -f nim-cosmos-reason1-7b  # Default container name

# Check health status
curl http://localhost:8000/v1/models

# Or check compose service status
docker compose ps
```

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `NGC_API_KEY` | ✅ Yes | NGC API key for model access | None |
| `NIM_IMAGE` | No | NIM container image | `nvcr.io/nim/nvidia/cosmos-reason1-7b:1.4.1` |
| `NIM_MODEL_NAME` | No | Model identifier for API | `nvidia/cosmos-reason1-7b` |
| `NIM_CONTAINER_NAME` | No | Container name | `nim-cosmos-reason1-7b` |

Set before running:
```bash
export NGC_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxx

# Optional: Override model (default is cosmos-reason1-7b)
export NIM_IMAGE=nvcr.io/nim/nvidia/llama-3.1-nemotron-nano-vl-8b-v1:1
export NIM_MODEL_NAME=nvidia/llama-3.1-nemotron-nano-vl-8b-v1
```

### Volume Mounts

**Model Cache** (`~/.cache/nim`):
- Stores downloaded models
- Persists across container restarts
- Size: ~10-15GB after first download

To use a different cache location:
```yaml
volumes:
  - /path/to/your/cache:/opt/nim/.cache:rw
```

### Troubleshooting NIM

**1. "unauthorized: 401 Authorization Required"**

You need to login to nvcr.io:
```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>
```

**2. "system has unsupported display driver / cuda driver combination"**

Your NVIDIA driver is too old. NIM requires:
- Driver 565+ (CUDA 12.9 support)

Upgrade:
```bash
sudo apt install nvidia-driver-565
sudo reboot
```

**3. "PermissionError: /opt/nim/.cache"**

Fix cache directory permissions:
```bash
sudo chown -R $(id -u):$(id -g) ~/.cache/nim
chmod -R 755 ~/.cache/nim
```

**4. NIM takes forever to start**

This is normal on first run (downloading model). Monitor:
```bash
# Watch logs (default container name)
docker logs -f nim-cosmos-reason1-7b

# Or use docker compose
docker compose logs -f nim
```

You should see:
- Download progress bars
- Model loading messages
- "Server started on port 8000" when ready

**5. Out of memory**

NIM requires significant VRAM (16GB+). If you have less:
- Close other GPU applications
- Reduce `shm_size` in docker-compose.yml
- Try a smaller model

---

## Customizing docker-compose

### Change Ports

```yaml
services:
  live-vlm-webui:
    environment:
      - PORT=3000  # Change from default 8090
```

### Add SSL Certificates

```yaml
services:
  live-vlm-webui:
    volumes:
      - ./your-cert.pem:/app/cert.pem:ro
      - ./your-key.pem:/app/key.pem:ro
```

### Use Different Model Cache

```yaml
services:
  ollama:
    volumes:
      - /mnt/data/ollama:/root/.ollama
```

### View Docker Compose Logs

```bash
# View all logs (active profile only)
docker compose logs -f

# View specific service
docker compose logs -f ollama
docker compose logs -f nim
docker compose logs -f live-vlm-webui

# View logs for all containers (even stopped)
docker compose logs --tail=100
```

---

## NVIDIA Container Toolkit Setup

For GPU support, you need NVIDIA Container Toolkit installed:

### Installation (Ubuntu/Debian)

```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Troubleshooting NVIDIA Container Toolkit

**If `--gpus all` doesn't work**, try CDI:

```bash
sudo mkdir -p /etc/cdi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo nvidia-ctk runtime configure --runtime=docker --cdi.enabled
sudo systemctl restart docker

# Test with CDI
docker run --rm --device nvidia.com/gpu=all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**Check Docker daemon config:**
```bash
cat /etc/docker/daemon.json
```

Should contain:
```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```


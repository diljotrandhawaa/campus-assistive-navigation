# Testing a Feature Branch on a Digital Ocean Droplet

Quick steps to run a feature branch (e.g. `feat/multi-session`) on a droplet that currently runs the `latest` Docker image.

---

## Quick test (copy-paste on droplet)

If you already cloned the repo, checked out the branch, and ran `pip install -e .` in a venv:

**1. Install OpenGL libs for headless (fixes `libGL.so.1` / cv2).** On Ubuntu 24.04 use `libgl1` (Noble dropped `libgl1-mesa-glx`):
```bash
sudo apt update && sudo apt install -y libgl1 libglib2.0-0t64
```
Ubuntu 22.04 or older: `libgl1-mesa-glx` instead of `libgl1` if needed.

**2. Run with the venv’s Python (no display needed):**
```bash
cd ~/live-vlm-webui
source .venv/bin/activate
export LIVE_VLM_API_BASE=https://integrate.api.nvidia.com/v1
export LIVE_VLM_PROCESS_EVERY=150
export LIVE_VLM_DEFAULT_MODEL=google/gemma-3-4b-it
.venv/bin/python -m live_vlm_webui.server --host 0.0.0.0 --port 8090 --ssl-cert cert.pem --ssl-key key.pem
```

If you don’t have certs yet: `./scripts/generate_cert.sh` from the repo root, or run with `--no-ssl` (HTTP only; camera may require HTTPS in the browser).

**3. Open in browser:** `https://YOUR_DROPLET_IP:8090` (accept the self-signed cert).

**Easiest option:** use [Docker from the branch](#4-or-build-and-run-docker-from-the-branch) so you don’t need venv or libGL on the host.

---

## 1. Get the feature branch

You cloned `https://github.com/NVIDIA-AI-IOT/live-vlm-webui`. The branch may be there (after a push) or on your fork.

**If the branch is on the same repo (origin):**
```bash
cd ~/live-vlm-webui
git fetch origin feat/multi-session
git checkout feat/multi-session
```

**If the branch is on your fork:**
```bash
cd ~/live-vlm-webui
git remote add myfork https://github.com/YOUR_USERNAME/live-vlm-webui.git   # once
git fetch myfork feat/multi-session
git checkout feat/multi-session
```

## 2. Free port 8090 (stop current container)

```bash
docker stop live-vlm-webui
# optional: docker rm live-vlm-webui
```

## 3. Run from source (recommended for testing a branch)

No need to build a new image. **Use a virtual environment** — on Ubuntu 24.04 (and other PEP 668 systems), `pip install` without a venv will fail with "externally-managed-environment". On a **headless server** (e.g. droplet), install OpenGL libs so OpenCV works: on Ubuntu 24.04 run `sudo apt install -y libgl1 libglib2.0-0t64` (on 22.04 or older use `libgl1-mesa-glx`). Then create and activate a venv and install:

```bash
cd ~/live-vlm-webui
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then set cloud env overrides and run:

```bash
# Cloud defaults: NVIDIA API, process every 150 frames (~5 s), optional default model
export LIVE_VLM_API_BASE=https://integrate.api.nvidia.com/v1
export LIVE_VLM_PROCESS_EVERY=150
export LIVE_VLM_DEFAULT_MODEL=google/gemma-3-4b-it
# Optional: export LIVE_VLM_API_KEY=your-nvidia-api-key

# Run (HTTPS with self-signed cert; use --no-ssl for HTTP only)
python -m live_vlm_webui.server --host 0.0.0.0 --port 8090
```

Make sure your shell shows `(.venv)` in the prompt before running `pip install -e .` or `python -m ...`.

Then open **https://YOUR_DROPLET_IP:8090** (e.g. `https://164.92.74.148:8090`). Accept the browser self-signed cert warning.

To run in the background:

```bash
nohup python -m live_vlm_webui.server --host 0.0.0.0 --port 8090 > server.log 2>&1 &
# or use tmux/screen
```

## 4. Or build and run Docker from the branch

```bash
cd ~/live-vlm-webui
docker build -t live-vlm-webui:feat -f docker/Dockerfile .

docker run -d --name live-vlm-webui-branch \
  -p 8090:8090 \
  -e LIVE_VLM_API_BASE=https://integrate.api.nvidia.com/v1 \
  -e LIVE_VLM_PROCESS_EVERY=150 \
  -e LIVE_VLM_DEFAULT_MODEL=google/gemma-3-4b-it \
  live-vlm-webui:feat
```

## 5. Restore the original container (when done testing)

```bash
docker stop live-vlm-webui-branch   # if you used the branch container
docker start live-vlm-webui         # or run the original `docker run` again
```

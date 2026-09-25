# Windows WSL Setup Guide

This guide covers running Live VLM WebUI on Windows using Windows Subsystem for Linux (WSL).

## Prerequisites

### 1. Install WSL2 with Ubuntu

If you haven't already installed WSL2:

```powershell
# Run in PowerShell as Administrator
wsl --install
# This installs WSL2 with Ubuntu by default

# Or install specific version
wsl --install -d Ubuntu-22.04
```

**Verify WSL2:**
```powershell
wsl --list --verbose
# Ensure VERSION shows 2, not 1
```

**If you're on WSL1, upgrade to WSL2:**
```powershell
wsl --set-version Ubuntu-22.04 2
```

### 2. Install NVIDIA GPU Drivers (for GPU support)

- Install the latest **NVIDIA Game Ready or Studio Drivers** on Windows
- WSL2 will automatically have GPU access (no driver installation needed in WSL!)
- Verify GPU access in WSL:

```bash
nvidia-smi
```

If you see your GPU, you're good to go! If not, see [NVIDIA WSL Documentation](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

---

## Installation

### Step 1: Launch WSL and Install Python

```bash
# Update package list
sudo apt update

# Install Python 3 and pip
sudo apt install -y python3 python3-pip python3-venv

# Create python symlink (optional but convenient)
sudo apt install -y python-is-python3
```

### Step 2: Install Live VLM WebUI

```bash
# Create virtual environment
python3 -m venv ~/venv-live-vlm
source ~/venv-live-vlm/bin/activate

# Install
pip install live-vlm-webui
```

### Step 3: Install Ollama (Recommended: Inside WSL)

**Option A: Ollama in WSL (Recommended)**

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a vision model
ollama pull llama3.2-vision:11b

# Start Ollama server
ollama serve &
```

**Option B: Use Ollama on Windows (Advanced)**

If you already have Ollama running on Windows, you need to access it via the Windows host IP:

```bash
# Find Windows host IP from WSL
ip route show | grep -i default | awk '{ print $3}'
# Example output: 172.27.128.1
```

Then configure the WebUI to use `http://172.27.128.1:11434/v1` as the API base.

**Note:** This may not work if Windows Firewall blocks the connection. Installing Ollama in WSL is simpler.

---

## Running the WebUI

### Start the Server

```bash
# Activate virtual environment
source ~/venv-live-vlm/bin/activate

# Run WebUI
live-vlm-webui
```

You'll see output like:
```
Access the server at:
  Local:   https://localhost:8090
  Network: https://172.27.142.43:8090
```

### Access from Windows Browser

**Method 1: Use WSL IP (Most Reliable)**

Open your Windows browser and navigate to:
```
https://172.27.142.43:8090
```

Replace `172.27.142.43` with the IP shown in your terminal under "Network:".

**Method 2: Use localhost (May Not Work)**

Some WSL2 configurations support localhost forwarding:
```
https://localhost:8090
```

If this doesn't work, use Method 1 (WSL IP).

**SSL Certificate Warning:**
- You'll see a browser security warning (self-signed certificate)
- Click "Advanced" → "Proceed to 172.27.142.43 (unsafe)" or similar
- This is safe for local development

---

## WSL Networking Explained

### How WSL2 Networking Works

- **WSL2 runs in a lightweight VM** with its own network adapter
- **WSL gets a dynamic IP** (e.g., `172.27.142.43`) that changes on reboot
- **Windows ↔ WSL communication** happens through this virtual network

### IP Addresses You'll See

```bash
# Inside WSL, run:
ip addr
```

You'll see:
- `127.0.0.1` - WSL's own localhost
- `172.27.x.x` - WSL's IP address (changes on reboot)
- Windows host IP: `172.27.128.1` (or similar) - visible via `ip route`

### Port Forwarding

WSL2 has automatic localhost port forwarding, but it's not always reliable:
- ✅ **WSL → Windows:** Usually works (WSL can access Windows localhost)
- ⚠️ **Windows → WSL:** May not work (Windows accessing WSL localhost)

**Solution:** Use WSL's IP address (`https://172.27.142.43:8090`) from Windows browser.

---

## GPU Support

### Verify GPU Access

```bash
# Should show your NVIDIA GPU
nvidia-smi

# Check CUDA version
nvcc --version
```

If GPU is detected, Live VLM WebUI will automatically:
- Display GPU metrics (utilization, VRAM, temperature)
- Use GPU for VLM inference (if using local Ollama/vLLM)

### GPU Not Detected?

1. **Update Windows NVIDIA drivers** to the latest version
2. **Ensure WSL2** (not WSL1): `wsl --list --verbose`
3. **Restart WSL:** `wsl --shutdown` in PowerShell, then relaunch
4. See [NVIDIA CUDA on WSL Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

---

## Troubleshooting

### Issue: Can't access https://localhost:8090 from Windows

**Solution:** Use the WSL IP address shown in the terminal instead:
```
https://172.27.142.43:8090
```

WSL2's localhost forwarding is inconsistent. The IP address always works.

---

### Issue: WebUI can't find Ollama running on Windows

**Problem:** Ollama on Windows listens on `127.0.0.1` (Windows localhost), which is different from WSL's network.

**Solution 1 (Recommended):** Install Ollama inside WSL:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
```

**Solution 2 (Advanced):** Find Windows host IP and configure manually:
```bash
# Get Windows IP from WSL
ip route show | grep -i default | awk '{ print $3}'
```

In WebUI settings:
- Backend: Custom
- API Base: `http://172.27.128.1:11434/v1` (use your Windows IP)

---

### Issue: WSL IP address keeps changing

**Problem:** WSL2 assigns a new IP address after each restart.

**Solutions:**
1. **Use a startup script** to find and display the IP:
   ```bash
   # Add to ~/.bashrc
   echo "WSL IP: https://$(hostname -I | awk '{print $1}'):8090"
   ```

2. **Install Ollama in WSL** to avoid cross-network issues

3. **Bookmark the IP** and update after WSL restarts

---

### Issue: Camera not working

WSL2 doesn't have direct USB camera access. Use:
1. **Remote camera streaming** from Windows
2. **Video file input** for testing
3. **Network camera (IP camera)**

---

### Issue: GPU monitoring shows intermittent 0% readings or takes time to start

**Problem:** WebUI initially shows `GPU=0.0%, VRAM=0.0GB` for the first few seconds, or occasionally shows 0% even when GPU is active.

**Root cause:** WSL2 has intermittent NVML (NVIDIA Management Library) errors. The WebUI now handles these gracefully with automatic retry logic.

**What happens:**
1. WebUI starts, GPU monitoring may show errors initially
2. After a few seconds, GPU stats appear correctly
3. Occasionally stats may drop to 0% temporarily during operation
4. Monitoring automatically recovers

**Current behavior (v0.1.1+):**
- ✅ GPU monitoring works on WSL2
- ⚠️ May show intermittent errors (logged once, then auto-recovers)
- ✅ Automatically retries and recovers from transient errors
- ✅ Only disables permanently after 10+ consecutive failures

**If GPU monitoring doesn't recover:**

Check the WebUI terminal for error messages. If you see persistent errors:

```bash
# Verify NVML works manually
python3 << 'EOF'
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
print(f"GPU: {util.gpu}%, Memory: {util.memory}%")
pynvml.nvmlShutdown()
EOF
```

**Workaround:** Use `nvidia-smi` in a separate terminal if GPU stats don't appear:
```bash
# Watch GPU stats in real-time
watch -n 0.5 nvidia-smi
```

**Note:** Native Linux and Windows installations don't experience these intermittent errors.

---

## Performance Considerations

### WSL2 vs Native Windows

- **GPU Performance:** Near-native (minimal overhead)
- **CPU Performance:** Near-native
- **I/O Performance:** Slower for Windows filesystem access
- **Networking:** Additional latency due to virtual network

### Optimization Tips

1. **Keep files in WSL filesystem** (`~/ or /home/`) not `/mnt/c/`
2. **Use WSL-native tools** (Ollama in WSL, not Windows)
3. **Close unnecessary Windows applications** to free GPU memory

---

## Advanced: Running in Background

### Keep WebUI Running After Closing Terminal

```bash
# Install screen or tmux
sudo apt install screen

# Start screen session
screen -S vlm-webui

# Run WebUI
source ~/venv-live-vlm/bin/activate
live-vlm-webui

# Detach: Press Ctrl+A then D
# Reattach later: screen -r vlm-webui
```

---

## Updating

```bash
source ~/venv-live-vlm/bin/activate
pip install --upgrade live-vlm-webui
```

---

## Uninstalling

```bash
# Remove virtual environment
rm -rf ~/venv-live-vlm

# Uninstall Ollama (if installed in WSL)
sudo systemctl stop ollama
sudo rm /usr/local/bin/ollama
sudo rm -rf /usr/share/ollama
```

---

## Comparison: WSL vs Native Windows

| Feature | WSL2 | Native Windows |
|---------|------|----------------|
| GPU Inference | ✅ Full (CUDA) | ✅ Full |
| GPU Monitoring | ✅ Works (intermittent errors handled) | ✅ Full (no errors) |
| Installation | Easy (pip) | Requires more setup |
| Ollama | Native Linux | Native Windows |
| Networking | Virtual network | Native |
| Camera Access | ❌ Limited | ✅ Full |
| Performance | ~95% native | 100% |
| Updates | Linux packages | Windows installers |

**Recommendation:** WSL2 is excellent for development and testing, especially if you're familiar with Linux tools.

---

## Related Documentation

- [Main Setup Guide](../setup/README.md)
- [VLM Backend Setup](../setup/vlm-backends.md)
- [GPU Monitoring](./gpu-monitoring.md)
- [Troubleshooting](../troubleshooting.md)

---

## Tested Configuration

This guide was tested on:
- **Windows 11** with WSL2
- **Ubuntu 22.04 LTS** on WSL
- **NVIDIA RTX A3000 Laptop GPU**
- **Ollama** running in WSL
- **Live VLM WebUI v0.1.0**

Your experience may vary with different hardware or configurations.

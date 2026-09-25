# Contributing to Live VLM WebUI

Thank you for your interest in contributing! This guide will help you get started.

## 🚀 How to Contribute

### Reporting Bugs

If you find a bug, please [open an issue](https://github.com/nvidia-ai-iot/live-vlm-webui/issues/new) with:
- **Platform**: PC (x86_64), DGX Spark, Jetson Orin, Jetson Thor, Mac, etc.
- **Setup**: Docker or manual installation
- **Steps to reproduce**: Clear instructions
- **Expected vs actual behavior**
- **Logs**: Error messages, stack traces
- **Environment**: Python version, GPU, NVIDIA driver version

### Suggesting Features

Have an idea? [Open an issue](https://github.com/nvidia-ai-iot/live-vlm-webui/issues/new) with:
- **Use case**: What problem does it solve?
- **Proposed solution**: How would it work?
- **Alternatives**: Other approaches you considered
- **Impact**: Who benefits from this feature?

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Test thoroughly** (see Testing section below)
5. **Commit with clear messages**: `git commit -m "Add amazing feature"`
6. **Push to your fork**: `git push origin feature/amazing-feature`
7. **Open a Pull Request** with:
   - Clear description of changes
   - Screenshots/videos if UI-related
   - Link to related issues

---

## 🧪 Testing Your Changes

### Testing Locally

```bash
# Install in development mode
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate certificates
./generate_cert.sh

# Start server
./start_server.sh
```

### Testing Docker Build

```bash
# Build x86 image
docker build -t live-vlm-webui:test .

# Build Jetson Orin image
docker build -f Dockerfile.jetson-orin -t live-vlm-webui:test-orin .

# Build Jetson Thor image
docker build -f Dockerfile.jetson-thor -t live-vlm-webui:test-thor .

# Test locally
docker run --rm -p 8090:8090 live-vlm-webui:test
```

### Testing Checklist

- [ ] Code works on PC (x86_64)
- [ ] Code works on Jetson (if applicable)
- [ ] No linter errors
- [ ] Documentation updated (if needed)
- [ ] Screenshots/videos added (if UI changed)
- [ ] Tested with at least one VLM backend (Ollama, vLLM, etc.)

---

## 📝 Code Style

### Python

- **PEP 8** compliant
- **Type hints** where appropriate
- **Docstrings** for classes and functions
- **Comments** for complex logic

```python
def process_frame(frame: VideoFrame, model: str) -> Dict[str, Any]:
    """
    Process a video frame with the VLM.

    Args:
        frame: Input video frame
        model: Model name to use

    Returns:
        Dictionary with analysis results
    """
    # Implementation here
    pass
```

### JavaScript/HTML/CSS

- **Consistent indentation** (2 spaces)
- **Descriptive variable names**
- **Comments** for complex UI logic

---

## 🌟 Priority Areas

We're particularly interested in contributions for:

### High Priority
- ➕ **Model download UI** - Ability to initiate backend's model donwload from Web UI
- 📏 **Jetson VRAM utilization** - Workaround for not being able to get GPU memory consumption via NVML
- ⚡ **Hardware-accelerated video on Jetson** - Use NVENC/NVDEC instead of CPU swscaler
- 🔄 **AMD GPU monitoring** - Add AMD GPU support

### Medium Priority
- 🏆 **Benchmark mode** - Side-by-side model comparison
- 👥 **Multi-session support** - Support multiple sessions, potentially enabling cloud hosting
- 🎥 **Multiple frame support** - Option to send multiple frames to VLM for temporal understanding
- 📜 **Log functionality** - Keep the past analysis results viewable
- 🔄 **Apple Silicon GPU monitoring** - Find a workaround for supporting M1/M2/M3/M4 in `gpu_monitor.py`

### Low Priority (Nice to Have)
- 🔊 **Audio output** - TTS for accessibility
- 🌐 **Internationalization** - Multi-language scheme

---

## 🐛 Debugging Tips

### Enable Verbose Logging

```bash
# Set log level to DEBUG
python server.py --log-level DEBUG

# Or via environment variable
export LOG_LEVEL=DEBUG
./start_server.sh
```

### Docker Debugging

```bash
# View logs
docker logs -f live-vlm-webui

# Enter container
docker exec -it live-vlm-webui bash

# Test GPU access
docker exec live-vlm-webui nvidia-smi  # PC
docker exec live-vlm-webui jtop  # Jetson
```

### Browser Debugging

- Open **Developer Tools** (F12)
- Check **Console** for JavaScript errors
- Check **Network** tab for WebSocket/API failures
- Check **Application** → LocalStorage for saved settings

---

## 📚 Documentation

### Adding New Documentation

- Main README: High-level overview and quick start
- `docs/setup/`: Setup and installation guides
- `docs/usage/`: Usage guides and tutorials
- `docs/development/`: Developer guides
- `docs/troubleshooting.md`: Common issues and solutions

### Documentation Style

- **Clear headings** with emojis for scannability
- **Code blocks** with syntax highlighting
- **Screenshots** for visual steps
- **Links** to related docs
- **Examples** before explanations

---

## 🚢 Release Process

> [!NOTE]
> We have not started the formal versionoig yet

### Versioning

We use [Semantic Versioning](https://semver.org/):
- **Major** (1.0.0 → 2.0.0): Breaking changes
- **Minor** (1.0.0 → 1.1.0): New features, backward compatible
- **Patch** (1.0.0 → 1.0.1): Bug fixes

### Creating a Release

1. **Update version** in relevant files
2. **Update CHANGELOG.md** with changes
3. **Create git tag**: `git tag v1.0.0`
4. **Push tag**: `git push origin v1.0.0`
5. **GitHub Actions** will build and publish Docker images

---

## 💬 Communication

- **GitHub Issues**: Bug reports, feature requests
- **Pull Requests**: Code contributions
- **NVIDIA Developer Forums**: Community support

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Thank You!

Every contribution helps make Live VLM WebUI better for everyone. Whether it's a bug report, documentation improvement, or new feature - we appreciate your help! 🎉


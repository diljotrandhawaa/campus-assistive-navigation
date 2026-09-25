# TODO Tracker for live-vlm-webui

**Last Updated:** 2025-11-13 (Preparing v0.2.0)

This document tracks active TODO items for upcoming releases.

---

## 🎯 v0.2.0 Release Goals

**Target Features:**
- Docker version picker
- RTSP streaming support (beta)
- ✅ Honor OS dark/light mode preference (COMPLETED)
- ✅ Markdown rendering in VLM output (COMPLETED)
- Display detailed VLM inference metrics
- Small UI/UX improvements

**Target Date:** TBD

---

## 🚀 v0.2.0 - Active Development

### Core Features

- [x] **RTSP IP Camera Support**
  - **Status**: Core functionality implemented, ready for testing
  - **What it does**: Process video from RTSP-compatible IP cameras instead of just webcams
  - **Use cases**:
    - Home safety monitoring (pool drowning detection, elder care fall detection)
    - Security surveillance with AI analysis
    - Pet/baby monitoring
    - Retail customer counting
    - Warehouse safety monitoring
    - Use Jetson/PC as dedicated edge AI surveillance appliance
  - **Implementation**:
    - ✅ `RTSPVideoTrack` class for RTSP stream ingestion with auto-reconnection
    - ✅ Backend endpoints: `/api/rtsp/start`, `/api/rtsp/stop`, `/api/rtsp/status`
    - ✅ Frontend UI: Input source selector (Webcam vs RTSP)
    - ✅ RTSP URL input with connection testing
    - ✅ Error handling and user-friendly messages
    - ✅ Comprehensive documentation with examples
  - **Documentation**: `docs/usage/rtsp-ip-cameras.md`
  - **Components modified**:
    - `src/live_vlm_webui/rtsp_track.py` (new)
    - `src/live_vlm_webui/server.py` (RTSP endpoints added)
    - `src/live_vlm_webui/static/index.html` (UI controls added)
  - **Known limitations**:
    - Single stream per session (multi-stream planned for future)
    - No video preview in UI (backend processing only)
    - CPU-based decode (hardware acceleration planned)
  - **Next steps**:
    - [ ] Unit tests for RTSPVideoTrack
    - [ ] Real IP camera testing (Reolink, Hikvision, etc.)
    - [ ] Performance benchmarking
    - [ ] Multi-stream support (future)
    - [ ] Video preview in UI via server-sent events (future)
    - [ ] Hardware-accelerated decode with NVDEC on Jetson (future)
  - **Target release**: v0.2.0

- [x] **Docker version picker** (`191a95a`, `ea8b857`)
  - Interactive version selection with `--version`, `--list-versions`, `--skip-version-pick` flags
  - Hybrid fetching: GHCR API → Releases API → Fallback list
  - Works without GITHUB_TOKEN (public API support)
  - Usage: `./scripts/start_container.sh --version 0.1.1`

### UI/UX Improvements

- [x] **Honor OS dark/light mode preference** ✅ **COMPLETED**
  - **Implementation**:
    - ✅ Detect OS preference via JavaScript (`window.matchMedia('(prefers-color-scheme: light)')`)
    - ✅ Auto-apply theme on page load based on OS preference
    - ✅ Listen for OS preference changes and update dynamically
    - ✅ Manual theme toggle cycles through: Auto → Light → Dark → Auto
    - ✅ Only manual overrides (Light/Dark) saved to localStorage; Auto mode always checks OS preference
    - ✅ Visual indicator shows current mode: Monitor icon for Auto, Sun for Light, Moon for Dark
  - **Benefits**:
    - Better UX for users with light mode preference
    - Respects accessibility settings
    - Modern web app best practice
  - **Files modified**: `src/live_vlm_webui/static/index.html`

- [ ] **Display detailed VLM inference metrics** (MEDIUM PRIORITY)
  - **Current state**: Only showing total latency (ms), avg latency, inference count
  - **Goal**: Display detailed breakdown of VLM inference phases
  - **Background**: VLM inference has two distinct phases:
    - **Prefill phase**: Image encoding + prompt → KV cache population (500-2000ms)
    - **Decode phase**: Generate tokens one-by-one (depends on response length)
  - **Metrics to display**:
    1. **Prefill time** (ms) - "Time to first token"
    2. **Decode speed** (tokens/sec) - "Generation speed"
    3. **Total time** (ms) - Already showing as "Latency"
    4. **Tokens generated** - Response length
    5. **Vision tokens** - Number of tokens from image (optional)
  - **Implementation**:
    - Extract metrics from Ollama API response (already provided)
    - Add metrics to inline display: "Prefill: Xms | Gen: Y tok/s | Tokens: Z"
    - Update WebSocket `vlm_response` message format
    - Update `vlm_service.py` to extract and return detailed metrics
  - **Note**: Ollama API provides rich metrics; vLLM/other backends may not
  - Effort: ~4-6 hours

- [x] **Markdown rendering in VLM output** ✅ **COMPLETED**
  - **Implementation**:
    - ✅ Added marked.js (v12.0.0) and DOMPurify (v3.0.8) libraries
    - ✅ Toggle button embedded in top-right corner of result balloon
    - ✅ Render VLM response HTML from markdown with sanitization
    - ✅ Toggle between "Markdown" and "Plain Text" modes
    - ✅ Markdown enabled by default
    - ✅ Supports: headers, lists, code blocks, tables, blockquotes, links, bold, italic
    - ✅ Button persists during streaming updates
    - ✅ Theme-aware styling for all markdown elements
    - ✅ Copy to clipboard button in lower-right corner with transparent overlay style
  - **Benefits**:
    - Better readability for formatted responses
    - Code blocks properly styled
    - Tables displayed with alternating row colors
    - All markdown elements styled to match UI theme
    - Easy copy functionality for sharing results
  - **Files modified**: `src/live_vlm_webui/static/index.html`

### Documentation

- [ ] **Remove "coming soon" notes from README**
  - PyPI package warning (if still present)
  - Download button mention (line 191)

- [ ] **Add CHANGELOG.md [Unreleased] section**
  - Document v0.2.0 changes as they're made
  - Prepare for release notes

---

## 📋 Backlog (Post v0.2.0)

### High Priority Features

- [ ] **Multi-frame temporal understanding** (Major feature)
  - Enable VLM to understand motion, actions, and changes over time
  - Option to ingest multiple frames for temporal context
  - Use cases: Action recognition, motion detection, change detection
  - Implementation: Multi-image API or grid/collage approach
  - Effort: ~2-3 days

- [ ] **Jetson GPU stats without jtop dependency** (Platform Support)
  - **Current issue**: jtop (jetson-stats) requirement complicates pip installation
  - **Goal**: Direct GPU utilization via NVML or sysfs
  - **Benefits**: Simpler pip installation, more efficient monitoring
  - Priority: High for Jetson users

- [ ] **Multi-user/multi-session support** (Architecture)
  - **Current limitation**: Single-user, single-session architecture
  - **Required for**: Cloud deployment, team demos, production use
  - **Implementation levels**: Basic session management → Request queue → Enterprise
  - **Current workaround**: Deploy multiple independent instances on different ports
  - Priority: Medium (required for public hosting)

### Medium Priority Features

- [ ] **Prompt template addition**
  - User-defined analysis prompts
  - More prompt templates for common use cases
  - Per-model prompt customization

- [ ] **Side-by-side VLMs comparison**
  - Compare two different VLM outputs side-by-side
  - Useful for model evaluation

- [ ] **Recording/export functionality**
  - Record analysis results with/without video stream
  - Export video as MP4/webm with annotations
  - Timestamp-based analysis log

- [ ] **Download button for recordings**
  - UI feature for downloading saved recordings
  - Priority: Low (nice-to-have)

### Platform Support

- [ ] **Hardware-accelerated video processing on Jetson**
  - Implement NVMM/VPI color space conversion
  - Reduce CPU load during video processing
  - Priority: Medium (optimization)

- [ ] **AMD GPU monitoring support**
  - ROCm/rocm-smi integration
  - Priority: Low (expand platform support)

- [ ] **Windows native installer**
  - MSI/EXE installer for Windows
  - Bundled Python runtime
  - One-click installation

- [ ] **Raspberry Pi testing**
  - Test on RPi 4/5 (if any VLM runs on RPi)
  - Document performance characteristics

### Testing & Infrastructure

- [ ] **Expand E2E test coverage**
  - Network interruptions during inference
  - Model switching edge cases
  - Camera permission denial handling
  - VLM API failures/timeouts

- [ ] **Performance benchmarking suite**
  - Standardized tests for video processing throughput
  - VLM inference latency measurements
  - GPU utilization tracking

- [ ] **Automated PyPI publishing on GitHub Release**
  - Update `.github/workflows/build-wheel.yml`
  - Configure PyPI Trusted Publishing
  - Document in `docs/development/releasing.md`

- [ ] **Code coverage improvement**
  - Current: ~20%
  - Target: >50% for core modules
  - Focus on `server.py`, `gpu_monitor.py`, `vlm_service.py`

### Backend Expansion

- [ ] **Additional VLM backends**
  - Local: vLLM (partially tested), SGLang, Local HF models
  - Cloud: OpenAI API (partially done), Anthropic Claude, Azure OpenAI

---

## 🔍 Investigation & Monitoring

### Items to Monitor

1. **jetson-stats PyPI availability for Thor**
   - Monitor: https://github.com/rbonghi/jetson_stats/releases
   - Action: Update docs when Thor support released to PyPI

2. **Python 3.13 compatibility**
   - Currently tested up to Python 3.12
   - Action: Test and update `pyproject.toml` when stable

3. **WebRTC browser compatibility**
   - Currently tested: Chrome, Firefox, Edge
   - Safari: May have WebRTC limitations
   - Action: Document browser-specific limitations

---

## ✅ Completed Releases

### v0.1.1 (Released)

- ✅ WSL2 GPU monitoring fix (`ec277db`)
- ✅ Comprehensive VLM documentation (`64ec81e`)
- ✅ Docker versioned tags and picker (`191a95a`, `ea8b857`)
- ✅ Release process documentation

### v0.1.0 (Released 2025-11-10)

- ✅ PyPI package published: `pip install live-vlm-webui`
- ✅ Multi-platform support (x86_64, ARM64, macOS, Jetson Thor/Orin)
- ✅ Docker images with multi-arch builds
- ✅ SSL certificate auto-generation
- ✅ Apache 2.0 license and OSRB approval
- ✅ Comprehensive documentation

---

## 📊 Priority Legend

- 🔴 **High**: Critical for v0.2.0 release
- 🟡 **Medium**: Important but can be deferred
- 🟢 **Low**: Nice to have, no rush

---

## 🔄 Maintenance Notes

**How to use this document:**

1. **Before each release**: Review and update priority sections
2. **During development**: Add TODOs here instead of scattered code comments
3. **After completing items**: Move to "Completed Releases" section
4. **Monthly review**: Re-prioritize based on user feedback

**Keep this document synchronized with:**
- Code comments (avoid duplicate TODOs)
- GitHub Issues (for community-reported items)
- CHANGELOG.md (for completed features)

---

**Document Status:** Active tracking for v0.2.0 development

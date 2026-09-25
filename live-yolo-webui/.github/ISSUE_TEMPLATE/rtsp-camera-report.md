---
name: RTSP Camera Compatibility Report
about: Report success or issues with your IP camera
title: '[RTSP] Camera Brand - Model Name'
labels: 'rtsp, beta, hardware-compatibility'
assignees: ''
---

## Camera Information

**Brand:**
**Model:**
**Firmware Version:**

## RTSP Configuration

**RTSP URL Format:** (redact password)
```
rtsp://username:****@ip:port/path
```

**Resolution:**
**Codec:** (H.264, H.265, MJPEG, etc.)
**Framerate:**

## Test Results

- [ ] ✅ Connection successful
- [ ] ✅ Video stream displays in WebUI
- [ ] ✅ VLM analysis works
- [ ] ❌ Connection failed
- [ ] ❌ Stream displays but no analysis
- [ ] ❌ Other issue (describe below)

## System Information

**Platform:** (PC/Jetson/Mac)
**Installation Method:** (Docker/pip)
**live-vlm-webui Version:**
**VLM Backend:** (Ollama/vLLM/etc.)

## Additional Notes

<!-- Any other details, error messages, or observations -->

## Logs

<details>
<summary>Server logs (if applicable)</summary>

```
Paste relevant server logs here
```

</details>

# Live YOLO camera alignment

This adds a separate camera mode to your **copied** `live-yolo-webui` project. It uses the same model and vocabulary as your uploaded `yolo-test.py`:

- `yolov8s-worldv2.pt`, through Ultralytics `YOLOWorld`
- chair, table, whiteboard, door, elevator door, sign, stairs
- CUDA device 0; inference size 640; initial confidence threshold 0.25

It adds `run_yolo.py` and the `yolo_live/` directory. It does not replace your original `server.py`, `video_processor.py`, `vlm_service.py`, `index.html`, or `yolo-test.py`. The old `live-vlm-webui` command keeps its existing installation. This add-on has its own camera page, styled around the same sidebar/video/output layout, with detector controls in place of VLM API controls.

## Put the files on the GB10

1. Download `live_yolo_update.zip`.
2. In the VS Code window connected to the GB10, open `/home/hp17/indoor-nav/live-yolo-webui`.
3. Drag the downloaded ZIP from your computer's file explorer into that **remote** folder in VS Code.
4. Open a terminal in that remote VS Code window and run:

```bash
conda activate indoor-nav
cd /home/hp17/indoor-nav/live-yolo-webui
python -m zipfile -e live_yolo_update.zip .
python run_yolo.py --port 8091
```

The final command stays running. Wait for the model to load and warm up, then for the server's `Running on https://127.0.0.1:8091` message.

Use the same Python environment where your single-image YOLO script works. The add-on needs `ultralytics`, a working CUDA-enabled `torch`, `numpy`, `Pillow`, and `aiohttp`. Your YOLO and Live VLM installations should already provide them. If the error specifically says `No module named aiohttp` or `PIL`, install only the missing dependency in the activated environment:

```bash
python -m pip install aiohttp Pillow
```

Do **not** reinstall PyTorch just to run this add-on, and do not run `pip install -e .` in the copied project. No new model server or ZRT service is required.

If the weights are already somewhere else, point at the existing file:

```bash
python run_yolo.py --port 8091 --weights /full/path/to/yolov8s-worldv2.pt
```

With the default asset name, Ultralytics may download the file if it cannot find it in the working directory. Its text encoder may also need a first-use download. Model files and caches live outside the Conda environment; Python dependencies are installed inside the active environment.

## Open the camera page on your laptop

1. Keep the terminal running on the GB10.
2. Open the **Ports** tab next to Terminal in VS Code. If it is hidden, use **View → Command Palette → Ports: Focus on Ports View**.
3. Choose **Forward a Port**, enter `8091`, and look at its **Forwarded Address**.
4. Open `https://localhost:8091` on the laptop. If VS Code assigned a different local port, use that forwarded port instead.
5. The app generates its own local self-signed certificate. For this server you just started, use the browser's **Advanced → Proceed to localhost** if it offers that option.
6. Click **Start camera**, allow camera access, then select your Logitech webcam or laptop camera in **Video source**. Device names usually appear after permission is granted.

The webcam plugs into the **computer running the browser**. It does not need to be plugged into the GB10. There is no API Base URL to enter: the page and detector communicate through the same port, 8091.

If your browser will not accept the local test certificate, stop only this YOLO server with Ctrl+C and restart it in loopback HTTP mode:

```bash
python run_yolo.py --port 8091 --http
```

Then open `http://localhost:8091` through the same forwarding. Browsers can allow camera access on HTTP localhost. Use HTTPS for direct access by a network address; ordinary HTTP on a remote IP is not a camera-capable secure context.

## First test

1. Select **whiteboard** as the target.
2. Put a whiteboard toward the right of the camera view. The instruction should be **Pan camera right** if it is detected.
3. Turn the camera toward the whiteboard until its detected center reaches the yellow center band. The instruction should become **Target centered**.
4. Turn farther so it moves to the left of the image. The instruction should become **Pan camera left**.
5. Point away. The instruction becomes **Whiteboard not detected**, rather than repeating an old direction.

The camera view is intentionally **not mirrored**, so image-left and image-right match the camera's directions. These are camera rotation cues, not instructions to step sideways or walk forward.

The large view displays the exact frame that produced the boxes and direction. The small inset is the current live preview. Their difference makes inference delay visible. Use **Save annotated frame** to save the currently fresh analyzed image to the laptop's Downloads folder. Direction updates also print in the GB10 terminal.

## How the live pipeline works

The browser gets frames with `getUserMedia`, copies the latest available frame, encodes it as JPEG, and sends it over a WebSocket. The GB10 runs YOLO in a worker thread and returns normalized boxes, confidence scores, directions, and timing. The browser draws the response on its retained copy of that exact frame.

This mode uses **JPEG frames over HTTPS/WebSocket**, rather than the original UI's WebRTC media transport. That lets the camera-to-model traffic pass through the same VS Code/Tailscale port forwarding as the page. It samples live video; it does not infer on every frame of a 30 FPS camera.

Only one frame is in flight per browser. No growing camera-frame queue is kept. When the GPU is processing another browser's frame, it returns `busy`; the client retries with a fresh capture. The model is shared, while each browser has its own target and alignment state. Multiple users share GPU throughput; this is not a guarantee of equal scheduling or a particular FPS.

The default upper limit is eight analyses per second. If inference and transport take longer, the actual update rate is lower. Adjust **Maximum analysis rate** and **Sent image** in the sidebar. The input is resized again by YOLO to `--imgsz` (default 640); increasing the transmitted image size alone does not increase the model's inference resolution.

## Centering and target selection

- Horizontal offset is `(box_center_x / image_width) - 0.5`.
- Within ±5% of image width counts as centered.
- Once centered, the target can move within ±8% before a pan cue appears. This hysteresis reduces flicker; the drawn band widens while centered.
- Small position changes are smoothed. Clear center crossings and large changes update directly.
- On acquisition, the highest-confidence matching detection is selected. Later frames prefer an overlapping or nearby matching box to reduce switching between objects.
- If the selected target disappears, guidance clears immediately. After a short loss, the controller can acquire another matching object. This is lightweight association, not reliable identity tracking of multiple similar doors or chairs.
- Results older than 1.5 seconds are discarded or marked stale; directional guidance clears. Camera stalls, hidden tabs, and disconnected sessions also clear guidance. A slow connection can therefore show no instruction until fresh results return.

These boxes do not measure distance, detect a safe floor route, read room names, or create a persistent navigation map. Those functions remain separate additions.

## Timing labels

| Label | What it measures |
| --- | --- |
| Frame → overlay | Browser time from copying the camera frame through JPEG encoding, transport, server processing, response handling, and submitting canvas drawing commands. It excludes camera exposure/driver buffering and the final physical display refresh. |
| Average | Arithmetic mean of Frame → overlay for displayed fresh results since the last camera start. |
| Detector | Server wall time for decoding the JPEG, running YOLO, and extracting CPU-side detections. Excludes network time. |
| Analysis updates | Recent displayed-result update rate, using an approximately five-second window. |
| Result age | Time since the currently analyzed frame was copied in the browser. |

These timers use local monotonic clocks; they do not subtract laptop and GB10 wall-clock timestamps.

## Troubleshooting

- **Address already in use:** use another port, such as `python run_yolo.py --port 8092`, and forward that port. Do not stop your teammate's processes to free a port.
- **No camera / permission denied:** allow camera permission on this page; check the USB connection and whether another app is holding the camera. Choose Default camera if a saved device is unavailable.
- **Server page works but no result:** check the GB10 terminal for an inference exception. The UI stops on errors and shows a message. Restart the camera after fixing the error.
- **CUDA unavailable or unsupported GPU:** this launcher must run in the same environment that successfully runs your YOLO image script. `nvidia-smi` alone does not prove that this environment's PyTorch supports the GPU.
- **Analysis is stale:** try a 640-pixel sent image, reduce competing GPU work, and inspect the connection. Raising the stale timeout makes older instructions visible; it does not improve speed.
- **Target not detected:** verify the target class, lighting, framing, and confidence threshold. A model can miss an object or misclassify it; a missing box does not establish an empty path.
- **Direct Tailscale access later:** the default binds only to `127.0.0.1` for SSH forwarding. For direct access, bind explicitly to the GB10's Tailscale IP, for example `--host 100.69.180.2`, and use HTTPS on that IP/hostname with an appropriate trusted certificate. The camera must still be allowed by the browser. Forwarding only 8091 is sufficient for this JPEG/WebSocket mode.

## Files to edit next

| File | Purpose |
| --- | --- |
| `run_yolo.py` | Entry point for this copy. |
| `yolo_live/engine.py` | Load YOLO once, set classes, run GPU inference off the web event loop. |
| `yolo_live/direction.py` | Image positions, target association, centering thresholds and direction text. |
| `yolo_live/server.py` | Camera sessions, frame/result messages, startup arguments and HTTPS. |
| `yolo_live/static/index.html` | Sidebar, analyzed frame, live inset, direction panel and metrics. |
| `yolo_live/static/app.js` | Camera capture, no-backlog loop, synchronized annotations and stale handling. |
| `yolo_live/static/styles.css` | Page appearance and responsive layout. |

## Validation

The package includes unit and WebSocket integration tests using a **mock detector**. They verify direction changes, hysteresis, loss, target association, GPU serialization, session isolation, and protocol handling:

```bash
python -m unittest discover -s tests_yolo_live -v
```

The optional client smoke test exercises the actual `app.js` against a local WebSocket server using a simulated DOM, camera, and detector. It needs Node 22+ on the testing machine, **not** for the production UI:

```bash
python tests_yolo_live/run_client_smoke.py
```

These checks do not test GB10 inference speed, actual camera hardware, or browser rendering. The GB10/webcam test above remains necessary. Do not use `tests_yolo_live/mock_server.py` as the real app: its detections are synthetic. Start production with **`python run_yolo.py`**.

API references: [Ultralytics YOLO-World](https://docs.ultralytics.com/models/yolo-world/) and [aiohttp WebSocket server](https://docs.aiohttp.org/en/stable/web_quickstart.html#websockets). Original UI project: [NVIDIA-AI-IOT/live-vlm-webui](https://github.com/NVIDIA-AI-IOT/live-vlm-webui).

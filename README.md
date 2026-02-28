# NERF Turret CV System v3.1

A real-time computer vision system that detects, tracks, and predicts target movements, identifies safe/hostile faces, recognizes hand gestures, and controls a NERF gun turret -- all accessible via a modern remote web dashboard.

## Current Status

| Component | Status | Notes |
|---|---|---|
| YOLO Detection (dual-model) | Implemented | YOLO11n + YOLOv8n with FP16, ROI, smoothing |
| Multi-object Tracking | Implemented | Hungarian + Kalman + re-ID |
| Face Recognition | Implemented | SQLite DB, top-K voting, multi-frame enrollment, CRUD API |
| Hand Gesture Recognition | Implemented | MediaPipe Tasks API, 5 gesture types |
| Turret Control | Implemented | PID, trapezoidal profiles, burst fire, patrol |
| Arduino Firmware | Implemented | v3.0 with watchdog, heartbeat, EEPROM |
| Remote Dashboard | Implemented | Face gallery, turret arc viz, hold-to-move, sensitivity slider |
| OpenCV HUD Overlay | Implemented | Crosshair, radar, compass, zoom, notifications |
| Adaptive Performance | Implemented | Threaded pipeline, degradation, frame skipping |
| HTTPS / TLS | Not yet | Use reverse proxy (see Remote Access) |
| Multi-user Auth | Not yet | Single admin user with JWT |

---

## Features

| Feature | Description |
|---|---|
| **YOLO Detection** | YOLO11 + YOLOv8-nano dual-model engine with FP16 half-precision, ROI detection, and temporal confidence smoothing |
| **Target Tracking** | Hungarian algorithm assignment with 6-state Kalman filter (position + velocity + acceleration), IoU+centroid hybrid cost, track maturity, and appearance re-identification |
| **Face Identification** | Classifies faces as SAFE, HOSTILE, or UNKNOWN using top-K voting, quality filtering (blur + size), and adaptive scan interval |
| **Hand Gestures** | MediaPipe gesture recognition (open palm, fist, thumbs up/down, peace, pointing) mapped to system actions |
| **Turret Control** | PID-driven pan/tilt with trapezoidal motion profiles, rate-limited serial comms, burst fire, and configurable patrol patterns |
| **Remote Dashboard** | FastAPI server with WebSocket real-time updates, MJPEG stream, virtual joystick, target radar, toast notifications, and keyboard shortcuts |
| **Military HUD** | Overlay with animated crosshair, pulsing threat brackets, target lock indicator, mini-map radar, zoom inset, compass bar, FPS sparkline, and notification toasts |
| **Performance Pipeline** | Threaded producer-consumer architecture with per-stage timing metrics, adaptive degradation, and configurable frame skipping |
| **Smart Firmware** | Arduino firmware with watchdog timer, heartbeat protocol, acceleration-limited servos, extended command set, and EEPROM config |
| **Cross-Platform** | Runs on Windows/Linux PC (GPU) and Raspberry Pi 4B (headless) |

---

## Architecture

```
main.py                        # Threaded pipeline orchestrator
├── core/
│   ├── camera.py              # Thread-safe camera (OpenCV / PiCamera2)
│   ├── detector.py            # YOLO11 + YOLOv8-nano with FP16, ROI, smoothing
│   ├── tracker.py             # Hungarian + 6-state Kalman + re-ID tracker
│   ├── face_manager.py        # Face recognition engine (top-K voting, enrollment)
│   ├── face_db.py             # SQLite face database with CRUD operations
│   ├── gesture.py             # MediaPipe hand gesture recognition
│   └── metrics.py             # Per-stage pipeline performance metrics
├── turret/
│   ├── controller.py          # Trapezoidal motion profiles, burst fire, patrol
│   └── targeting.py           # Adaptive PID, dead zone, ballistic lead, priority
├── server/
│   ├── auth.py                # JWT authentication + credential checks
│   ├── app.py                 # FastAPI REST + MJPEG + WebSocket + face CRUD API
│   └── templates/
│       └── dashboard.html     # Responsive command center with face gallery
├── ui/
│   └── display.py             # Full military HUD with radar, zoom, compass
├── firmware/
│   └── turret_controller.ino  # Arduino v3.0 with watchdog, heartbeat, EEPROM
├── config.yaml                # All system configuration
├── data/
│   ├── faces/safe/            # Safe face images (person_name/img.jpg)
│   ├── faces/hostile/         # Hostile face images
│   ├── faces/faces.db         # SQLite face encoding database (auto-created)
│   └── models/                # Downloaded YOLO weights (auto-managed)
└── logs/                      # Per-run log files
```

---

## Quick Start

### 1. Install Dependencies

**PC (Windows/Linux with GPU):**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

**Raspberry Pi 4B:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-rpi.txt
```

> **Note on `dlib`**: The `face-recognition` package requires `dlib`. On Windows, install Visual Studio Build Tools first. On RPi, `sudo apt install cmake libopenblas-dev` before pip install.

### 2. Configure

Edit `config.yaml` to match your setup:

```yaml
system:
  camera_index: 0          # Your webcam index
  resolution: [640, 480]

detection:
  active_model: high       # 'high' for PC, 'lite' for RPi
  half_precision: true     # FP16 on CUDA (auto-disabled on CPU)

turret:
  enabled: false           # Set true when hardware is connected
  serial_port: COM3        # Or /dev/ttyUSB0 on Linux
  motion_profile: trapezoidal  # or 'exponential'

targeting:
  fov_h: 60.0             # Your camera's horizontal FOV
  adaptive_pid: true       # Auto-scale PID gains

performance:
  detection_thread: true   # Background detection for higher FPS
  face_thread: true        # Background face recognition
  adaptive_skip: true      # Auto-disable subsystems under load

server:
  enabled: true
  admin_password: changeme # CHANGE THIS
  secret_key: CHANGE-ME    # CHANGE THIS

hud:
  minimap: true
  zoom_inset: true
  compass: true
```

### 3. Run

```bash
# Standard (with local display)
python main.py

# Headless (RPi, server-only)
python main.py --headless

# Custom config
python main.py --config my_config.yaml
```

### 4. Access Remote Dashboard

Open `http://<your-ip>:8000` in a browser. Log in with the credentials from `config.yaml`.

The dashboard features:
- Live MJPEG video feed with HUD overlay (click to fullscreen, press `F`)
- Real-time WebSocket status updates (~10Hz) with latency indicator
- **Turret control**: D-pad with hold-to-move, sensitivity slider (1-15), pan/tilt arc visualization
- Virtual joystick on mobile (auto-switches under 900px)
- **Face gallery**: view enrolled faces, reclassify (safe/hostile), delete, multi-frame enrollment
- Target radar canvas with sweep animation
- Subsystem and HUD element toggles (detection, tracking, faces, gestures, turret, HUD)
- Keyboard shortcuts (press `?` for help)
- Activity log with timestamped events
- Toast notifications for all system actions

---

## Keyboard Controls (Local Display)

| Key | Action |
|---|---|
| `ESC` | Quit |
| `A` | Toggle ARM/DISARM turret |
| `F` | Fire (if armed) |
| `T` | Cycle mode (tracking > patrol > manual > idle) |
| `M` | Toggle between high-perf and lite model |
| `C` | Center turret |

---

## Hand Gesture Commands

| Gesture | Action |
|---|---|
| Open Palm | Stop tracking (idle mode) |
| Fist | Engage target (tracking mode) |
| Thumbs Up | Mark target as safe |
| Thumbs Down | Mark target as hostile |
| Peace Sign | Cycle system mode |

---

## Face Database

Face encodings are stored in a **SQLite database** (`data/faces/faces.db`) with full CRUD support. Face images are saved in `data/faces/safe/` and `data/faces/hostile/`.

### Option A: Pre-load from images

Create subdirectories for each person:

```
data/faces/
├── safe/
│   ├── alice/
│   │   ├── alice_001.jpg
│   │   ├── alice_002.jpg
│   └── bob/
│       └── bob_001.jpg
└── hostile/
    └── target_1/
        ├── t1_001.jpg
        └── t1_002.jpg
```

On first run, images are encoded and stored in the SQLite database automatically.

### Option B: Live enrollment via dashboard

1. Open the remote dashboard and navigate to the **Face Database** card
2. Enter a name and select safe/hostile
3. **Quick enroll** (`+` button): captures 1 frame from the current camera view
4. **Multi-frame enroll** (`5x` button): auto-captures 5 frames over ~3 seconds for better accuracy
5. The progress bar shows capture status; frames are quality-checked (blur/size filter)

### Option C: API enrollment

```bash
# Single-frame enroll from current camera view
curl -X POST http://localhost:8000/api/faces/enroll \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "alice", "threat_level": "safe"}'

# Multi-frame (call repeatedly until complete=true)
curl -X POST http://localhost:8000/api/faces/enroll/multi \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "alice", "threat_level": "safe"}'
```

### Managing faces

- **Dashboard**: hover over a face card to reclassify or delete
- **API**: `DELETE /api/faces/{name}`, `POST /api/faces/reclassify`
- **Rebuild**: re-scans image directories and rebuilds the database
- Top-K voting uses multiple encodings per person for robust matching
- Blurry or too-small faces are auto-filtered (configurable thresholds)

### Migration from v3.0

If you have an existing `data/faces/encodings.pkl`, it will be **automatically migrated** to SQLite on first run and renamed to `encodings.pkl.bak`.

---

## Turret Hardware Setup

### Wiring (Arduino)

| Component | Arduino Pin |
|---|---|
| Pan Servo Signal | D9 (PWM) |
| Tilt Servo Signal | D10 (PWM) |
| Fire Relay Signal | D7 |
| Status LED | D13 (built-in) |

### Firmware v3.0

1. Open `firmware/turret_controller.ino` in Arduino IDE
2. Upload to your Arduino (Uno/Nano/Mega)
3. Set the correct serial port in `config.yaml`
4. Set `turret.enabled: true`

New firmware features:
- **Watchdog**: Auto-centers and disarms if no command received in 2 seconds
- **Heartbeat**: Sends `HB:<pan>,<tilt>,<armed>` every 500ms for health monitoring
- **Acceleration-limited movement**: Smooth physical servo motion with configurable max acceleration
- **Extended commands**: `C` (center), `D` (disarm), `S` (status), `R` (reboot), `ARM` (arm)
- **EEPROM config**: Stores center offsets and servo limits, configurable via `CFG:PO<val>TO<val>`

### Serial Protocol

```
PC -> Arduino:  P090T045F0\n    (pan=90, tilt=45, don't fire)
Arduino -> PC:  OK\n

PC -> Arduino:  C\n              (center)
PC -> Arduino:  D\n              (disarm)
PC -> Arduino:  S\n              (status query)
Arduino -> PC:  ST:90,90,0,0,90,90\n

Arduino -> PC:  HB:90,45,1\n    (heartbeat: pan=90, tilt=45, armed=true)
```

---

## API Reference

All endpoints (except `/api/login`, `/api/stream`, and `/ws/status`) require a Bearer token.

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/login` | Get JWT token `{username, password}` |
| GET | `/api/stream` | MJPEG video stream (no auth) |
| WS | `/ws/status` | WebSocket real-time status + bidirectional commands |
| GET | `/api/status` | Full system status JSON |
| GET | `/api/logs` | Recent system event log |
| POST | `/api/mode/{mode}` | Set mode: `tracking`, `patrol`, `manual`, `idle` |
| POST | `/api/turret/move` | Move turret `{pan, tilt, relative}` |
| POST | `/api/turret/center` | Center turret |
| POST | `/api/turret/arm` | Arm turret |
| POST | `/api/turret/disarm` | Disarm turret |
| POST | `/api/turret/fire` | Fire (requires `allow_remote_fire: true`) |
| POST | `/api/detector/{type}` | Switch model: `high` or `lite` |
| POST | `/api/target/{id}` | Set primary tracking target |
| POST | `/api/target/clear` | Clear primary target |
| POST | `/api/faces/enroll` | Enroll face from current frame `{name, threat_level}` |
| POST | `/api/faces/enroll/multi` | Multi-frame enrollment (call repeatedly) |
| POST | `/api/faces/enroll/cancel` | Cancel multi-frame enrollment |
| POST | `/api/faces/rebuild` | Rebuild face encoding database |
| GET | `/api/faces/stats` | Face database statistics |
| GET | `/api/faces/list` | List all enrolled people with counts |
| DELETE | `/api/faces/{name}` | Delete all encodings for a person |
| POST | `/api/faces/reclassify` | Change threat level `{name, new_threat_level}` |
| GET | `/api/faces/thumbnail/{id}` | Serve face thumbnail image |

### WebSocket Commands

Send JSON via `/ws/status` for real-time control:

```json
{"action": "move", "pan": 5, "tilt": 0}
{"action": "center"}
{"action": "arm"}
{"action": "disarm"}
{"action": "fire"}
{"action": "mode", "mode": "tracking"}
{"action": "select_target", "target_id": 3}
{"action": "switch_model", "model": "lite"}
```

---

## Performance Architecture

The v3.0 pipeline uses a producer-consumer threading model:

```
Camera Thread -> [Frame Queue] -> Detection Thread -> [Result Buffer]
                                                          |
                     Main Loop <--------------------------+
                       |
              +--------+--------+--------+
              |        |        |        |
           Tracker  Face ID  Gesture  Targeting
              |        |        |        |
              +--------+--------+--------+
                       |
                   HUD Render -> Display / MJPEG Stream
```

- **Per-stage metrics**: Every pipeline stage is timed; stats exposed via `/api/status` and `/ws/status`
- **Adaptive degradation**: When FPS drops below target, subsystems are disabled in configurable priority order (gestures first, then face recognition, then detection resolution)
- **Configurable**: All threading and degradation behavior controlled via `config.yaml`

---

## RPi Deployment Notes

- Use `yolov8n` (lite) model -- YOLO11 is too heavy for real-time on RPi4B CPU
- Set `system.headless: true` and access via web dashboard only
- Consider overclocking RPi4B to 2.0GHz for better FPS
- Use `picamera2` for native RPi camera module support (auto-detected)
- Install with: `pip install -r requirements-rpi.txt`
- For GPIO-direct servo control (no Arduino), extend `turret/controller.py` with `pigpio`
- Set `performance.detection_thread: true` for best results on RPi

---

## Remote Access Setup

The dashboard is accessible on your local network at `http://<ip>:8000`.

**For access outside your LAN:**

| Method | Complexity | Notes |
|---|---|---|
| **Tailscale / ZeroTier** | Low | Mesh VPN, no port forwarding needed |
| **ngrok** | Low | `ngrok http 8000` for quick tunneling |
| **Reverse proxy (Caddy)** | Medium | Auto-HTTPS: `caddy reverse-proxy --to localhost:8000` |
| **Reverse proxy (nginx)** | Medium | Manual TLS cert setup |
| **Port forwarding** | Medium | Router config + dynamic DNS |

**Security checklist:**
- Change default credentials in `config.yaml` (the system warns on startup if defaults are detected)
- Change `secret_key` to a random string
- Restrict `cors_origins` to your domain/IP instead of `["*"]`
- Keep `allow_remote_fire: false` unless you explicitly need it
- Use HTTPS via reverse proxy for any internet-facing deployment

---

## Safety

- The turret starts **DISARMED** and in **IDLE** mode by default
- Remote fire is **disabled** by default (`allow_remote_fire: false`)
- Auto-fire only engages targets classified as **HOSTILE** when armed
- **SAFE**-classified faces are never auto-targeted
- Open palm gesture acts as an emergency stop
- Arduino firmware has a 2-second watchdog: auto-disarms if communication is lost
- The server warns loudly on startup if default admin credentials are in use
- Always supervise the system when armed

---

## License

This project is for educational and hobbyist purposes. Use responsibly.

"""
FastAPI remote admin server v3.0.
Provides REST API, MJPEG video streaming, WebSocket real-time status,
CORS support, and system log endpoint.
"""

import asyncio
import cv2
import json
import logging
import mimetypes
import threading
import time
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .auth import AuthManager, get_auth_dependency, check_default_credentials

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


class LoginRequest(BaseModel):
    username: str
    password: str


class TurretMoveRequest(BaseModel):
    pan: Optional[float] = None
    tilt: Optional[float] = None
    relative: bool = False


class FaceEnrollRequest(BaseModel):
    name: str
    threat_level: str


class FaceReclassifyRequest(BaseModel):
    name: str
    new_threat_level: str


class FaceDeleteRequest(BaseModel):
    name: str


class SystemState:
    """Shared state container passed to the server from main orchestrator."""

    def __init__(self):
        self.frame: Optional[np.ndarray] = None
        self.hud_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.detector = None
        self.tracker = None
        self.face_manager = None
        self.gesture_recognizer = None
        self.turret_controller = None
        self.targeting_system = None
        self.camera = None
        self.metrics = None
        self.config: dict = {}
        self.system_armed: bool = False
        self.active_mode: str = "tracking"

        # Subsystem toggles (controllable from dashboard)
        self.detection_enabled: bool = True
        self.tracking_enabled: bool = True
        self.faces_enabled: bool = True
        self.gestures_enabled: bool = True
        self.turret_enabled: bool = True
        self.hud_enabled: bool = True

        # HUD element toggles
        self.hud_crosshair: bool = True
        self.hud_trails: bool = True
        self.hud_predictions: bool = True
        self.hud_minimap: bool = True
        self.hud_zoom: bool = True
        self.hud_compass: bool = True
        self.hud_notifications: bool = True
        self.hud_radar: bool = True


class EventLog:
    """Ring buffer for system events."""

    def __init__(self, maxlen: int = 200):
        self._events: list[dict] = []
        self._maxlen = maxlen

    def add(self, event_type: str, message: str):
        self._events.append({
            "time": time.time(),
            "type": event_type,
            "message": message,
        })
        if len(self._events) > self._maxlen:
            self._events = self._events[-self._maxlen:]

    @property
    def recent(self) -> list[dict]:
        return self._events[-50:]


def _build_status_dict(state: SystemState) -> dict:
    tracker_targets = {}
    if state.tracker:
        for tid, t in state.tracker.targets.items():
            tracker_targets[str(tid)] = {
                "center": t.center,
                "bbox": t.bbox,
                "confidence": round(t.confidence, 2),
                "class": t.class_name,
                "threat": t.threat_level,
                "face_name": t.face_name,
                "speed": round(t.kalman.speed, 1),
                "primary": t.is_primary_target,
            }

    turret_status = state.turret_controller.status if state.turret_controller else {}
    targeting_status = state.targeting_system.status if state.targeting_system else {}
    face_stats = state.face_manager.stats if state.face_manager else {}
    camera_fps = state.camera.fps if state.camera else 0
    detector_ms = state.detector.inference_time_ms if state.detector else 0
    metrics_snap = state.metrics.snapshot() if state.metrics else {}

    return {
        "mode": state.active_mode,
        "armed": state.system_armed,
        "camera_fps": round(camera_fps, 1),
        "inference_ms": round(detector_ms, 1),
        "targets": tracker_targets,
        "turret": turret_status,
        "targeting": targeting_status,
        "faces": face_stats,
        "detector_model": state.detector.active_model_name if state.detector else "N/A",
        "metrics": metrics_snap,
        "toggles": {
            "detection": state.detection_enabled,
            "tracking": state.tracking_enabled,
            "faces": state.faces_enabled,
            "gestures": state.gestures_enabled,
            "turret": state.turret_enabled,
            "hud": state.hud_enabled,
            "hud_crosshair": state.hud_crosshair,
            "hud_trails": state.hud_trails,
            "hud_predictions": state.hud_predictions,
            "hud_minimap": state.hud_minimap,
            "hud_zoom": state.hud_zoom,
            "hud_compass": state.hud_compass,
            "hud_notifications": state.hud_notifications,
            "hud_radar": state.hud_radar,
        },
    }


def create_app(state: SystemState, config: dict) -> FastAPI:
    server_cfg = config.get("server", {})
    admin_user = server_cfg.get("admin_username", "admin")
    admin_pass = server_cfg.get("admin_password", "changeme")
    check_default_credentials(admin_user, admin_pass)
    auth_manager = AuthManager(
        secret_key=server_cfg.get("secret_key", "default-insecure-key"),
        admin_username=admin_user,
        admin_password=admin_pass,
    )
    require_auth = get_auth_dependency(auth_manager)

    app = FastAPI(title="NERF Turret CV System", version="3.1.0")
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
    event_log = EventLog()

    cors_origins = server_cfg.get("cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ws_hz = server_cfg.get("websocket_hz", 10)
    ws_interval = 1.0 / max(1, ws_hz)

    # ── Login ──────────────────────────────────────────────────

    @app.post("/api/login")
    async def login(req: LoginRequest):
        if not auth_manager.verify_credentials(req.username, req.password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        token = auth_manager.create_token(req.username)
        event_log.add("auth", f"User '{req.username}' logged in")
        return {"access_token": token, "token_type": "bearer"}

    # ── Dashboard ──────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return templates.TemplateResponse("dashboard.html", {"request": request})

    # ── Video Stream (MJPEG) ───────────────────────────────────

    def generate_mjpeg():
        while True:
            with state.lock:
                frame = state.hud_frame if state.hud_frame is not None else state.frame
            if frame is not None:
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.033)

    @app.get("/api/stream")
    async def video_stream():
        return StreamingResponse(
            generate_mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # ── WebSocket Status ───────────────────────────────────────

    @app.websocket("/ws/status")
    async def ws_status(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = _build_status_dict(state)
                data["events"] = event_log.recent
                await websocket.send_text(json.dumps(data, default=str))

                try:
                    msg = await asyncio.wait_for(websocket.receive_text(), timeout=ws_interval)
                    cmd = json.loads(msg)
                    _handle_ws_command(cmd, state, server_cfg, event_log)
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug(f"WebSocket error: {e}")

    # ── System Status ──────────────────────────────────────────

    @app.get("/api/status")
    async def get_status(user: str = Depends(require_auth)):
        return _build_status_dict(state)

    # ── Turret Control ─────────────────────────────────────────

    @app.post("/api/turret/move")
    async def turret_move(req: TurretMoveRequest, user: str = Depends(require_auth)):
        if not state.turret_controller:
            raise HTTPException(400, "Turret not initialized")
        if req.relative:
            state.turret_controller.move_relative(req.pan or 0, req.tilt or 0)
        else:
            pan = req.pan if req.pan is not None else state.turret_controller.current_position[0]
            tilt = req.tilt if req.tilt is not None else state.turret_controller.current_position[1]
            state.turret_controller.move_to(pan, tilt)
        return {"status": "ok", "position": state.turret_controller.current_position}

    @app.post("/api/turret/center")
    async def turret_center(user: str = Depends(require_auth)):
        if state.turret_controller:
            state.turret_controller.center()
            event_log.add("turret", "Turret centered")
        return {"status": "ok"}

    @app.post("/api/turret/fire")
    async def turret_fire(user: str = Depends(require_auth)):
        if not server_cfg.get("allow_remote_fire", False):
            raise HTTPException(403, "Remote fire is disabled in config")
        if not state.turret_controller:
            raise HTTPException(400, "Turret not initialized")
        state.turret_controller.fire()
        event_log.add("turret", "Remote FIRE command")
        return {"status": "fired"}

    @app.post("/api/turret/arm")
    async def turret_arm(user: str = Depends(require_auth)):
        if state.turret_controller:
            state.turret_controller.arm()
            state.system_armed = True
            event_log.add("turret", "Turret ARMED")
        return {"status": "armed"}

    @app.post("/api/turret/disarm")
    async def turret_disarm(user: str = Depends(require_auth)):
        if state.turret_controller:
            state.turret_controller.disarm()
            state.system_armed = False
            event_log.add("turret", "Turret DISARMED")
        return {"status": "disarmed"}

    # ── Mode Control ───────────────────────────────────────────

    @app.post("/api/mode/{mode}")
    async def set_mode(mode: str, user: str = Depends(require_auth)):
        valid = ["tracking", "patrol", "manual", "idle"]
        if mode not in valid:
            raise HTTPException(400, f"Invalid mode. Choose from: {valid}")
        state.active_mode = mode
        if mode == "idle" and state.targeting_system:
            state.targeting_system.stop_tracking()
        elif mode == "tracking" and state.targeting_system:
            state.targeting_system.start_tracking()
        event_log.add("mode", f"Mode set to {mode}")
        return {"status": "ok", "mode": mode}

    # ── Detector Control ───────────────────────────────────────

    @app.post("/api/detector/{model_type}")
    async def switch_detector(model_type: str, user: str = Depends(require_auth)):
        if model_type not in ("high", "lite"):
            raise HTTPException(400, "Choose 'high' or 'lite'")
        if state.detector:
            state.detector.set_mode(model_type)
            event_log.add("detector", f"Switched to {model_type} model")
        return {"status": "ok", "model": model_type}

    # ── Target Selection ───────────────────────────────────────

    @app.post("/api/target/{target_id}")
    async def set_primary_target(target_id: int, user: str = Depends(require_auth)):
        if state.tracker:
            state.tracker.set_primary_target(target_id)
            event_log.add("target", f"Primary target set to #{target_id}")
        return {"status": "ok", "primary_target": target_id}

    @app.post("/api/target/clear")
    async def clear_target(user: str = Depends(require_auth)):
        if state.tracker:
            state.tracker.clear_primary_target()
        return {"status": "ok"}

    # ── Face Management ────────────────────────────────────────

    @app.post("/api/faces/enroll")
    async def enroll_face(req: FaceEnrollRequest, user: str = Depends(require_auth)):
        if not state.face_manager:
            raise HTTPException(400, "Face manager not initialized")
        with state.lock:
            frame = state.frame
        if frame is None:
            raise HTTPException(400, "No frame available")
        success = state.face_manager.add_face(frame, req.name, req.threat_level)
        if not success:
            raise HTTPException(400, "No face detected in current frame")
        event_log.add("faces", f"Enrolled '{req.name}' as {req.threat_level}")
        return {"status": "ok", "name": req.name, "threat_level": req.threat_level}

    @app.post("/api/faces/enroll/multi")
    async def enroll_multi_frame(req: FaceEnrollRequest, user: str = Depends(require_auth)):
        """Multi-frame enrollment: call repeatedly to capture multiple angles."""
        if not state.face_manager:
            raise HTTPException(400, "Face manager not initialized")
        with state.lock:
            frame = state.frame
        if frame is None:
            raise HTTPException(400, "No frame available")
        result = state.face_manager.enroll_multi_frame(frame, req.name, req.threat_level)
        if result.get("complete"):
            event_log.add("faces", f"Multi-frame enrolled '{req.name}' as {req.threat_level}")
        return result

    @app.post("/api/faces/enroll/cancel")
    async def cancel_enrollment(user: str = Depends(require_auth)):
        if state.face_manager:
            state.face_manager.cancel_enrollment()
        return {"status": "ok"}

    @app.post("/api/faces/rebuild")
    async def rebuild_faces(user: str = Depends(require_auth)):
        if state.face_manager:
            state.face_manager.rebuild_database()
            event_log.add("faces", "Face database rebuilt")
        return {"status": "ok", "stats": state.face_manager.stats if state.face_manager else {}}

    @app.get("/api/faces/stats")
    async def face_stats(user: str = Depends(require_auth)):
        if not state.face_manager:
            raise HTTPException(400, "Face manager not initialized")
        return state.face_manager.stats

    @app.get("/api/faces/list")
    async def list_faces(user: str = Depends(require_auth)):
        """List all enrolled people with encoding counts."""
        if not state.face_manager:
            raise HTTPException(400, "Face manager not initialized")
        return {"people": state.face_manager.list_people()}

    @app.delete("/api/faces/{name}")
    async def delete_person(name: str, user: str = Depends(require_auth)):
        """Delete all encodings for a named person."""
        if not state.face_manager:
            raise HTTPException(400, "Face manager not initialized")
        count = state.face_manager.delete_person(name)
        if count == 0:
            raise HTTPException(404, f"No person named '{name}' found")
        event_log.add("faces", f"Deleted '{name}' ({count} encodings)")
        return {"status": "ok", "deleted": count}

    @app.post("/api/faces/reclassify")
    async def reclassify_face(req: FaceReclassifyRequest, user: str = Depends(require_auth)):
        """Change threat level for all encodings of a person."""
        if not state.face_manager:
            raise HTTPException(400, "Face manager not initialized")
        if req.new_threat_level not in ("safe", "hostile"):
            raise HTTPException(400, "threat_level must be 'safe' or 'hostile'")
        count = state.face_manager.reclassify_person(req.name, req.new_threat_level)
        if count == 0:
            raise HTTPException(404, f"No person named '{req.name}' found")
        event_log.add("faces", f"Reclassified '{req.name}' -> {req.new_threat_level}")
        return {"status": "ok", "updated": count}

    @app.get("/api/faces/thumbnail/{face_id}")
    async def face_thumbnail(face_id: int):
        """Serve a face thumbnail image."""
        if not state.face_manager:
            raise HTTPException(400, "Face manager not initialized")
        img_path = state.face_manager.get_face_thumbnail_path(face_id)
        if not img_path or not Path(img_path).exists():
            raise HTTPException(404, "Thumbnail not found")
        mime = mimetypes.guess_type(img_path)[0] or "image/jpeg"
        return StreamingResponse(open(img_path, "rb"), media_type=mime)

    # ── Subsystem Toggles ─────────────────────────────────────

    @app.post("/api/toggle/{subsystem}")
    async def toggle_subsystem(subsystem: str, user: str = Depends(require_auth)):
        toggle_map = {
            "detection": "detection_enabled",
            "tracking": "tracking_enabled",
            "faces": "faces_enabled",
            "gestures": "gestures_enabled",
            "turret": "turret_enabled",
            "hud": "hud_enabled",
            "hud_crosshair": "hud_crosshair",
            "hud_trails": "hud_trails",
            "hud_predictions": "hud_predictions",
            "hud_minimap": "hud_minimap",
            "hud_zoom": "hud_zoom",
            "hud_compass": "hud_compass",
            "hud_notifications": "hud_notifications",
            "hud_radar": "hud_radar",
        }
        attr = toggle_map.get(subsystem)
        if not attr:
            raise HTTPException(400, f"Unknown subsystem: {subsystem}. Valid: {list(toggle_map.keys())}")
        current = getattr(state, attr)
        setattr(state, attr, not current)
        new_val = not current
        event_log.add("toggle", f"{subsystem} {'ON' if new_val else 'OFF'}")
        return {"status": "ok", "subsystem": subsystem, "enabled": new_val}

    # ── Event Log ──────────────────────────────────────────────

    @app.get("/api/logs")
    async def get_logs(user: str = Depends(require_auth)):
        return {"events": event_log.recent}

    return app


def _handle_ws_command(cmd: dict, state: SystemState, server_cfg: dict, event_log: EventLog):
    """Handle a command received over WebSocket."""
    action = cmd.get("action")
    if not action:
        return

    if action == "move":
        if state.turret_controller:
            state.turret_controller.move_relative(cmd.get("pan", 0), cmd.get("tilt", 0))
    elif action == "center":
        if state.turret_controller:
            state.turret_controller.center()
    elif action == "arm":
        if state.turret_controller:
            state.turret_controller.arm()
            state.system_armed = True
            event_log.add("turret", "Turret ARMED (WS)")
    elif action == "disarm":
        if state.turret_controller:
            state.turret_controller.disarm()
            state.system_armed = False
            event_log.add("turret", "Turret DISARMED (WS)")
    elif action == "fire":
        if server_cfg.get("allow_remote_fire", False) and state.turret_controller:
            state.turret_controller.fire()
            event_log.add("turret", "Remote FIRE (WS)")
    elif action == "mode":
        mode = cmd.get("mode", "idle")
        state.active_mode = mode
        if mode == "tracking" and state.targeting_system:
            state.targeting_system.start_tracking()
        elif state.targeting_system:
            state.targeting_system.stop_tracking()
        event_log.add("mode", f"Mode set to {mode} (WS)")
    elif action == "select_target":
        if state.tracker:
            state.tracker.set_primary_target(cmd.get("target_id", 0))
    elif action == "switch_model":
        if state.detector:
            state.detector.set_mode(cmd.get("model", "high"))
    elif action == "toggle":
        subsystem = cmd.get("subsystem", "")
        toggle_map = {
            "detection": "detection_enabled",
            "tracking": "tracking_enabled",
            "faces": "faces_enabled",
            "gestures": "gestures_enabled",
            "turret": "turret_enabled",
            "hud": "hud_enabled",
            "hud_crosshair": "hud_crosshair",
            "hud_trails": "hud_trails",
            "hud_predictions": "hud_predictions",
            "hud_minimap": "hud_minimap",
            "hud_zoom": "hud_zoom",
            "hud_compass": "hud_compass",
            "hud_notifications": "hud_notifications",
            "hud_radar": "hud_radar",
        }
        attr = toggle_map.get(subsystem)
        if attr:
            current = getattr(state, attr)
            new_val = cmd.get("enabled", not current)
            setattr(state, attr, new_val)
            event_log.add("toggle", f"{subsystem} {'ON' if new_val else 'OFF'} (WS)")

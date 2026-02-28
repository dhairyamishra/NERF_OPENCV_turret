"""
NERF Turret CV System v3.0 - Main Orchestrator
================================================
Producer-consumer threaded pipeline that coordinates all subsystems:
camera, detection, tracking, face ID, gesture recognition, turret control,
HUD rendering, and remote server.

Usage:
    python main.py                  # Run with default config
    python main.py --config my.yaml # Custom config
    python main.py --headless       # No local display (RPi)
"""

import argparse
import cv2
import logging
import queue
import signal
import sys
import threading
import time
import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.camera import Camera
from core.detector import Detector
from core.tracker import TargetTracker
from core.face_manager import FaceManager
from core.gesture import GestureRecognizer
from core.metrics import PipelineMetrics, Timer
from turret.controller import TurretController
from turret.targeting import TargetingSystem
from ui.display import HUDRenderer
from server.app import SystemState, create_app

console = Console()


# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

def _detect_low_perf_hardware() -> bool:
    """Heuristic: detect RPi, ARM, or low core-count systems."""
    import platform as _plat
    machine = _plat.machine().lower()
    if machine.startswith("aarch64") or machine.startswith("arm"):
        return True
    import os
    try:
        if os.cpu_count() and os.cpu_count() <= 4:
            return True
    except Exception:
        pass
    return False


def _apply_low_perf_preset(config: dict):
    """Override config values for low-performance hardware."""
    perf = config.setdefault("performance", {})
    perf.setdefault("detect_every_n_frames", 3)
    perf.setdefault("gesture_every_n_frames", 6)
    perf.setdefault("detect_resolution", [320, 240])
    perf["target_fps"] = perf.get("target_fps", 15)

    config["detection"]["active_model"] = "lite"
    config["detection"]["confidence"] = max(config["detection"].get("confidence", 0.5), 0.45)

    config["system"]["resolution"] = config["system"].get("resolution", [640, 480])
    res = config["system"]["resolution"]
    if res[0] > 640:
        config["system"]["resolution"] = [640, 480]

    config.setdefault("faces", {})["downscale"] = 0.25
    config.setdefault("faces", {}).setdefault("scan_interval", 10)

    config.setdefault("tracking", {})["prediction_steps"] = 5
    config.setdefault("tracking", {})["trail_length"] = 20
    config.setdefault("tracking", {})["reid_enabled"] = False

    hud = config.setdefault("hud", {})
    hud["minimap"] = False
    hud["zoom_inset"] = False
    hud["fps_graph"] = False
    hud["compass"] = False
    hud["engagement_zone"] = False
    hud["target_lock_animation"] = False

    console.print("[yellow]Low-perf preset applied: frame-skip, downscale, reduced HUD[/yellow]")


def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        console.print(f"[red]Config file not found: {path}[/red]")
        sys.exit(1)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    preset = config.get("performance", {}).get("preset", "auto")
    if preset == "low" or (preset == "auto" and _detect_low_perf_hardware()):
        _apply_low_perf_preset(config)
    elif preset == "auto":
        console.print("[green]Hardware looks capable - using full performance config[/green]")

    return config


def setup_logging(config: dict) -> Path:
    log_cfg = config.get("logging", {})
    log_dir = Path(log_cfg.get("log_dir", "logs"))
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"log_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"

    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    handlers = []
    if log_cfg.get("file_logging", True):
        handlers.append(logging.FileHandler(log_file))

    if log_cfg.get("console_rich", True):
        from rich.logging import RichHandler
        handlers.append(RichHandler(console=console, rich_tracebacks=True, show_path=False))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    return log_file


# ══════════════════════════════════════════════════════════════
# Main Application
# ══════════════════════════════════════════════════════════════

class NerfTurretSystem:
    """Top-level system orchestrator with threaded pipeline."""

    def __init__(self, config: dict, headless: bool = False):
        self.config = config
        self.headless = headless or config["system"].get("headless", False)
        self.running = False
        self._shutdown_event = threading.Event()

        perf_cfg = config.get("performance", {})
        self._use_detect_thread = perf_cfg.get("detection_thread", True)
        self._use_face_thread = perf_cfg.get("face_thread", True)
        self._adaptive_skip = perf_cfg.get("adaptive_skip", True)
        self._target_fps = perf_cfg.get("target_fps", 25)
        self._degradation_order = perf_cfg.get("degradation_order", [])

        self._detect_every_n = perf_cfg.get("detect_every_n_frames", 1)
        self._gesture_every_n = perf_cfg.get("gesture_every_n_frames", 3)
        detect_res = perf_cfg.get("detect_resolution", None)
        self._detect_input_size = tuple(detect_res) if detect_res else None

        self.metrics = PipelineMetrics()

        # Subsystems
        self.camera = Camera(config)
        self.detector = Detector(config)
        self.tracker = TargetTracker(config)
        self.face_manager = FaceManager(config)
        self.gesture = GestureRecognizer(config)
        self.turret = TurretController(config)
        self.targeting = TargetingSystem(config)
        self.hud = HUDRenderer(config)

        # Shared state for server
        self.state = SystemState()
        self.state.config = config
        self.state.detector = self.detector
        self.state.tracker = self.tracker
        self.state.face_manager = self.face_manager
        self.state.gesture_recognizer = self.gesture
        self.state.turret_controller = self.turret
        self.state.targeting_system = self.targeting
        self.state.camera = self.camera
        self.state.metrics = self.metrics

        # Thread-safe queues for pipeline
        self._detect_queue = queue.Queue(maxsize=2)
        self._detect_result_queue = queue.Queue(maxsize=2)
        self._face_queue = queue.Queue(maxsize=1)
        self._face_result_queue = queue.Queue(maxsize=1)

        # Latest results (lock-free read with volatile-style pattern)
        self._latest_detections: list = []
        self._latest_face_results: list = []
        self._detect_lock = threading.Lock()
        self._face_lock = threading.Lock()

        self._server_thread = None
        self._detect_thread = None
        self._face_thread = None
        self._degraded_subsystems: set[str] = set()

        self._last_gesture_action = ""
        self._last_gesture_time = 0.0
        self._gesture_cooldown = 1.5  # seconds between same gesture action

        self.logger = logging.getLogger("system")

    def start(self):
        """Initialize all subsystems and begin processing."""
        self.logger.info("=" * 60)
        self.logger.info("NERF TURRET CV SYSTEM v3.0 STARTING")
        self.logger.info("=" * 60)

        self.camera.start()
        self.logger.info("Camera initialized")

        self.detector.load()
        self.logger.info("Detection models loaded")

        self.face_manager.load_database()
        face_stats = self.face_manager.stats
        self.logger.info(f"Face DB: {face_stats['unique_safe']} safe, {face_stats['unique_hostile']} hostile")

        self.gesture.start()
        self.turret.connect()

        if self.config.get("server", {}).get("enabled", False):
            self._start_server()

        # Start background threads
        if self._use_detect_thread:
            self._detect_thread = threading.Thread(target=self._detection_worker, daemon=True)
            self._detect_thread.start()
            self.logger.info("Detection thread started")

        if self._use_face_thread:
            self._face_thread = threading.Thread(target=self._face_worker, daemon=True)
            self._face_thread.start()
            self.logger.info("Face recognition thread started")

        self.running = True
        self._run_loop()

    def _start_server(self):
        import uvicorn

        app = create_app(self.state, self.config)
        server_cfg = self.config["server"]
        host = server_cfg.get("host", "0.0.0.0")
        port = server_cfg.get("port", 8000)

        uvi_config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(uvi_config)

        self._server_thread = threading.Thread(target=server.run, daemon=True)
        self._server_thread.start()
        self.logger.info(f"Remote server running at http://{host}:{port}")

    # ── Background Workers ────────────────────────────────────

    def _detection_worker(self):
        """Background thread: pulls frames, runs YOLO, publishes results."""
        while not self._shutdown_event.is_set():
            try:
                frame = self._detect_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                with Timer(self.metrics.stage("detection")):
                    detections = self.detector.detect(
                        frame, input_size=self._detect_input_size
                    )
                with self._detect_lock:
                    self._latest_detections = detections
            except Exception as e:
                self.logger.error(f"Detection worker error: {e}")

    def _face_worker(self):
        """Background thread: pulls frames, runs face recognition, publishes results."""
        while not self._shutdown_event.is_set():
            try:
                frame = self._face_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                with Timer(self.metrics.stage("face_id")):
                    results = self.face_manager.identify_faces(frame, force=True)
                with self._face_lock:
                    self._latest_face_results = results
            except Exception as e:
                self.logger.error(f"Face worker error: {e}")

    # ── Adaptive Degradation ──────────────────────────────────

    def _check_degradation(self):
        current_fps = self.metrics.pipeline_fps
        if not self._adaptive_skip or current_fps <= 0:
            return

        if current_fps < self._target_fps * 0.8:
            for subsystem in self._degradation_order:
                if subsystem not in self._degraded_subsystems:
                    self._degraded_subsystems.add(subsystem)
                    self.logger.warning(f"Performance: disabling {subsystem} (FPS={current_fps:.0f})")
                    break
        elif current_fps > self._target_fps * 1.1:
            if self._degraded_subsystems:
                restored = self._degraded_subsystems.pop()
                self.logger.info(f"Performance: re-enabling {restored} (FPS={current_fps:.0f})")

    # ── Main Loop ─────────────────────────────────────────────

    def _run_loop(self):
        self.logger.info("Entering main processing loop")
        frame_interval_check = 0
        loop_frame_count = 0
        cached_detections = []
        cached_gesture_results = []

        while self.running and not self._shutdown_event.is_set():
            self.metrics.tick_fps()
            loop_frame_count += 1

            # 1. Capture frame
            with Timer(self.metrics.stage("capture")):
                ret, frame = self.camera.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            with self.state.lock:
                self.state.frame = frame

            run_detection = (loop_frame_count % self._detect_every_n == 0)
            run_gesture = (loop_frame_count % self._gesture_every_n == 0)

            # 2. Object detection (skip frames for performance)
            if self.state.detection_enabled and run_detection:
                if self._use_detect_thread:
                    try:
                        self._detect_queue.put_nowait(frame)
                    except queue.Full:
                        pass
                    with self._detect_lock:
                        cached_detections = list(self._latest_detections)
                else:
                    try:
                        with Timer(self.metrics.stage("detection")):
                            cached_detections = self.detector.detect(
                                frame, input_size=self._detect_input_size
                            )
                    except Exception as e:
                        self.logger.error(f"Detection error: {e}")
            detections = cached_detections

            # 3. Update tracker (respects toggle)
            if self.state.tracking_enabled:
                with Timer(self.metrics.stage("tracking")):
                    targets = self.tracker.update(detections)
            else:
                targets = self.tracker.targets

            # 4. Face identification (respects toggle)
            face_results = []
            if self.state.faces_enabled and "faces" not in self._degraded_subsystems:
                if self._use_face_thread:
                    frame_interval_check += 1
                    if frame_interval_check % self.face_manager.scan_interval == 0:
                        try:
                            self._face_queue.put_nowait(frame)
                        except queue.Full:
                            pass
                    with self._face_lock:
                        face_results = list(self._latest_face_results)
                else:
                    try:
                        with Timer(self.metrics.stage("face_id")):
                            face_results = self.face_manager.identify_faces(frame)
                    except Exception as e:
                        self.logger.error(f"Face ID error: {e}")

                self._match_faces_to_targets(targets, face_results)

            # 5. Gesture recognition (skip frames for performance)
            if self.state.gestures_enabled and "gestures" not in self._degraded_subsystems and run_gesture:
                try:
                    with Timer(self.metrics.stage("gesture")):
                        cached_gesture_results = self.gesture.process(frame)
                    self._handle_gestures(cached_gesture_results)
                except Exception as e:
                    self.logger.error(f"Gesture error: {e}")
            gesture_results = cached_gesture_results

            # 6. Targeting + Turret update (respects toggle)
            if self.state.turret_enabled:
                with Timer(self.metrics.stage("targeting")):
                    if self.state.active_mode == "tracking":
                        self._update_targeting(targets, frame.shape[1], frame.shape[0])
                    elif self.state.active_mode == "patrol":
                        self._update_patrol()
                self.turret.update()

            # 7. Render HUD (renders in-place on a copy to avoid mutating the capture)
            hud_frame = frame.copy()
            if self.state.hud_enabled:
                self.hud.show_crosshair = self.state.hud_crosshair
                self.hud.show_trails = self.state.hud_trails
                self.hud.show_predictions = self.state.hud_predictions
                self.hud.show_minimap = self.state.hud_minimap
                self.hud.show_zoom = self.state.hud_zoom
                self.hud.show_compass = self.state.hud_compass
                self.hud.show_notifications = self.state.hud_notifications

                with Timer(self.metrics.stage("render")):
                    hud_frame = self.hud.render(
                        frame=hud_frame,
                        targets=targets,
                        face_results=face_results,
                        gesture_results=gesture_results,
                        turret_status=self.turret.status,
                        targeting_status=self.targeting.status,
                        system_mode=self.state.active_mode,
                        camera_fps=self.camera.fps,
                        inference_ms=self.detector.inference_time_ms,
                        armed=self.state.system_armed,
                        metrics=self.metrics,
                    )

            with self.state.lock:
                self.state.hud_frame = hud_frame

            # 8. Local display
            if not self.headless:
                cv2.imshow("NERF Turret - Live Feed", hud_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    self.logger.info("ESC pressed, shutting down")
                    break
                elif key == ord('a'):
                    self._toggle_arm()
                elif key == ord('f'):
                    self.turret.fire()
                elif key == ord('t'):
                    self._cycle_mode()
                elif key == ord('m'):
                    self._toggle_model()
                elif key == ord('c'):
                    self.turret.center()

            # 9. Adaptive degradation check
            self._check_degradation()

        self.shutdown()

    def _match_faces_to_targets(self, targets: dict, face_results: list):
        for face in face_results:
            fx1, fy1, fx2, fy2 = face.bbox
            face_center = ((fx1 + fx2) // 2, (fy1 + fy2) // 2)

            best_target = None
            best_dist = float("inf")

            for tid, target in targets.items():
                tx1, ty1, tx2, ty2 = target.bbox
                if tx1 <= face_center[0] <= tx2 and ty1 <= face_center[1] <= ty2:
                    dist = ((target.center[0] - face_center[0]) ** 2 +
                            (target.center[1] - face_center[1]) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_target = target

            if best_target:
                best_target.threat_level = face.threat_level
                best_target.face_name = face.name

    def _handle_gestures(self, gesture_results: list):
        now = time.time()
        for g in gesture_results:
            if g.action == "none":
                continue
            if g.action == self._last_gesture_action and (now - self._last_gesture_time) < self._gesture_cooldown:
                continue
            self._last_gesture_action = g.action
            self._last_gesture_time = now
            if g.action == "stop_tracking":
                self.state.active_mode = "idle"
                self.targeting.stop_tracking()
                self.hud.add_notification("STOP - Open Palm")
                self.logger.info("Gesture: STOP (open palm)")
            elif g.action == "engage_target":
                self.state.active_mode = "tracking"
                self.targeting.start_tracking()
                self.hud.add_notification("ENGAGE - Fist")
                self.logger.info("Gesture: ENGAGE (fist)")
            elif g.action == "mark_safe":
                self.hud.add_notification("MARK SAFE - Thumbs Up")
                self.logger.info("Gesture: MARK SAFE (thumbs up)")
            elif g.action == "mark_hostile":
                self.hud.add_notification("MARK HOSTILE - Thumbs Down")
                self.logger.info("Gesture: MARK HOSTILE (thumbs down)")
            elif g.action == "toggle_mode":
                self._cycle_mode()
                self.hud.add_notification("MODE CYCLE - Peace")
                self.logger.info("Gesture: TOGGLE MODE (peace)")
            elif g.action == "fire":
                if self.state.system_armed:
                    self.turret.fire()
                    self.hud.add_notification("FIRE - Pointing")
                    self.logger.info("Gesture: FIRE (pointing)")
                else:
                    self.hud.add_notification("FIRE BLOCKED - Not Armed")
                    self.logger.warning("Gesture: FIRE rejected (not armed)")

    def _update_targeting(self, targets: dict, frame_w: int, frame_h: int):
        primary = self.tracker.primary_target

        if primary is None:
            hostiles = [t for t in targets.values() if t.threat_level == "hostile"]
            if hostiles:
                primary = self.targeting.select_priority_target(hostiles, frame_w, frame_h)
                self.tracker.set_primary_target(primary.target_id)
            else:
                primary = self.tracker.get_closest_to_center(frame_w, frame_h)

        if primary is None:
            return

        predicted = None
        if primary.predicted_path:
            steps_ahead = min(3, len(primary.predicted_path) - 1)
            predicted = primary.predicted_path[steps_ahead]

        pan, tilt = self.turret.current_position
        new_pan, new_tilt, on_target = self.targeting.compute_angles(
            target_center=primary.center,
            predicted_center=predicted,
            current_pan=pan,
            current_tilt=tilt,
        )

        self.turret.move_to(new_pan, new_tilt)

        if (self.state.system_armed and on_target and
                primary.threat_level == "hostile" and not self.turret._is_firing):
            self.turret.fire()
            self.hud.add_notification(f"AUTO-FIRE #{primary.target_id}")
            self.logger.warning(f"AUTO-FIRE on target #{primary.target_id} ({primary.face_name or 'Unknown'})")

    def _update_patrol(self):
        pan, tilt = self.turret.get_patrol_position()
        self.turret.move_to(pan, tilt)

    def _toggle_arm(self):
        if self.state.system_armed:
            self.turret.disarm()
            self.state.system_armed = False
            self.hud.add_notification("TURRET DISARMED")
        else:
            self.turret.arm()
            self.state.system_armed = True
            self.hud.add_notification("TURRET ARMED")

    def _cycle_mode(self):
        modes = ["tracking", "patrol", "manual", "idle"]
        idx = modes.index(self.state.active_mode) if self.state.active_mode in modes else 0
        new_mode = modes[(idx + 1) % len(modes)]
        self.state.active_mode = new_mode
        self.hud.add_notification(f"MODE: {new_mode.upper()}")
        self.logger.info(f"Mode changed to: {new_mode}")
        if new_mode == "tracking":
            self.targeting.start_tracking()
        else:
            self.targeting.stop_tracking()

    def _toggle_model(self):
        if self.detector._active == "high":
            self.detector.set_mode("lite")
            self.hud.add_notification("MODEL: LITE")
        else:
            self.detector.set_mode("high")
            self.hud.add_notification("MODEL: HIGH PERF")

    def shutdown(self):
        self.running = False
        self._shutdown_event.set()

        self.logger.info("Shutting down...")
        self.turret.disarm()
        self.turret.disconnect()
        self.gesture.stop()
        self.camera.stop()

        if not self.headless:
            cv2.destroyAllWindows()

        self.logger.info("System shutdown complete")


# ══════════════════════════════════════════════════════════════
# Terminal Banner
# ══════════════════════════════════════════════════════════════

def print_banner(config: dict, log_file: Path):
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan", justify="right")
    table.add_column(style="white")

    table.add_row("Platform", config["system"]["platform"])
    table.add_row("Camera", str(config["system"]["camera_index"]))
    table.add_row("Resolution", f"{config['system']['resolution'][0]}x{config['system']['resolution'][1]}")
    table.add_row("Model (High)", config["detection"]["model_high"])
    table.add_row("Model (Lite)", config["detection"]["model_lite"])
    table.add_row("Turret", "ENABLED" if config["turret"]["enabled"] else "SIMULATION")
    table.add_row("Server", f"http://0.0.0.0:{config['server']['port']}" if config["server"]["enabled"] else "DISABLED")
    table.add_row("Faces", f"{config['faces']['safe_dir']} / {config['faces']['hostile_dir']}")
    table.add_row("Pipeline", "THREADED" if config.get("performance", {}).get("detection_thread", True) else "SYNC")
    table.add_row("Log", str(log_file))

    panel = Panel(
        table,
        title="[bold cyan]NERF TURRET CV SYSTEM v3.0[/bold cyan]",
        subtitle="[dim]ESC=quit | A=arm | F=fire | T=mode | M=model | C=center[/dim]",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)


# ══════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NERF Turret CV System v3.0")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--headless", action="store_true", help="Run without local display")
    args = parser.parse_args()

    config = load_config(args.config)
    log_file = setup_logging(config)

    print_banner(config, log_file)

    system = NerfTurretSystem(config, headless=args.headless)

    def signal_handler(sig, frame):
        console.print("\n[yellow]Interrupt received, shutting down...[/yellow]")
        system.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        system.start()
    except KeyboardInterrupt:
        system.shutdown()
    except Exception as e:
        logging.getLogger("system").exception("Fatal error")
        console.print(f"[red]Fatal error: {e}[/red]")
        system.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Face identification system v3.1 with:
- Thread-safe design for background processing
- Adaptive scan interval based on scene activity
- Top-K distance voting for robust matching
- Face quality filter (blur + minimum size check)
- OpenCV LBPH fallback for detection-only mode
- SQLite-backed persistent face database with full CRUD
- Multi-frame enrollment support
- Legacy pickle migration
"""

import cv2
import numpy as np
import logging
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from .face_db import FaceDatabase

logger = logging.getLogger(__name__)

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    logger.warning("face_recognition not installed. Face ID will use OpenCV LBPH fallback.")


@dataclass
class FaceResult:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    name: str
    threat_level: str                # safe | hostile | unknown
    confidence: float
    encoding: Optional[np.ndarray] = None


class FaceManager:
    """
    Manages a database of known faces and identifies faces in frames.
    Designed for thread-safe background operation.
    Uses SQLite-backed FaceDatabase for persistent storage.
    """

    def __init__(self, config: dict):
        face_cfg = config["faces"]
        self.enabled = face_cfg.get("enabled", True)
        self.tolerance = face_cfg.get("recognition_tolerance", 0.5)
        self.safe_dir = Path(face_cfg["safe_dir"])
        self.hostile_dir = Path(face_cfg["hostile_dir"])
        self.unknown_is_hostile = face_cfg.get("unknown_is_hostile", False)
        self.scan_interval = face_cfg.get("scan_interval", 5)
        self.min_face_size = face_cfg.get("min_face_size", 40)
        self.blur_threshold = face_cfg.get("blur_threshold", 50.0)
        self.top_k = face_cfg.get("top_k_voting", 3)

        # SQLite face database
        self._db = FaceDatabase()

        # In-memory cache (loaded from DB for fast matching)
        self._known_encodings: list[np.ndarray] = []
        self._known_names: list[str] = []
        self._known_threats: list[str] = []
        self._known_ids: list[int] = []
        self._lock = threading.Lock()
        self._frame_counter = 0
        self._last_results: list[FaceResult] = []

        # Adaptive scan interval
        self._base_interval = self.scan_interval
        self._unknown_streak = 0
        self._downscale = face_cfg.get("downscale", 0.5)

        # Multi-frame enrollment buffer
        self._enroll_buffer: list[dict] = []
        self._enroll_target_frames = 5

        self._lbph_recognizer = None
        self._lbph_labels: dict[int, tuple[str, str]] = {}

    def load_database(self):
        if not self.enabled:
            logger.info("Face recognition disabled")
            return

        self.safe_dir.mkdir(parents=True, exist_ok=True)
        self.hostile_dir.mkdir(parents=True, exist_ok=True)

        # Open SQLite database
        self._db.open()

        # Migrate legacy pickle if it exists and DB is empty
        db_stats = self._db.stats()
        if db_stats["total_encodings"] == 0:
            migrated = self._db.migrate_from_pickle()
            if migrated > 0:
                logger.info(f"Migrated {migrated} encodings from legacy pickle")

        # If DB is still empty, build from image directories
        db_stats = self._db.stats()
        if db_stats["total_encodings"] == 0:
            self._build_database()

        # Load into memory cache for fast matching
        self._reload_cache()

    def _reload_cache(self):
        """Load all encodings from SQLite into memory for fast matching."""
        encodings, names, threats, ids = self._db.get_all_encodings()
        with self._lock:
            self._known_encodings = encodings
            self._known_names = names
            self._known_threats = threats
            self._known_ids = ids
        logger.info(f"Face cache loaded: {len(names)} encodings")

    def _build_database(self):
        """Scan image directories and populate the SQLite database."""
        count = 0
        for threat_level, directory in [("safe", self.safe_dir), ("hostile", self.hostile_dir)]:
            if not directory.exists():
                continue
            for person_dir in directory.iterdir():
                if not person_dir.is_dir():
                    continue
                person_name = person_dir.name
                for img_path in person_dir.glob("*"):
                    if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                        continue
                    count += self._encode_face_image(img_path, person_name, threat_level)

        logger.info(f"Built face database from images: {count} encodings")

    def _encode_face_image(self, img_path: Path, person_name: str, threat_level: str) -> int:
        """Encode faces from an image file and store in the database. Returns count added."""
        if not FACE_REC_AVAILABLE:
            return 0

        try:
            image = face_recognition.load_image_file(str(img_path))
            encodings = face_recognition.face_encodings(image)
            for enc in encodings:
                self._db.add_face(
                    name=person_name,
                    threat_level=threat_level,
                    encoding=enc,
                    image_path=str(img_path),
                    source="import",
                    quality=0.0,
                )
            if encodings:
                logger.debug(f"Encoded {len(encodings)} face(s) from {img_path.name} ({person_name})")
            return len(encodings)
        except Exception as e:
            logger.warning(f"Failed to encode {img_path}: {e}")
            return 0

    def _is_face_quality_ok(self, frame: np.ndarray, bbox: tuple) -> bool:
        """Check if a face region meets minimum quality standards."""
        x1, y1, x2, y2 = bbox
        face_w = x2 - x1
        face_h = y2 - y1

        if face_w < self.min_face_size or face_h < self.min_face_size:
            return False

        h, w = frame.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            return False

        face_roi = frame[y1c:y2c, x1c:x2c]
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        return laplacian_var >= self.blur_threshold

    def _match_with_voting(self, encoding: np.ndarray) -> tuple[str, str, float]:
        """
        Match a face encoding using top-K distance voting.
        Returns (name, threat_level, confidence).
        """
        with self._lock:
            if not self._known_encodings:
                return "Unknown", "hostile" if self.unknown_is_hostile else "unknown", 0.0

            distances = face_recognition.face_distance(self._known_encodings, encoding)

        k = min(self.top_k, len(distances))
        top_k_indices = np.argsort(distances)[:k]

        votes: dict[str, list[float]] = {}
        for idx in top_k_indices:
            idx = int(idx)
            dist = distances[idx]
            if dist < self.tolerance:
                name = self._known_names[idx]
                threat = self._known_threats[idx]
                key = f"{name}|{threat}"
                if key not in votes:
                    votes[key] = []
                votes[key].append(1.0 - dist)

        if not votes:
            return "Unknown", "hostile" if self.unknown_is_hostile else "unknown", 0.0

        best_key = max(votes, key=lambda vk: (len(votes[vk]), sum(votes[vk]) / len(votes[vk])))
        parts = best_key.split("|")
        name = parts[0]
        threat = parts[1]
        avg_conf = sum(votes[best_key]) / len(votes[best_key])

        return name, threat, avg_conf

    def _update_adaptive_interval(self, results: list[FaceResult]):
        """Increase scan frequency when unknowns detected, decrease when stable."""
        unknowns = sum(1 for r in results if r.name == "Unknown")
        if unknowns > 0:
            self._unknown_streak += 1
            self.scan_interval = max(2, self._base_interval - self._unknown_streak)
        else:
            self._unknown_streak = max(0, self._unknown_streak - 1)
            self.scan_interval = min(self._base_interval + 2, self._base_interval + self._unknown_streak)

    def identify_faces(self, frame: np.ndarray, force: bool = False) -> list[FaceResult]:
        if not self.enabled:
            return []

        self._frame_counter += 1
        if not force and (self._frame_counter % self.scan_interval != 0):
            return self._last_results

        if FACE_REC_AVAILABLE:
            results = self._identify_with_face_recognition(frame)
        else:
            results = self._identify_with_opencv(frame)

        self._update_adaptive_interval(results)

        with self._lock:
            self._last_results = results
        return results

    def _identify_with_face_recognition(self, frame: np.ndarray) -> list[FaceResult]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ds = self._downscale
        small = cv2.resize(rgb, (0, 0), fx=ds, fy=ds)
        inv_ds = 1.0 / ds

        face_locations = face_recognition.face_locations(small, model="hog")
        face_encodings = face_recognition.face_encodings(small, face_locations)

        results = []
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            top = int(top * inv_ds)
            right = int(right * inv_ds)
            bottom = int(bottom * inv_ds)
            left = int(left * inv_ds)

            if not self._is_face_quality_ok(frame, (left, top, right, bottom)):
                continue

            name, threat, confidence = self._match_with_voting(encoding)

            results.append(FaceResult(
                bbox=(left, top, right, bottom),
                name=name,
                threat_level=threat,
                confidence=confidence,
                encoding=encoding,
            ))

        return results

    def _identify_with_opencv(self, frame: np.ndarray) -> list[FaceResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, 1.3, 5)

        results = []
        for (x, y, w, h) in faces:
            if w < self.min_face_size or h < self.min_face_size:
                continue
            threat = "hostile" if self.unknown_is_hostile else "unknown"
            results.append(FaceResult(
                bbox=(x, y, x + w, y + h),
                name="Unknown",
                threat_level=threat,
                confidence=0.0,
            ))
        return results

    def add_face(self, frame: np.ndarray, name: str, threat_level: str) -> bool:
        """Enroll a face from the current frame into the database."""
        if not FACE_REC_AVAILABLE:
            logger.error("face_recognition library required to add faces")
            return False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")

        if not locations:
            logger.warning("No face detected in frame for enrollment")
            return False

        largest = max(locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
        top, right, bottom, left = largest

        # Save face image
        target_dir = self.safe_dir if threat_level == "safe" else self.hostile_dir
        person_dir = target_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)

        existing = list(person_dir.glob("*.jpg"))
        img_num = len(existing) + 1
        img_path = person_dir / f"{name}_{img_num:03d}.jpg"

        face_crop = frame[top:bottom, left:right]
        cv2.imwrite(str(img_path), face_crop)
        logger.info(f"Saved face image: {img_path}")

        # Compute quality score
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        quality = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Encode and store in SQLite
        encodings = face_recognition.face_encodings(rgb, [largest])
        if encodings:
            self._db.add_face(
                name=name,
                threat_level=threat_level,
                encoding=encodings[0],
                image_path=str(img_path),
                source="webcam",
                quality=quality,
            )
            self._reload_cache()

        return True

    def enroll_multi_frame(self, frame: np.ndarray, name: str, threat_level: str) -> dict:
        """
        Multi-frame enrollment: call repeatedly to capture multiple angles.
        Returns {"captured": N, "target": M, "complete": bool, "quality": float}.
        """
        if not FACE_REC_AVAILABLE:
            return {"captured": 0, "target": self._enroll_target_frames, "complete": False, "error": "face_recognition not available"}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")

        if not locations:
            return {
                "captured": len(self._enroll_buffer),
                "target": self._enroll_target_frames,
                "complete": False,
                "error": "No face detected",
            }

        largest = max(locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
        top, right, bottom, left = largest

        face_crop = frame[top:bottom, left:right]
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        quality = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if quality < self.blur_threshold:
            return {
                "captured": len(self._enroll_buffer),
                "target": self._enroll_target_frames,
                "complete": False,
                "error": f"Face too blurry (quality={quality:.0f}, need {self.blur_threshold:.0f})",
            }

        encodings = face_recognition.face_encodings(rgb, [largest])
        if not encodings:
            return {
                "captured": len(self._enroll_buffer),
                "target": self._enroll_target_frames,
                "complete": False,
                "error": "Could not encode face",
            }

        self._enroll_buffer.append({
            "encoding": encodings[0],
            "crop": face_crop.copy(),
            "quality": quality,
            "location": largest,
        })

        captured = len(self._enroll_buffer)
        complete = captured >= self._enroll_target_frames

        if complete:
            self._finalize_enrollment(name, threat_level)

        return {
            "captured": captured,
            "target": self._enroll_target_frames,
            "complete": complete,
            "quality": quality,
        }

    def _finalize_enrollment(self, name: str, threat_level: str):
        """Save all buffered enrollment frames to the database."""
        target_dir = self.safe_dir if threat_level == "safe" else self.hostile_dir
        person_dir = target_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)

        existing = list(person_dir.glob("*.jpg"))
        img_num = len(existing) + 1

        for entry in self._enroll_buffer:
            img_path = person_dir / f"{name}_{img_num:03d}.jpg"
            cv2.imwrite(str(img_path), entry["crop"])

            self._db.add_face(
                name=name,
                threat_level=threat_level,
                encoding=entry["encoding"],
                image_path=str(img_path),
                source="enrollment",
                quality=entry["quality"],
            )
            img_num += 1

        logger.info(f"Enrolled '{name}' ({threat_level}) with {len(self._enroll_buffer)} frames")
        self._enroll_buffer.clear()
        self._reload_cache()

    def cancel_enrollment(self):
        """Discard any buffered enrollment frames."""
        self._enroll_buffer.clear()

    # ── Face CRUD (delegated to database) ─────────────────

    def list_people(self) -> list[dict]:
        """List unique people with encoding counts."""
        return self._db.list_people()

    def delete_person(self, name: str) -> int:
        """Delete all encodings for a person. Returns count deleted."""
        count = self._db.delete_person(name)
        if count > 0:
            self._reload_cache()
        return count

    def delete_face_by_id(self, face_id: int) -> bool:
        """Delete a single face encoding by ID."""
        result = self._db.delete_face(face_id)
        if result:
            self._reload_cache()
        return result

    def reclassify_person(self, name: str, new_threat_level: str) -> int:
        """Change threat level for all encodings of a person."""
        count = self._db.reclassify_person(name, new_threat_level)
        if count > 0:
            self._reload_cache()
        return count

    def get_face_thumbnail_path(self, face_id: int) -> Optional[str]:
        """Get the image path for a face record."""
        record = self._db.get_by_id(face_id)
        return record.image_path if record else None

    def rebuild_database(self):
        """Clear the database and re-scan image directories."""
        # Delete all records from SQLite
        for person in self._db.list_people():
            self._db.delete_person(person["name"])
        self._build_database()
        self._reload_cache()

    @property
    def stats(self) -> dict:
        return self._db.stats()

    @property
    def database(self) -> FaceDatabase:
        """Expose the underlying database for direct queries."""
        return self._db

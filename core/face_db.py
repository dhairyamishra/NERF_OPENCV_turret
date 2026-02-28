"""
SQLite-backed face database for persistent, queryable face storage.
Replaces the flat pickle-based approach with full CRUD operations,
metadata tracking, and migration support from legacy encodings.pkl.
"""

import io
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DB_PATH = Path("data/faces/faces.db")
LEGACY_PKL_PATH = Path("data/faces/encodings.pkl")


@dataclass
class FaceRecord:
    """A single face entry from the database."""
    id: int
    name: str
    threat_level: str
    encoding: np.ndarray
    image_path: Optional[str]
    created_at: float
    updated_at: float
    source: str
    quality: float


def _encode_numpy(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to bytes for BLOB storage."""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _decode_numpy(data: bytes) -> np.ndarray:
    """Deserialize bytes back to a numpy array."""
    buf = io.BytesIO(data)
    return np.load(buf, allow_pickle=False)


class FaceDatabase:
    """
    Thread-safe SQLite face database with full CRUD.
    Stores 128-d face encodings as BLOBs alongside metadata.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def open(self):
        """Open the database connection and ensure schema exists."""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        logger.info(f"Face database opened: {self._db_path}")

    def _create_schema(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS faces (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT NOT NULL,
                    threat_level TEXT NOT NULL CHECK(threat_level IN ('safe', 'hostile')),
                    encoding    BLOB NOT NULL,
                    image_path  TEXT,
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL,
                    source      TEXT NOT NULL DEFAULT 'import',
                    quality     REAL NOT NULL DEFAULT 0.0
                );
                CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name);
                CREATE INDEX IF NOT EXISTS idx_faces_threat ON faces(threat_level);
            """)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Create ─────────────────────────────────────────────

    def add_face(
        self,
        name: str,
        threat_level: str,
        encoding: np.ndarray,
        image_path: Optional[str] = None,
        source: str = "import",
        quality: float = 0.0,
    ) -> int:
        """Insert a face record. Returns the new row ID."""
        now = time.time()
        blob = _encode_numpy(encoding)
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO faces (name, threat_level, encoding, image_path,
                   created_at, updated_at, source, quality)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (name, threat_level, blob, image_path, now, now, source, quality),
            )
            self._conn.commit()
            return cur.lastrowid

    def add_faces_bulk(self, records: list[tuple]) -> int:
        """Bulk insert. Each tuple: (name, threat_level, encoding, image_path, source, quality)."""
        now = time.time()
        rows = []
        for name, threat_level, encoding, image_path, source, quality in records:
            blob = _encode_numpy(encoding)
            rows.append((name, threat_level, blob, image_path, now, now, source, quality))
        with self._lock:
            self._conn.executemany(
                """INSERT INTO faces (name, threat_level, encoding, image_path,
                   created_at, updated_at, source, quality)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            self._conn.commit()
            return len(rows)

    # ── Read ───────────────────────────────────────────────

    def get_all(self) -> list[FaceRecord]:
        """Return all face records."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, threat_level, encoding, image_path, created_at, updated_at, source, quality FROM faces"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_by_id(self, face_id: int) -> Optional[FaceRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, threat_level, encoding, image_path, created_at, updated_at, source, quality FROM faces WHERE id=?",
                (face_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_name(self, name: str) -> list[FaceRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, threat_level, encoding, image_path, created_at, updated_at, source, quality FROM faces WHERE name=?",
                (name,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_all_encodings(self) -> tuple[list[np.ndarray], list[str], list[str], list[int]]:
        """Return parallel lists of (encodings, names, threat_levels, ids) for matching."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, threat_level, encoding FROM faces"
            ).fetchall()
        ids, names, threats, encodings = [], [], [], []
        for row in rows:
            ids.append(row[0])
            names.append(row[1])
            threats.append(row[2])
            encodings.append(_decode_numpy(row[3]))
        return encodings, names, threats, ids

    def list_people(self) -> list[dict]:
        """Return a summary list of unique people with counts and threat levels."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT name, threat_level, COUNT(*) as count,
                   MIN(image_path) as sample_image, MIN(created_at) as first_seen
                   FROM faces GROUP BY name, threat_level ORDER BY name"""
            ).fetchall()
        return [
            {
                "name": r[0],
                "threat_level": r[1],
                "encoding_count": r[2],
                "sample_image": r[3],
                "first_seen": r[4],
            }
            for r in rows
        ]

    # ── Update ─────────────────────────────────────────────

    def update_face(self, face_id: int, name: Optional[str] = None,
                    threat_level: Optional[str] = None) -> bool:
        """Update name and/or threat_level for a single face record."""
        fields, values = [], []
        if name is not None:
            fields.append("name=?")
            values.append(name)
        if threat_level is not None:
            if threat_level not in ("safe", "hostile"):
                return False
            fields.append("threat_level=?")
            values.append(threat_level)
        if not fields:
            return False
        fields.append("updated_at=?")
        values.append(time.time())
        values.append(face_id)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE faces SET {', '.join(fields)} WHERE id=?", values
            )
            self._conn.commit()
            return cur.rowcount > 0

    def reclassify_person(self, name: str, new_threat_level: str) -> int:
        """Change threat_level for ALL encodings of a named person."""
        if new_threat_level not in ("safe", "hostile"):
            return 0
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE faces SET threat_level=?, updated_at=? WHERE name=?",
                (new_threat_level, now, name),
            )
            self._conn.commit()
            return cur.rowcount

    # ── Delete ─────────────────────────────────────────────

    def delete_face(self, face_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM faces WHERE id=?", (face_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def delete_person(self, name: str) -> int:
        """Delete ALL encodings for a named person. Returns count deleted."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM faces WHERE name=?", (name,))
            self._conn.commit()
            return cur.rowcount

    # ── Stats ──────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
            safe = self._conn.execute("SELECT COUNT(*) FROM faces WHERE threat_level='safe'").fetchone()[0]
            hostile = self._conn.execute("SELECT COUNT(*) FROM faces WHERE threat_level='hostile'").fetchone()[0]
            unique_safe = self._conn.execute(
                "SELECT COUNT(DISTINCT name) FROM faces WHERE threat_level='safe'"
            ).fetchone()[0]
            unique_hostile = self._conn.execute(
                "SELECT COUNT(DISTINCT name) FROM faces WHERE threat_level='hostile'"
            ).fetchone()[0]
        return {
            "total_encodings": total,
            "safe_encodings": safe,
            "hostile_encodings": hostile,
            "unique_safe": unique_safe,
            "unique_hostile": unique_hostile,
        }

    # ── Migration ──────────────────────────────────────────

    def migrate_from_pickle(self, pkl_path: Optional[Path] = None) -> int:
        """Import legacy encodings.pkl into the database. Returns count imported."""
        import pickle
        pkl = pkl_path or LEGACY_PKL_PATH
        if not pkl.exists():
            return 0

        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            encodings = data.get("encodings", [])
            names = data.get("names", [])
            threats = data.get("threats", [])
        except Exception as e:
            logger.error(f"Failed to load legacy pickle: {e}")
            return 0

        if not encodings:
            return 0

        records = []
        for enc, name, threat in zip(encodings, names, threats):
            records.append((name, threat, enc, None, "migrated_pkl", 0.0))

        count = self.add_faces_bulk(records)
        logger.info(f"Migrated {count} face encodings from {pkl}")

        # Rename old pickle so we don't re-import
        backup = pkl.with_suffix(".pkl.bak")
        pkl.rename(backup)
        logger.info(f"Legacy pickle backed up to {backup}")

        return count

    # ── Internal ───────────────────────────────────────────

    @staticmethod
    def _row_to_record(row: tuple) -> FaceRecord:
        return FaceRecord(
            id=row[0],
            name=row[1],
            threat_level=row[2],
            encoding=_decode_numpy(row[3]),
            image_path=row[4],
            created_at=row[5],
            updated_at=row[6],
            source=row[7],
            quality=row[8],
        )

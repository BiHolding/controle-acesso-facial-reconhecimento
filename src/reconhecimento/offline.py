"""Resiliência local para operação rápida quando a internet oscila."""

from __future__ import annotations

import ctypes
import io
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from reconhecimento.api.client import ApiUnavailableError, RecognitionResult
from reconhecimento.recognition.matcher import InMemoryFaceIndex


def _data_dir() -> Path:
    root = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(Path.home())
    return Path(root) / "VIPISPEvolution"


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _dpapi_transform(data: bytes, protect: bool) -> bytes:
    if os.name != "nt":
        raise RuntimeError("O cache biométrico persistente requer Windows DPAPI")
    buffer = ctypes.create_string_buffer(data)
    source = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    destination = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    function = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    description = ctypes.c_wchar_p("VIP ISP Evolution face cache") if protect else None
    if not function(
        ctypes.byref(source), description, None, None, None, 0,
        ctypes.byref(destination),
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(destination.pbData, destination.cbData)
    finally:
        kernel32.LocalFree(destination.pbData)


class EncryptedFaceCache:
    """Snapshot biométrico criptografado para o usuário do Windows atual."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        protect: Callable[[bytes], bytes] | None = None,
        unprotect: Callable[[bytes], bytes] | None = None,
    ) -> None:
        self.path = Path(path) if path else _data_dir() / "face-cache.dpapi"
        self._protect = protect or (lambda value: _dpapi_transform(value, True))
        self._unprotect = unprotect or (lambda value: _dpapi_transform(value, False))

    def save(self, eligible: dict[str, tuple[np.ndarray, str]]) -> None:
        guest_ids = list(eligible)
        embeddings = (
            np.stack([eligible[guest_id][0] for guest_id in guest_ids]).astype(np.float32)
            if guest_ids else np.empty((0, 512), dtype=np.float32)
        )
        names = np.asarray([eligible[guest_id][1] for guest_id in guest_ids], dtype="U160")
        payload = io.BytesIO()
        np.savez_compressed(
            payload,
            version=np.asarray([1], dtype=np.int16),
            guest_ids=np.asarray(guest_ids, dtype="U32"),
            names=names,
            embeddings=embeddings,
        )
        encrypted = self._protect(payload.getvalue())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(encrypted)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self) -> dict[str, tuple[np.ndarray, str]]:
        if not self.path.exists():
            return {}
        try:
            plain = self._unprotect(self.path.read_bytes())
            with np.load(io.BytesIO(plain), allow_pickle=False) as archive:
                if int(archive["version"][0]) != 1:
                    return {}
                guest_ids = archive["guest_ids"]
                names = archive["names"]
                embeddings = archive["embeddings"]
                if embeddings.shape != (len(guest_ids), 512) or len(names) != len(guest_ids):
                    return {}
                return {
                    str(guest_id): (np.asarray(embedding, dtype=np.float32), str(name))
                    for guest_id, name, embedding in zip(guest_ids, names, embeddings, strict=True)
                }
        except (OSError, ValueError, KeyError, RuntimeError):
            return {}


@dataclass(frozen=True)
class OfflineDecision:
    allowed: bool
    reason: str


class OfflineAccessStore:
    """Ocupação e fila compartilhadas pelas duas estações via SQLite/WAL."""

    def __init__(self, path: str | Path | None = None, maximum_capacity: int = 400) -> None:
        self.path = Path(path) if path else _data_dir() / "offline-access.sqlite3"
        self.maximum_capacity = maximum_capacity
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS local_presence (
                    participant_key TEXT PRIMARY KEY,
                    inside INTEGER NOT NULL CHECK (inside IN (0, 1)),
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pending_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    participant_key TEXT NOT NULL,
                    access_point TEXT NOT NULL,
                    direction TEXT NOT NULL CHECK (direction IN ('ENTRY', 'EXIT')),
                    occurred_at TEXT NOT NULL
                );
            """)

    def record_offline(self, guest_id: str, access_point: str, direction: str) -> OfflineDecision:
        participant_key = f"GUEST:{guest_id}"
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT inside FROM local_presence WHERE participant_key = ?",
                (participant_key,),
            ).fetchone()
            inside = bool(row["inside"]) if row is not None else False
            if direction == "ENTRY":
                if inside:
                    return OfflineDecision(False, "DUPLICATE_ENTRY")
                occupied = int(connection.execute(
                    "SELECT COUNT(*) FROM local_presence WHERE inside = 1"
                ).fetchone()[0])
                if occupied >= self.maximum_capacity:
                    return OfflineDecision(False, "CAPACITY_FULL")
                next_inside = 1
            elif direction == "EXIT":
                if not inside:
                    return OfflineDecision(False, "DUPLICATE_EXIT")
                next_inside = 0
            else:
                return OfflineDecision(False, "ACCESS_POINT_UNAVAILABLE")

            connection.execute(
                "INSERT INTO local_presence(participant_key, inside, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(participant_key) DO UPDATE SET inside=excluded.inside, updated_at=excluded.updated_at",
                (participant_key, next_inside, now),
            )
            connection.execute(
                "INSERT INTO pending_events(event_id, participant_key, access_point, direction, occurred_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), participant_key, access_point, direction, now),
            )
        return OfflineDecision(True, "OFFLINE_AUTHORIZED")

    def reconcile_online(self, result: RecognitionResult) -> None:
        if result.guest_id is None or result.direction not in {"ENTRY", "EXIT"}:
            return
        inside: int | None = None
        if result.allowed:
            inside = 1 if result.direction == "ENTRY" else 0
        elif result.reason == "DUPLICATE_ENTRY":
            inside = 1
        elif result.reason == "DUPLICATE_EXIT":
            inside = 0
        if inside is None:
            return
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO local_presence(participant_key, inside, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(participant_key) DO UPDATE SET inside=excluded.inside, updated_at=excluded.updated_at",
                (f"GUEST:{result.guest_id}", inside, datetime.now(timezone.utc).isoformat()),
            )

    def next_event(self) -> sqlite3.Row | None:
        with self._connection() as connection:
            return connection.execute(
                "SELECT * FROM pending_events ORDER BY sequence ASC LIMIT 1"
            ).fetchone()

    def complete_event(self, event_id: str) -> None:
        with self._lock, self._connection() as connection:
            connection.execute("DELETE FROM pending_events WHERE event_id = ?", (event_id,))

    def pending_count(self) -> int:
        with self._connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM pending_events").fetchone()[0])


class HybridRecognitionService:
    """API como autoridade; cache local como continuidade em queda de conexão."""

    def __init__(
        self,
        online_client,
        index: InMemoryFaceIndex,
        names: Callable[[], dict[str, str]],
        store: OfflineAccessStore,
        direction: str,
        access_point: str,
    ) -> None:
        self.online_client = online_client
        self.index = index
        self.names = names
        self.store = store
        self.direction = direction
        self.access_point = access_point

    def recognize(self, embedding: np.ndarray) -> RecognitionResult:
        match = self.index.match(embedding)
        # Preserva a ordem global: enquanto existir fila, as duas estações
        # continuam em contingência até a estação principal concluir o replay.
        if self.store.pending_count() > 0:
            return self._offline_result(match)
        try:
            result = self.online_client.recognize(embedding)
            self.store.reconcile_online(result)
            return result
        except ApiUnavailableError:
            return self._offline_result(match)

    def _offline_result(self, match) -> RecognitionResult:
        if match is None:
            return RecognitionResult(False, False, "SERVICE_UNAVAILABLE")
        decision = self.store.record_offline(match.guest_id, self.access_point, self.direction)
        return RecognitionResult(
            recognized=True,
            allowed=decision.allowed,
            reason=decision.reason,
            guest_id=match.guest_id,
            participant_type="guest",
            participant_id=match.guest_id,
            name=self.names().get(match.guest_id),
            similarity=match.similarity,
            direction=self.direction,
        )


class OfflineReplayThread(threading.Thread):
    """Reenvia a fila na ordem original quando a API volta a responder."""

    def __init__(self, store: OfflineAccessStore, cache: EncryptedFaceCache, clients: dict[str, object]) -> None:
        super().__init__(daemon=True, name="offline-replay")
        self.store = store
        self.cache = cache
        self.clients = clients
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            event = self.store.next_event()
            if event is None:
                self._stop_event.wait(2.0)
                continue
            guest_id = str(event["participant_key"]).split(":", 1)[1]
            cached = self.cache.load().get(guest_id)
            client = self.clients.get(str(event["access_point"]))
            if cached is None or client is None:
                self._stop_event.wait(5.0)
                continue
            try:
                result = client.recognize(cached[0])
            except ApiUnavailableError:
                self._stop_event.wait(2.0)
                continue
            if result.allowed or result.reason in {
                "DUPLICATE_ENTRY", "DUPLICATE_EXIT", "GUEST_NOT_ELIGIBLE",
                "CLIENT_INACTIVE", "REFERENCE_STALE", "NOT_AUTHORIZED",
            }:
                self.store.reconcile_online(result)
                self.store.complete_event(str(event["event_id"]))
            else:
                self._stop_event.wait(5.0)

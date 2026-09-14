"""Repositório MySQL para reconhecimento facial local-first.

Lê embeddings elegíveis, revalida guests/clients e registra eventos de acesso.
BLOB armazenado no formato PHP pack('g*') = float32 little-endian nativo (2048 bytes).
"""

import json
import logging
import os
import struct
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pymysql
from pymysql.cursors import DictCursor
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_DIMENSION = 512
EMBEDDING_BYTE_LENGTH = 2048
LOCAL_ACCESS_LOG = os.getenv(
    "LOCAL_ACCESS_LOG",
    os.path.join(os.path.dirname(__file__), "..", "..", "access_events.jsonl"),
)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EligibleEmbedding:
    guest_id: str
    name: str
    embedding: np.ndarray
    photo_checksum: str


@dataclass(frozen=True)
class RevalidationResult:
    eligible: bool
    guest_id: str
    name: str


@dataclass(frozen=True)
class AccessEventResult:
    logged: bool


class FaceDatabaseError(RuntimeError):
    """Falha de conexão ou consulta ao banco de dados."""


class FaceRepository:
    """Repositório MySQL para operações de reconhecimento facial.

    Usa pymysql com connection pooling via ThreadSafePool.
    Todas as operações são fail-closed: erro de conexão = resultado negativo.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        device_id: int | None = None,
        access_point: str | None = None,
        model: str = "buffalo_l",
        model_version: str = "",
        dimension: int = EMBEDDING_DIMENSION,
        normalization: str = "l2",
    ) -> None:
        self.host = host or os.getenv("DB_HOST", "127.0.0.1")
        self.port = int(port or os.getenv("DB_PORT", "3306"))
        self.database = database or os.getenv("DB_NAME", "ispevolution_p")
        self.user = user or os.getenv("DB_USER", "")
        self.password = password or os.getenv("DB_PASSWORD", "")
        self.device_id = device_id or int(os.getenv("DEVICE_ID", "0"))
        self.access_point = access_point or os.getenv("ACCESS_POINT", "vip_room")
        self.model = model
        self.model_version = model_version or os.getenv("FACE_MODEL_VERSION", "")
        self.dimension = dimension
        self.normalization = normalization

        self._lock = threading.Lock()
        self._connection: pymysql.Connection | None = None

    def _get_connection(self) -> pymysql.Connection:
        """Retorna conexão existente ou cria uma nova (thread-safe)."""
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.ping(reconnect=True)
                    return self._connection
                except pymysql.Error:
                    self._connection = None

            try:
                self._connection = pymysql.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    charset="utf8mb4",
                    cursorclass=DictCursor,
                    autocommit=True,
                    connect_timeout=5,
                    read_timeout=10,
                    write_timeout=10,
                )
                return self._connection
            except pymysql.Error as exc:
                self._connection = None
                raise FaceDatabaseError(f"Falha ao conectar ao MySQL: {exc}") from exc

    def close(self) -> None:
        """Fecha a conexão ativa."""
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                except pymysql.Error:
                    pass
                self._connection = None

    def load_eligible_embeddings(self) -> dict[str, tuple[np.ndarray, str]]:
        """Carrega embeddings elegíveis: guest completed + client active.

        Returns:
            dict[guest_id] = (embedding_512_float32, guest_name)
        """
        sql = """
            SELECT gfe.participant_type, gfe.participant_id,
                gfe.embedding, gfe.photo_checksum, g.name AS participant_name
            FROM participant_face_embeddings gfe
            JOIN guests g ON g.id = gfe.participant_id
            JOIN clients c ON c.id = g.client_id
            WHERE gfe.active = 1
              AND gfe.participant_type = 'GUEST'
              AND gfe.model = %s
              AND gfe.dimension = %s
              AND gfe.normalization = %s
              AND g.status = 'completed'
              AND c.status = 'active'
            UNION ALL
            SELECT gfe.participant_type, gfe.participant_id,
                gfe.embedding, gfe.photo_checksum, u.name AS participant_name
            FROM participant_face_embeddings gfe
            JOIN users u ON u.id = gfe.participant_id
            JOIN clients c ON c.id = u.client_id
            WHERE gfe.active = 1 AND gfe.participant_type = 'CLIENT'
              AND gfe.model = %s AND gfe.dimension = %s AND gfe.normalization = %s
              AND u.role = 'CLIENT' AND u.active = 1 AND c.status = 'active'
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(sql, (self.model, self.dimension, self.normalization, self.model, self.dimension, self.normalization))
                rows = cursor.fetchall()
        except (pymysql.Error, FaceDatabaseError) as exc:
            raise FaceDatabaseError(f"Falha ao carregar embeddings: {exc}") from exc

        result: dict[str, tuple[np.ndarray, str]] = {}
        for row in rows:
            participant_id = row.get("participant_id", row.get("guest_id"))
            guest_id = (
                f"{row['participant_type']}:{participant_id}"
                if "participant_type" in row else str(participant_id)
            )
            blob = row["embedding"]
            name = row.get("participant_name", row.get("guest_name", ""))
            embedding = self._decode_blob(blob)
            if embedding is not None:
                result[guest_id] = (embedding, name)
        return result

    def revalidate_guest(self, guest_id: str) -> tuple[bool, str]:
        """Revalida se o guest está elegível (completed + client active).

        Returns:
            (eligible, guest_name) — se não elegível, name pode ser vazio.
        """
        sql = """
            SELECT g.name, g.status AS guest_status, c.status AS client_status
            FROM guests g
            JOIN clients c ON c.id = g.client_id
            WHERE g.id = %s
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(sql, (guest_id,))
                row = cursor.fetchone()
        except (pymysql.Error, FaceDatabaseError) as exc:
            raise FaceDatabaseError(f"Falha ao revalidar guest: {exc}") from exc

        if row is None:
            return False, ""
        if row["guest_status"] != "completed" or row["client_status"] != "active":
            return False, ""
        return True, row["name"]

    def revalidate_embedding(
        self, guest_id: str, current_photo_checksum: str
    ) -> bool:
        """Verifica se o embedding do guest ainda está ativo e consistente.

        Returns:
            True se o embedding está válido e ativo.
        """
        sql = """
            SELECT gfe.active, gfe.photo_checksum
            FROM participant_face_embeddings gfe
            WHERE gfe.participant_type = 'GUEST'
              AND gfe.participant_id = %s
              AND gfe.active = 1
              AND gfe.model = %s
              AND gfe.dimension = %s
              AND gfe.normalization = %s
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (guest_id, self.model, self.dimension, self.normalization),
                )
                row = cursor.fetchone()
        except (pymysql.Error, FaceDatabaseError) as exc:
            raise FaceDatabaseError(
                f"Falha ao revalidar embedding: {exc}"
            ) from exc

        if row is None:
            return False
        return row["photo_checksum"] == current_photo_checksum

    def log_access_event(
        self,
        guest_id: str | None,
        recognized: bool,
        allowed: bool,
        similarity: float | None,
        reason: str,
    ) -> AccessEventResult:
        """Registra evento de acesso na tabela access_events.

        Quando DEVICE_ID=0 (modo local), registra em arquivo JSON local
        porque a tabela access_events exige device_id NOT NULL com FK.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        if self.device_id == 0:
            return self._log_access_event_local(
                guest_id, recognized, allowed, similarity, reason, now
            )

        sql = """
            INSERT INTO access_events
                (guest_id, device_id, access_point, recognized, allowed,
                 similarity, reason, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (
                        guest_id,
                        self.device_id,
                        self.access_point,
                        int(recognized),
                        int(allowed),
                        similarity,
                        reason,
                        now,
                    ),
                )
            return AccessEventResult(logged=True)
        except (pymysql.Error, FaceDatabaseError) as exc:
            # Fail-closed: log mas nao impede decisao
            _logger.warning("Falha ao registrar access_event no MySQL: %s", exc)
            return AccessEventResult(logged=False)

    def _log_access_event_local(
        self,
        guest_id: str | None,
        recognized: bool,
        allowed: bool,
        similarity: float | None,
        reason: str,
        timestamp: str,
    ) -> AccessEventResult:
        """Registra evento de acesso em arquivo JSON local (modo local-first).

        Usado quando DEVICE_ID=0 e a tabela access_events nao aceita
        device_id=NULL (FK NOT NULL constraint).
        Armazena apenas campos seguros — sem PII, sem embeddings, sem fotos.
        """
        event = {
            "guest_id": guest_id,
            "access_point": self.access_point,
            "recognized": recognized,
            "allowed": allowed,
            "similarity": similarity,
            "reason": reason,
            "created_at": timestamp,
        }
        try:
            log_path = os.path.abspath(LOCAL_ACCESS_LOG)
            log_dir = os.path.dirname(log_path)
            os.makedirs(log_dir, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            return AccessEventResult(logged=True)
        except OSError as exc:
            _logger.warning("Falha ao registrar access_event local: %s", exc)
            return AccessEventResult(logged=False)

    def load_checksums(self) -> dict[str, str]:
        """Carrega checksums de todos os embeddings ativos.

        Returns:
            dict[guest_id] = photo_checksum
        """
        sql = """
            SELECT participant_id AS guest_id, photo_checksum
            FROM participant_face_embeddings
            WHERE active = 1
              AND participant_type = 'GUEST'
              AND model = %s
              AND dimension = %s
              AND normalization = %s
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(sql, (self.model, self.dimension, self.normalization))
                rows = cursor.fetchall()
            return {str(row["guest_id"]): row["photo_checksum"] for row in rows}
        except (pymysql.Error, FaceDatabaseError) as exc:
            raise FaceDatabaseError(f"Falha ao carregar checksums: {exc}") from exc

    # ── Enrollment methods ────────────────────────────────────────────────

    def find_pending_enrollment(self) -> list[dict]:
        """Encontra Guests pendentes de enrollment.

        Critérios:
        - guest.status = completed
        - client.status = active
        - photo_reference IS NOT NULL
        - Não existe participant_face_embeddings ativo para este guest

        Returns:
            Lista de dicts com guest_id e photo_reference.
        """
        sql = """
            SELECT 'GUEST' AS participant_type, g.id AS participant_id, g.photo_reference
            FROM guests g
            JOIN clients c ON c.id = g.client_id
            WHERE g.status = 'completed'
              AND c.status = 'active'
              AND g.photo_reference IS NOT NULL
              AND g.photo_reference != ''
              AND NOT EXISTS (
                  SELECT 1
                  FROM participant_face_embeddings gfe
                  WHERE gfe.participant_type = 'GUEST'
                    AND gfe.participant_id = g.id
                    AND gfe.active = 1
                    AND gfe.model = %s
                    AND gfe.dimension = %s
                    AND gfe.normalization = %s
              )
            UNION ALL
            SELECT 'CLIENT' AS participant_type, u.id AS participant_id, u.photo_reference
            FROM users u JOIN clients c ON c.id = u.client_id
            WHERE u.role = 'CLIENT' AND u.active = 1 AND c.status = 'active'
              AND u.photo_reference IS NOT NULL AND u.photo_reference != ''
              AND NOT EXISTS (
                  SELECT 1 FROM participant_face_embeddings pfe
                  WHERE pfe.participant_type = 'CLIENT' AND pfe.participant_id = u.id
                    AND pfe.active = 1 AND pfe.model = %s AND pfe.dimension = %s AND pfe.normalization = %s
              )
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(sql, (self.model, self.dimension, self.normalization, self.model, self.dimension, self.normalization))
                rows = cursor.fetchall()
            return [
                ({"participant_type": row["participant_type"], "participant_id": str(row["participant_id"]), "photo_reference": row["photo_reference"]}
                 if "participant_type" in row else {"guest_id": str(row["guest_id"]), "photo_reference": row["photo_reference"]})
                for row in rows
            ]
        except (pymysql.Error, FaceDatabaseError) as exc:
            raise FaceDatabaseError(f"Falha ao buscar guests pendentes: {exc}") from exc

    def find_embedding_by_guest(self, guest_id: str) -> dict | None:
        """Busca embedding existente para um guest.

        Returns:
            Dict com id, photo_checksum, revision ou None se não existe.
        """
        sql = """
            SELECT id, photo_checksum, revision
            FROM participant_face_embeddings
            WHERE participant_type = 'GUEST'
              AND participant_id = %s
              AND active = 1
              AND model = %s
              AND dimension = %s
              AND normalization = %s
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    sql, (guest_id, self.model, self.dimension, self.normalization)
                )
                row = cursor.fetchone()
            if row is None:
                return None
            return {
                "id": row["id"],
                "photo_checksum": row["photo_checksum"],
                "revision": row["revision"] or 0,
            }
        except (pymysql.Error, FaceDatabaseError) as exc:
            raise FaceDatabaseError(
                f"Falha ao buscar embedding do guest {guest_id}: {exc}"
            ) from exc

    def upsert_embedding(
        self,
        guest_id: str,
        embedding: np.ndarray,
        photo_checksum: str,
        revision: int = 0,
    ) -> None:
        """Insere ou atualiza embedding de um guest.

        Se já existe embedding ativo para o guest, atualiza.
        Se não existe, insere novo.
        """
        # Encode embedding para BLOB
        blob = embedding.astype(np.float32).tobytes()
        if len(blob) != EMBEDDING_BYTE_LENGTH:
            raise FaceDatabaseError(
                f"BLOB inválido: {len(blob)} bytes, esperado {EMBEDDING_BYTE_LENGTH}"
            )

        # Verifica se já existe
        existing = self.find_embedding_by_guest(guest_id)

        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                if existing is not None:
                    # Update existente
                    sql = """
                        UPDATE participant_face_embeddings
                        SET embedding = %s,
                            photo_checksum = %s,
                            revision = %s,
                            updated_at = NOW()
                        WHERE id = %s
                    """
                    cursor.execute(sql, (blob, photo_checksum, revision, existing["id"]))
                else:
                    # Insert novo
                    sql = """
                        INSERT INTO participant_face_embeddings
                            (participant_type, participant_id, model, model_version, dimension, normalization,
                             embedding, photo_checksum, active, revision,
                             created_at, updated_at)
                        VALUES ('GUEST', %s, %s, %s, %s, %s, %s, %s, 1, %s, NOW(), NOW())
                    """
                    cursor.execute(
                        sql,
                        (
                            guest_id,
                            self.model,
                            self.model_version,
                            self.dimension,
                            self.normalization,
                            blob,
                            photo_checksum,
                            revision,
                        ),
                    )
        except (pymysql.Error, FaceDatabaseError) as exc:
            raise FaceDatabaseError(
                f"Falha ao persistir embedding do guest {guest_id}: {exc}"
            ) from exc

    def _decode_blob(self, blob: bytes | None) -> np.ndarray | None:
        """Decodifica BLOB PHP pack('g*') em numpy float32 array.

        PHP pack('g*') armazena float32 little-endian nativo.
        """
        if blob is None or len(blob) != EMBEDDING_BYTE_LENGTH:
            return None
        try:
            vector = np.frombuffer(blob, dtype=np.float32).copy()
            if vector.shape != (EMBEDDING_DIMENSION,):
                return None
            if not np.all(np.isfinite(vector)):
                return None
            norm = float(np.linalg.norm(vector))
            if norm <= np.finfo(np.float32).eps:
                return None
            return vector / norm
        except Exception:
            return None

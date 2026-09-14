"""Cliente M2M para a API de controle de acesso do Portal VIP."""

import math
import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_DIMENSION = 512
_ACCESS_POINT_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TIMEOUT = httpx.Timeout(
    connect=float(os.getenv("API_CONNECT_TIMEOUT_SECONDS", "0.6")),
    read=float(os.getenv("API_READ_TIMEOUT_SECONDS", "1.2")),
    write=2.0,
    pool=2.0,
)


class RecognitionApiError(RuntimeError):
    """Falha que impede uma decisão de acesso confiável."""


class ApiAuthenticationError(RecognitionApiError):
    """A credencial do dispositivo foi rejeitada."""


class ApiUnavailableError(RecognitionApiError):
    """A API não respondeu com sucesso dentro do contrato esperado."""


class ApiResponseError(RecognitionApiError):
    """A API respondeu fora do contrato esperado."""


@dataclass(frozen=True)
class RecognitionResult:
    recognized: bool
    allowed: bool
    reason: str
    guest_id: str | None = None
    user_id: str | None = None
    participant_type: str | None = None
    participant_id: str | None = None
    name: str | None = None
    similarity: float | None = None
    direction: str | None = None


class AccessControlClient:
    """Cliente persistente, sem retries automáticos, para reconhecimento online."""

    def __init__(
        self,
        api_url: str,
        device_key: str,
        access_point: str,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.device_key = device_key
        self.access_point = access_point
        self._validate_configuration()
        self._client = httpx.Client(timeout=_TIMEOUT, transport=transport)

    def _validate_configuration(self) -> None:
        if not self.api_url:
            raise RecognitionApiError("API_URL não configurada no .env")
        if not self.device_key:
            raise RecognitionApiError("DEVICE_KEY não configurada no .env")
        if not _ACCESS_POINT_PATTERN.fullmatch(self.access_point):
            raise RecognitionApiError("ACCESS_POINT inválido no .env")

        parsed = urlparse(self.api_url)
        is_local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if parsed.scheme not in {"http", "https"} or (parsed.scheme != "https" and not is_local):
            raise RecognitionApiError("API_URL deve usar HTTPS fora do ambiente local")

    def recognize(self, embedding: np.ndarray) -> RecognitionResult:
        vector = _validate_embedding(embedding)
        payload = {
            "embedding": vector.tolist(),
            "accessPoint": self.access_point,
        }

        try:
            response = self._client.post(
                f"{self.api_url}/access-control/recognize",
                json=payload,
                headers={"X-Device-Key": self.device_key},
            )
        except httpx.RequestError as exc:
            raise ApiUnavailableError("Não foi possível conectar à API de acesso") from exc

        if response.status_code in {401, 403}:
            raise ApiAuthenticationError("Dispositivo não autorizado pela API de acesso")
        if response.status_code >= 500:
            raise ApiUnavailableError("API de acesso temporariamente indisponível")
        if response.status_code != 200:
            raise ApiResponseError(f"API de acesso retornou HTTP {response.status_code}")

        try:
            body = response.json()
        except ValueError as exc:
            raise ApiResponseError("API de acesso retornou JSON inválido") from exc

        return _parse_result(body)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AccessControlClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class EnrollmentClient:
    """Publica embeddings de convidados pela API oficial do Portal VIP."""

    def __init__(
        self,
        api_url: str,
        device_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.device_key = device_key
        parsed = urlparse(self.api_url)
        is_local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if not self.api_url:
            raise RecognitionApiError("API_URL não configurada no .env")
        if not self.device_key:
            raise RecognitionApiError("ENROLLMENT_DEVICE_KEY não configurada no .env")
        if parsed.scheme not in {"http", "https"} or (parsed.scheme != "https" and not is_local):
            raise RecognitionApiError("API_URL deve usar HTTPS fora do ambiente local")
        self._client = httpx.Client(timeout=_TIMEOUT, transport=transport)

    def enroll_guest(
        self,
        guest_id: str,
        embedding: np.ndarray,
        photo_checksum: str,
        *,
        model_version: str = "1",
    ) -> None:
        self._enroll_at_path(f"guests/{guest_id}", guest_id, embedding, photo_checksum, model_version)

    def enroll_participant(self, participant_type: str, participant_id: str, embedding: np.ndarray, photo_checksum: str, *, model_version: str = "1") -> None:
        self._enroll_at_path(f"participants/{participant_type}/{participant_id}", participant_id, embedding, photo_checksum, model_version, participant_type)

    def _enroll_at_path(self, path: str, participant_id: str, embedding: np.ndarray, photo_checksum: str, model_version: str, participant_type: str = "guest") -> None:
        vector = _validate_embedding(embedding)
        if participant_type not in {"guest", "client"} or not re.fullmatch(r"[0-9]+", str(participant_id)):
            raise ValueError("participante inválido")
        if not re.fullmatch(r"[a-f0-9]{64}", photo_checksum):
            raise ValueError("photo_checksum inválido")
        payload = {
            "model": "buffalo_l",
            "modelVersion": model_version,
            "dimension": EMBEDDING_DIMENSION,
            "normalization": "l2",
            "embedding": vector.tolist(),
            "photoChecksum": photo_checksum,
        }
        try:
            response = self._client.put(
                f"{self.api_url}/access-control/{path}/embedding",
                json=payload,
                headers={"X-Device-Key": self.device_key},
            )
        except httpx.RequestError as exc:
            raise ApiUnavailableError("Não foi possível conectar à API de enrollment") from exc
        if response.status_code in {401, 403}:
            raise ApiAuthenticationError("Dispositivo de enrollment não autorizado")
        if response.status_code >= 500:
            raise ApiUnavailableError("API de enrollment temporariamente indisponível")
        if response.status_code != 200:
            raise ApiResponseError(f"Enrollment retornou HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise ApiResponseError("API de enrollment retornou JSON inválido") from exc
        if not isinstance(body, dict) or body.get("success") is not True:
            raise ApiResponseError("Resposta de enrollment fora do contrato")

    def close(self) -> None:
        self._client.close()


def _validate_embedding(embedding: np.ndarray) -> np.ndarray:
    try:
        vector = np.asarray(embedding, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding deve conter somente números") from exc

    if vector.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(f"Embedding deve ter {EMBEDDING_DIMENSION} dimensões")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Embedding deve conter somente valores finitos")

    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError("Embedding deve estar normalizado em L2")
    return vector


def _parse_result(body: object) -> RecognitionResult:
    if not isinstance(body, dict) or body.get("success") is not True:
        raise ApiResponseError("Resposta da API fora do contrato")
    if set(body) != {"success", "data"}:
        raise ApiResponseError("Envelope da API contém campos ausentes ou inesperados")

    data = body.get("data")
    if not isinstance(data, dict):
        raise ApiResponseError("Resposta da API sem objeto data")
    expected_fields = {
        "recognized", "allowed", "reason", "participantType", "participantId",
        "guestId", "userId", "name", "similarity", "direction",
    }
    if set(data) != expected_fields:
        raise ApiResponseError("Resposta da API contém campos ausentes ou inesperados")

    recognized = data.get("recognized")
    allowed = data.get("allowed")
    reason = data.get("reason")
    if type(recognized) is not bool or type(allowed) is not bool:
        raise ApiResponseError("recognized e allowed devem ser booleanos")
    if not isinstance(reason, str) or not reason:
        raise ApiResponseError("reason deve ser uma string não vazia")
    if allowed and not recognized:
        raise ApiResponseError("A API não pode autorizar um rosto não reconhecido")
    if allowed and reason != "AUTHORIZED":
        raise ApiResponseError("Resposta autorizada com reason inconsistente")
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", reason):
        raise ApiResponseError("reason fora do contrato")

    guest_id_value = data.get("guestId")
    if guest_id_value is not None and (
        isinstance(guest_id_value, bool) or not isinstance(guest_id_value, (str, int))
    ):
        raise ApiResponseError("guestId inválido")
    guest_id = str(guest_id_value) if guest_id_value is not None else None
    user_id_value = data.get("userId")
    if user_id_value is not None and (
        isinstance(user_id_value, bool) or not isinstance(user_id_value, (str, int))
    ):
        raise ApiResponseError("userId inválido")
    user_id = str(user_id_value) if user_id_value is not None else None

    participant_type = data.get("participantType")
    participant_id_value = data.get("participantId")
    if participant_type is not None and participant_type not in {"guest", "client"}:
        raise ApiResponseError("participantType inválido")
    if participant_id_value is not None and (
        isinstance(participant_id_value, bool)
        or not isinstance(participant_id_value, (str, int))
    ):
        raise ApiResponseError("participantId inválido")
    participant_id = str(participant_id_value) if participant_id_value is not None else None
    if recognized and (participant_type is None or participant_id is None):
        raise ApiResponseError("Rosto reconhecido sem participante")
    if participant_type == "guest" and guest_id != participant_id:
        raise ApiResponseError("guestId inconsistente")
    if participant_type == "client" and user_id != participant_id:
        raise ApiResponseError("userId inconsistente")
    if not recognized and any((participant_type, participant_id, guest_id, user_id)):
        raise ApiResponseError("Rosto desconhecido não pode identificar participante")

    name = data.get("name")
    if name is not None and not isinstance(name, str):
        raise ApiResponseError("name inválido")
    if allowed and (name is None or not name.strip()):
        raise ApiResponseError("Rosto reconhecido sem name")
    if not recognized and name is not None:
        raise ApiResponseError("Rosto desconhecido não pode conter nome")

    similarity_value = data.get("similarity")
    if similarity_value is not None and (
        isinstance(similarity_value, bool) or not isinstance(similarity_value, (int, float))
    ):
        raise ApiResponseError("similarity inválida")
    similarity = float(similarity_value) if similarity_value is not None else None
    if similarity is not None and (not math.isfinite(similarity) or not -1.0 <= similarity <= 1.0):
        raise ApiResponseError("similarity fora do intervalo de cosseno")
    if recognized and similarity is None:
        raise ApiResponseError("Rosto reconhecido sem similarity")

    direction = data.get("direction")
    if direction is not None and direction not in {"ENTRY", "EXIT"}:
        raise ApiResponseError("direction inválida")
    if allowed and direction is None:
        raise ApiResponseError("Acesso autorizado sem direction")

    return RecognitionResult(
        recognized=recognized,
        allowed=allowed,
        reason=reason,
        guest_id=guest_id,
        user_id=user_id,
        participant_type=participant_type,
        participant_id=participant_id,
        name=name,
        similarity=similarity,
        direction=direction,
    )


_default_client: AccessControlClient | None = None


def recognize(embedding: np.ndarray) -> RecognitionResult:
    """Envia um embedding usando a configuração local do dispositivo."""
    global _default_client
    if _default_client is None:
        _default_client = AccessControlClient(
            api_url=os.getenv("API_URL", ""),
            device_key=os.getenv("DEVICE_KEY", ""),
            access_point=os.getenv("ACCESS_POINT", "vip_room"),
        )
    return _default_client.recognize(embedding)

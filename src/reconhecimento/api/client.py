"""Cliente HTTP para a API de reconhecimento facial.

Faz apenas uma coisa: envia um embedding para a API e devolve a decisão de acesso.
A API hospedada cuida do banco, da busca por similaridade e do log do evento.

Configuração via .env:
    API_URL     = URL base da API  (ex: https://meu-servidor.com)
    API_KEY     = chave de autenticação
    ACCESS_POINT = ponto de acesso  (ex: main_door)
"""
import os
from dataclasses import dataclass

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()

_API_URL     = os.getenv("API_URL", "").rstrip("/")
_API_KEY     = os.getenv("API_KEY", "")
_ACCESS_POINT = os.getenv("ACCESS_POINT", "main_door")

# Timeout generoso para redes lentas (ex: 4G num evento)
_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)


@dataclass
class RecognitionResult:
    recognized: bool
    allowed: bool
    reason: str
    person_id: str | None = None
    name: str | None = None
    similarity: float | None = None


def recognize(embedding: np.ndarray) -> RecognitionResult:
    """Envia o embedding para a API e retorna a decisão de acesso.

    Raises:
        RuntimeError: se a API retornar erro HTTP ou a conexão falhar.
    """
    if not _API_URL:
        raise RuntimeError("API_URL não configurada no .env")
    if not _API_KEY:
        raise RuntimeError("API_KEY não configurada no .env")

    payload = {
        "embedding": embedding.tolist(),
        "access_point": _ACCESS_POINT,
    }

    try:
        response = httpx.post(
            f"{_API_URL}/api/recognize",
            json=payload,
            headers={"X-API-Key": _API_KEY},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"API retornou erro {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Falha de conexão com a API: {exc}") from exc

    data = response.json()

    return RecognitionResult(
        recognized=data["recognized"],
        allowed=data["allowed"],
        reason=data["reason"],
        person_id=data.get("person_id"),
        name=data.get("name"),
        similarity=data.get("similarity"),
    )

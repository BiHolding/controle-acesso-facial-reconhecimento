import os


def camera_index_from_env() -> int:
    raw_value = os.getenv("CAMERA_INDEX", "0")
    try:
        camera_index = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"[CAMERA] Invalid CAMERA_INDEX: {raw_value}") from exc
    if camera_index < 0:
        raise ValueError(f"[CAMERA] Invalid CAMERA_INDEX: {raw_value}")
    return camera_index


def access_direction_from_env() -> str:
    direction = os.getenv("ACCESS_DIRECTION", "ENTRY").strip().upper()
    if direction not in {"ENTRY", "EXIT"}:
        raise ValueError(f"[ACCESS] Invalid ACCESS_DIRECTION: {direction}")
    return direction

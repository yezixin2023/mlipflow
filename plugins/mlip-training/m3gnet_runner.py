"""MatGL M3GNet runner."""

from pathlib import Path

FRAMEWORK = "m3gnet"


def model_directory(value: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir():
        raise ValueError(f"MatGL model directory does not exist: {path}")
    return path

import json
import os
from pathlib import Path

from langchain_core.stores import BaseStore
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class LocalFileStore(BaseStore[str, str]):

    def __init__(self, path: str):
        self.path = path
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump({}, f)

    def _read(self) -> dict:
        with open(self.path, "r") as f:
            return json.load(f)

    def _write(self, data: dict) -> None:
        with open(self.path, "w") as f:
            json.dump(data, f)

    def mget(self, keys: list[str]) -> list[str | None]:
        data = self._read()
        return [data.get(k) for k in keys]

    def mset(self, key_value_pairs: list[tuple[str, str]]) -> None:
        data = self._read()
        for k, v in key_value_pairs:
            data[k] = v
        self._write(data)

    def mdelete(self, keys: list[str]) -> None:
        data = self._read()
        for k in keys:
            data.pop(k, None)
        self._write(data)

    def yield_keys(self, *, prefix: str | None = None):
        data = self._read()
        for k in data:
            if prefix is None or k.startswith(prefix):
                yield k


def get_docstore() -> BaseStore:
    settings = get_settings()
    if settings.use_redis:
        from langchain_community.storage import RedisStore
        import redis

        logger.info("Using Redis docstore at %s", settings.redis_url)
        client = redis.from_url(settings.redis_url)
        return RedisStore(client=client)

    logger.info("Using local file docstore at %s", settings.local_docstore_path)
    return LocalFileStore(settings.local_docstore_path)

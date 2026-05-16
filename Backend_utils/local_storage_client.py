"""
Local filesystem storage client for Chainlit.
Stores element files on disk and serves them via a route on Chainlit's own FastAPI server.
Drop this file next to chainlit_app.py.
"""

import os
import aiofiles
from typing import Any, Dict, Union
from chainlit.data.storage_clients.base import BaseStorageClient
from chainlit.logger import logger


class LocalStorageClient(BaseStorageClient):
    """
    Stores files on the local filesystem under storage_dir.
    URLs returned are relative (/element-files/...) and served by
    a route mounted on Chainlit's own FastAPI server — same origin,
    so auth/cookies/CORS just work.
    """

    def __init__(self, storage_dir: str, public_base: str = ""):
        self.storage_dir = storage_dir
        self.public_base = public_base.rstrip("/")  # "" means relative URLs
        os.makedirs(self.storage_dir, exist_ok=True)
        logger.info(f"LocalStorageClient initialized: dir={self.storage_dir}")

    def _full_path(self, object_key: str) -> str:
        safe_key = object_key.lstrip("/")
        full = os.path.join(self.storage_dir, safe_key)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        return full

    def _url_for(self, object_key: str) -> str:
        safe_key = object_key.lstrip("/")
        return f"{self.public_base}/element-files/{safe_key}"

    async def upload_file(
        self,
        object_key: str,
        data: Union[bytes, str],
        mime: str = "application/octet-stream",
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        try:
            path = self._full_path(object_key)
            if not overwrite and os.path.exists(path):
                return {"object_key": object_key, "url": self._url_for(object_key)}

            mode = "wb" if isinstance(data, bytes) else "w"
            async with aiofiles.open(path, mode) as f:
                await f.write(data)

            size = len(data) if hasattr(data, "__len__") else 0
            logger.info(f"LocalStorage: uploaded {object_key} ({size} bytes)")
            return {"object_key": object_key, "url": self._url_for(object_key)}
        except Exception as e:
            logger.warning(f"LocalStorage upload error: {e}")
            return {}

    async def delete_file(self, object_key: str) -> bool:
        try:
            path = self._full_path(object_key)
            if os.path.exists(path):
                os.remove(path)
            return True
        except Exception as e:
            logger.warning(f"LocalStorage delete error: {e}")
            return False

    async def get_read_url(self, object_key: str) -> str:
        return self._url_for(object_key)

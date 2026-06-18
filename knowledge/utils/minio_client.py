"""MinIO 客户端工具"""

import json
import logging
from pathlib import Path

from minio import Minio

from knowledge.processor.import_process.config import get_config

logger = logging.getLogger("import.minio")

PUBLIC_READ_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": ["*"]},
            "Action": ["s3:GetObject"],
            "Resource": ["arn:aws:s3:::{bucket}/*"],
        }
    ],
}


class MinioClient:
    """MinIO 客户端封装"""

    def __init__(self):
        config = get_config()
        self.endpoint = config.minio_endpoint
        self.access_key = config.minio_access_key
        self.secret_key = config.minio_secret_key
        self.bucket = config.minio_bucket or "product-assistant-images"
        self.secure = config.minio_secure
        self._client = Minio(
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        """确保 bucket 存在并设为公开读"""
        if not self._client.bucket_exists(self.bucket):
            self._client.make_bucket(self.bucket)
            logger.info(f"创建 MinIO bucket: {self.bucket}")

        # 每次初始化都确保公开读策略（修复已存在 bucket 的权限）
        policy = json.dumps(PUBLIC_READ_POLICY).replace("{bucket}", self.bucket)
        try:
            self._client.set_bucket_policy(self.bucket, policy)
            logger.info(f"已设置 bucket 公开读策略: {self.bucket}")
        except Exception as e:
            logger.warning(f"设置 bucket 策略失败（可能 MinIO 版本不支持）: {e}")

    def upload_file(self, local_path: str, object_name: str = None) -> str:
        """
        上传文件到 MinIO，返回访问 URL

        Args:
            local_path: 本地文件路径
            object_name: MinIO 中的对象名，默认使用文件名

        Returns:
            文件的 MinIO URL
        """
        if object_name is None:
            object_name = Path(local_path).name

        self._client.fput_object(
            bucket_name=self.bucket,
            object_name=object_name,
            file_path=local_path,
        )
        protocol = "https" if self.secure else "http"
        url = f"{protocol}://{self.endpoint}/{self.bucket}/{object_name}"
        logger.info(f"上传成功: {local_path} -> {url}")
        return url

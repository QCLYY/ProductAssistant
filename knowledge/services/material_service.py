"""Local material management and runtime configuration checks."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


class MaterialService:
    """Read and manage imported local materials."""

    def list_materials(self, limit: int = 5000) -> dict[str, Any]:
        from knowledge.processor.query_process.config import get_config
        from knowledge.tools.milvus_utils import get_milvus_client

        config = get_config()
        client = get_milvus_client(config.milvus_url)
        if not client:
            return {
                "total": 0,
                "materials": [],
                "errors": ["无法连接本地索引服务，请确认 Docker 服务已启动。"],
            }

        errors: list[str] = []
        material_map: dict[str, dict[str, Any]] = {}

        chunk_rows = self._safe_query(
            client,
            config.chunks_collection,
            ["file_title", "item_name", "chunk_id"],
            limit,
            errors,
        )
        for row in chunk_rows:
            material = self._material_from_row(row)
            if not material:
                continue
            key = self._material_key(material["file_title"], material["material_name"])
            target = material_map.setdefault(key, material)
            self._prefer_specific_material_name(target, material)
            target["chunk_count"] = target.get("chunk_count", 0) + 1

        name_rows = self._safe_query(
            client,
            config.item_name_collection,
            ["file_title", "item_name"],
            limit,
            errors,
        )
        for row in name_rows:
            material = self._material_from_row(row)
            if not material:
                continue
            key = self._material_key(material["file_title"], material["material_name"])
            target = material_map.setdefault(key, material)
            self._prefer_specific_material_name(target, material)
            target["has_name_index"] = True

        materials = sorted(
            material_map.values(),
            key=lambda item: (
                item.get("file_title") or item.get("material_name") or ""
            ).lower(),
        )
        for material in materials:
            material["id"] = self._material_id(
                material.get("file_title", ""),
                material.get("material_name", ""),
            )
            material.setdefault("chunk_count", 0)
            material.setdefault("has_name_index", False)
            material["file_type"] = self._guess_file_type(material.get("file_title", ""))

        return {"total": len(materials), "materials": materials, "errors": errors}

    def delete_material(self, file_title: str = "", material_name: str = "") -> dict[str, Any]:
        file_title = (file_title or "").strip()
        material_name = (material_name or "").strip()
        if not file_title and not material_name:
            raise ValueError("请提供 file_title 或 material_name")

        from knowledge.processor.query_process.config import get_config
        from knowledge.tools.milvus_utils import get_milvus_client

        config = get_config()
        client = get_milvus_client(config.milvus_url)
        if not client:
            raise RuntimeError("无法连接本地索引服务，请确认 Docker 服务已启动")

        related_names = self._resolve_material_names(file_title, material_name)
        if material_name and material_name not in related_names:
            related_names.append(material_name)

        deleted: dict[str, int] = {}
        errors: list[str] = []

        for collection, fields in [
            (config.chunks_collection, ("file_title", "item_name")),
            (config.item_name_collection, ("file_title", "item_name")),
            (config.entity_name_collection, ("item_name",)),
        ]:
            if not collection:
                continue
            expr = self._build_delete_filter(fields, file_title, related_names)
            if not expr:
                continue
            try:
                if not client.has_collection(collection_name=collection):
                    deleted[collection] = 0
                    continue
                result = client.delete(collection_name=collection, filter=expr)
                deleted[collection] = int((result or {}).get("delete_count", 0))
            except Exception as exc:  # pragma: no cover - depends on local services
                errors.append(f"{collection}: {exc}")

        neo4j_deleted = self._delete_neo4j_materials(related_names, errors)

        return {
            "message": "资料删除已执行",
            "file_title": file_title,
            "material_names": related_names,
            "deleted": deleted,
            "neo4j_deleted": neo4j_deleted,
            "errors": errors,
        }

    def check_config(self) -> dict[str, Any]:
        from knowledge.processor.import_process.config import get_config as get_import_config
        from knowledge.processor.query_process.config import get_config as get_query_config

        import_config = get_import_config()
        query_config = get_query_config()

        env_items = [
            ("OPENAI_API_BASE", bool(import_config.openai_api_base), "模型服务地址"),
            ("OPENAI_API_KEY", bool(import_config.openai_api_key), "模型服务密钥"),
            ("MODEL", bool(import_config.default_model), "问答模型"),
            ("ITEM_MODEL", bool(import_config.item_model), "资料名称识别模型"),
            ("VL_MODEL", bool(import_config.vl_model), "视觉模型"),
            ("BGE_M3_PATH", bool(import_config.bge_m3_path), "本地向量模型"),
            ("MILVUS_URL", bool(import_config.milvus_url), "本地索引地址"),
            ("MONGO_URL", bool(query_config.mongo_url), "聊天历史 MongoDB"),
            ("NEO4J_URI", bool(import_config.neo4j_uri), "知识关联服务"),
            ("MINIO_ENDPOINT", bool(import_config.minio_endpoint), "图片/文件对象存储"),
            ("TAVILY_API_KEY", bool(query_config.tavily_api_key), "联网搜索密钥"),
        ]

        checks = [
            {"key": key, "label": label, "ok": ok, "status": "已配置" if ok else "缺失"}
            for key, ok, label in env_items
        ]

        bge_path = import_config.bge_m3_path or ""
        bge_info = {"value_type": "未配置", "exists": False}
        if bge_path:
            local_path = Path(bge_path)
            if local_path.exists():
                bge_info = {"value_type": "本地路径", "exists": True}
            else:
                bge_info = {"value_type": "模型仓库名称", "exists": False}

        services = [
            self._check_milvus(import_config.milvus_url),
            self._check_mongo(query_config.mongo_url),
            self._check_neo4j(import_config.neo4j_uri),
            self._check_minio(import_config),
        ]

        return {
            "app_name": "品辅",
            "checks": checks,
            "services": services,
            "bge": bge_info,
            "web_search_enabled": bool(query_config.enable_web_search),
            "web_search_provider": query_config.web_search_provider or "tavily",
        }

    @staticmethod
    def _safe_query(client, collection_name: str, output_fields: list[str], limit: int, errors: list[str]) -> list[dict]:
        if not collection_name:
            return []
        try:
            if not client.has_collection(collection_name=collection_name):
                return []
            return client.query(
                collection_name=collection_name,
                filter="",
                output_fields=output_fields,
                limit=limit,
            ) or []
        except Exception as exc:  # pragma: no cover - depends on local services
            errors.append(f"读取 {collection_name} 失败: {exc}")
            return []

    @staticmethod
    def _material_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
        file_title = str(row.get("file_title") or "").strip()
        material_name = str(row.get("item_name") or "").strip()
        if not (file_title or material_name):
            return None
        return {
            "file_title": file_title or material_name,
            "material_name": material_name or file_title,
            "chunk_count": 0,
            "has_name_index": False,
        }

    @staticmethod
    def _material_key(file_title: str, material_name: str) -> str:
        return (file_title or material_name or "").lower()

    @staticmethod
    def _prefer_specific_material_name(target: dict[str, Any], candidate: dict[str, Any]) -> None:
        target_name = str(target.get("material_name") or "").strip()
        target_title = str(target.get("file_title") or "").strip()
        candidate_name = str(candidate.get("material_name") or "").strip()
        if candidate_name and candidate_name != target_title and (
            not target_name or target_name == target_title
        ):
            target["material_name"] = candidate_name

    @staticmethod
    def _material_id(file_title: str, material_name: str) -> str:
        raw = f"{file_title or ''}::{material_name or ''}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:16]

    @staticmethod
    def _guess_file_type(file_title: str) -> str:
        suffix = Path(file_title or "").suffix.lower().lstrip(".")
        return suffix.upper() if suffix else "资料"

    def _resolve_material_names(self, file_title: str, material_name: str) -> list[str]:
        names: list[str] = []
        for item in self.list_materials().get("materials", []):
            if file_title and item.get("file_title") != file_title:
                continue
            name = str(item.get("material_name") or "").strip()
            if name and name not in names:
                names.append(name)
        if material_name and material_name not in names:
            names.append(material_name)
        return names

    @classmethod
    def _build_delete_filter(cls, fields: tuple[str, ...], file_title: str, material_names: list[str]) -> str:
        parts: list[str] = []
        if "file_title" in fields and file_title:
            parts.append(f'file_title == "{cls._escape_milvus_string(file_title)}"')
        if "item_name" in fields and material_names:
            quoted = ", ".join(f'"{cls._escape_milvus_string(name)}"' for name in material_names if name)
            if quoted:
                parts.append(f"item_name in [{quoted}]")
        return " or ".join(f"({part})" for part in parts)

    @staticmethod
    def _escape_milvus_string(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _delete_neo4j_materials(material_names: list[str], errors: list[str]) -> int:
        if not material_names:
            return 0
        try:
            from knowledge.processor.query_process.config import get_config
            from knowledge.tools.neo4j_utils import get_neo4j_driver

            config = get_config()
            driver = get_neo4j_driver()
            with driver.session(database=config.neo4j_database) as session:
                result = session.run(
                    "MATCH (n) WHERE n.item_name IN $names "
                    "WITH collect(n) AS nodes, count(n) AS deleted "
                    "FOREACH (node IN nodes | DETACH DELETE node) "
                    "RETURN deleted",
                    names=material_names,
                )
                record = result.single()
                return int(record["deleted"]) if record else 0
        except Exception as exc:  # pragma: no cover - depends on local services
            errors.append(f"知识关联服务: {exc}")
            return 0

    @staticmethod
    def _check_milvus(url: str) -> dict[str, Any]:
        try:
            from knowledge.tools.milvus_utils import get_milvus_client

            client = get_milvus_client(url)
            collections = client.list_collections() if client else []
            return {"name": "本地索引服务", "ok": bool(client), "detail": f"{len(collections)} 个集合"}
        except Exception as exc:
            return {"name": "本地索引服务", "ok": False, "detail": str(exc)}

    @staticmethod
    def _check_mongo(url: str) -> dict[str, Any]:
        try:
            from pymongo import MongoClient

            client = MongoClient(url, serverSelectionTimeoutMS=1500)
            client.admin.command("ping")
            return {"name": "MongoDB", "ok": True, "detail": "连接正常"}
        except Exception as exc:
            return {"name": "MongoDB", "ok": False, "detail": str(exc)}

    @staticmethod
    def _check_neo4j(uri: str) -> dict[str, Any]:
        if not uri:
            return {"name": "知识关联服务", "ok": False, "detail": "未配置知识关联服务地址"}
        try:
            from knowledge.tools.neo4j_utils import get_neo4j_driver

            driver = get_neo4j_driver()
            driver.verify_connectivity()
            return {"name": "知识关联服务", "ok": True, "detail": "连接正常"}
        except Exception as exc:
            return {"name": "知识关联服务", "ok": False, "detail": str(exc)}

    @staticmethod
    def _check_minio(config) -> dict[str, Any]:
        if not config.minio_endpoint:
            return {"name": "MinIO", "ok": False, "detail": "未配置 MINIO_ENDPOINT"}
        try:
            from minio import Minio

            client = Minio(
                endpoint=config.minio_endpoint,
                access_key=config.minio_access_key,
                secret_key=config.minio_secret_key,
                secure=config.minio_secure,
            )
            bucket = config.minio_bucket or "product-assistant-images"
            ok = client.bucket_exists(bucket)
            return {"name": "MinIO", "ok": ok, "detail": f"bucket: {bucket}"}
        except Exception as exc:
            return {"name": "MinIO", "ok": False, "detail": str(exc)}


material_service = MaterialService()

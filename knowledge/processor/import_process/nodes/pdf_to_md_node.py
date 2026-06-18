import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import (
    FileProcessingError,
    PdfConversionError,
    ValidationError,
)
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    """Convert an uploaded PDF to Markdown with MinerU."""

    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        import_file_path_obj, file_dir_obj = self._validate_state_input_path(state)

        process_code = self._execute_mineru(import_file_path_obj, file_dir_obj)
        if process_code != 0:
            raise PdfConversionError("MinerU PDF conversion failed", self.name)

        self.log_step("step3", "Find converted Markdown")
        file_title = state.get("file_title", "") or import_file_path_obj.stem
        md_output_path = self._find_output_md(file_dir_obj, file_title)
        if not md_output_path:
            raise FileProcessingError(
                f"Converted Markdown file was not found under {file_dir_obj}",
                self.name,
            )

        self.logger.info(f"Converted Markdown path: {md_output_path}")
        state["md_path"] = str(md_output_path)
        return state

    def _validate_state_input_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        self.log_step("step1", "Validate PDF input path")
        import_file_path = state.get("import_file_path", "")
        file_dir = state.get("file_dir", "")

        if not import_file_path:
            raise ValidationError("Missing import_file_path", self.name)

        import_file_path_obj = Path(import_file_path)
        if not import_file_path_obj.exists():
            raise FileProcessingError(f"Input file does not exist: {import_file_path}", self.name)

        file_dir_obj = Path(file_dir) if file_dir else import_file_path_obj.parent
        self.logger.info(f"Input PDF: {import_file_path_obj}, output dir: {file_dir_obj}")
        return import_file_path_obj, file_dir_obj

    def _execute_mineru(self, import_file_path_obj: Path, file_dir_obj: Path) -> int:
        self.log_step("step2", "Run MinerU")
        mineru_cmd = self._resolve_mineru_cmd()
        cmd = [
            mineru_cmd,
            "-p", str(import_file_path_obj),
            "-o", str(file_dir_obj),
        ]

        backend = os.getenv("MINERU_BACKEND", "pipeline").strip()
        method = os.getenv("MINERU_METHOD", "auto").strip()
        lang = os.getenv("MINERU_LANG", "ch").strip()

        if backend:
            cmd.extend(["-b", backend])
        if method:
            cmd.extend(["-m", method])
        if lang:
            cmd.extend(["-l", lang])

        self.logger.info(f"MinerU command: {cmd}")
        begin = time.time()
        process = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            errors="replace",
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        if process.stdout:
            for output in process.stdout:
                self.logger.info(f"MinerU: {output.strip()}")

        process_code = process.wait()
        elapsed = time.time() - begin
        if process_code != 0:
            self.logger.info(f"MinerU failed: {import_file_path_obj.name}, {elapsed:.2f}s")
        else:
            self.logger.info(f"MinerU succeeded: {import_file_path_obj.name}, {elapsed:.2f}s")
        return process_code

    @staticmethod
    def _resolve_mineru_cmd() -> str:
        configured = os.getenv("MINERU_CMD", "").strip()
        candidates = [
            configured,
            shutil.which("mineru") or "",
            str(Path(sys.executable).parent / "Scripts" / "mineru.exe"),
            str(Path(sys.executable).parent / "mineru.exe"),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        raise PdfConversionError(
            "MinerU command was not found. Set MINERU_CMD in knowledge/.env.",
            PdfToMdNode.name,
        )

    @staticmethod
    def _find_output_md(file_dir_obj: Path, file_title: str) -> Path | None:
        preferred_paths = [
            file_dir_obj / file_title / "hybrid_auto" / f"{file_title}.md",
            file_dir_obj / file_title / "auto" / f"{file_title}.md",
            file_dir_obj / file_title / f"{file_title}.md",
        ]
        for path in preferred_paths:
            if path.exists():
                return path

        search_root = file_dir_obj / file_title
        if not search_root.exists():
            search_root = file_dir_obj

        matches = sorted(
            search_root.rglob("*.md"),
            key=lambda p: (p.stem != file_title, -p.stat().st_mtime),
        )
        return matches[0] if matches else None


if __name__ == "__main__":
    setup_logging()

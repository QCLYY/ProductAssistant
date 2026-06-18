import json
from pathlib import Path
from typing import Tuple
import subprocess

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    """
    pdf转md节点
    """
    name="pdf_to_md_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """

        Args:
            state:

        Returns:

        """
        #1.对参数作校验
        import_file_path_obj,file_dir_obj=self._validate_state_input_path(state)


        #2.利用 MinerU 工具解析pdf成为md
        process_code=self._execute_mineru(import_file_path_obj,file_dir_obj)
        if process_code != 0:
            raise PdfConversionError("执行MinerU解析PDF失败",self.name)
        #3.获取解析后的md的path
        self.log_step("step3", "[获取解析后的MD路径]")
        file_title = state.get("file_title", "")
        if file_title=="":
            file_title=import_file_path_obj.stem
        md_output_path = file_dir_obj / file_title / "hybrid_auto" / f"{file_title}.md"
        if not md_output_path.exists():
            raise FileProcessingError(f"解析后的MD文件不存在: {md_output_path}", self.name)
        self.logger.info(f"解析后的MD路径: {md_output_path}")
        #4.更新state 字典的md_path
        state["md_path"] = str(md_output_path)
        #5.返回state
        return state

    def _validate_state_input_path(self, state: ImportGraphState)->Tuple[Path, Path]:
        """

        Args:
            state:

        Returns:

        """
        self.log_step("step1","[校验输入文件路径]")
        import_file_path=state.get("import_file_path",'')
        file_dir=state.get("file_dir",'')
        if not import_file_path:
            raise ValidationError("解析的文件不存在",self.name)
        import_file_path_obj=Path(import_file_path)
        if not import_file_path_obj.exists():
            raise FileProcessingError("解析的文件路径不存在",self.name)
        if not file_dir:
            #E:\AI+Py\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用.pdf
            file_dir=import_file_path_obj.parent

        file_dir_obj=Path(file_dir)
        self.logger.info(f"输入文件路径: {import_file_path_obj}, 文件目录: {file_dir_obj}")
        return import_file_path_obj,file_dir_obj

    def _execute_mineru(self, import_file_path_obj, file_dir_obj)->int:
        self.log_step("step2","[执行MinerU解析PDF]")
        #执行命令mineru -p <input_path> -o <output_path>
        #1.构建命令行
        cmd = [
            "mineru",
            "-p", str(import_file_path_obj),
            "-o", str(file_dir_obj),
            "--source", "local"
        ]
        import time
        begin=time.time()
        #2.执行命令（子进程执行）
        process = subprocess.Popen(
            args=cmd,#命令行参数列表，推荐使用列表形式，避免空； 格和特殊字符问题
            stdout=subprocess.PIPE,#捕获标准输出
            stderr=subprocess.STDOUT,#合并标准错误到标准输出，方便统一处理
            errors="replace",#替换无法解码的字符，避免乱码
            text=True,#以文本模式处理输出，直接得到字符串
            encoding="utf-8",#指定编码，避免乱码
            bufsize=1,#行缓冲，实时输出日志
        )
        #3.获取日志信息
        for output in process.stdout:
            self.logger.info(f"执行MinerU产生的日志: {output.strip()}")
        #4.等待命令执行完成
        process_code=process.wait()
        end=time.time()

        #5.返回状态码
        if process_code != 0:
            self.logger.info(f"执行MinerU解析PDF失败:{import_file_path_obj.name} 耗时{end-begin:.2f}s")
            return process_code
        else:
            self.logger.info(f"执行MinerU解析PDF成功:{import_file_path_obj.name} 耗时{end-begin:.2f}s")
            return process_code


if  __name__ == "__main__":
    setup_logging()
    # 简单测试
    node = PdfToMdNode()
    test_state = {
        "import_file_path": r"E:\AI+Py\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\hak180产品安全手册.pdf",
        "file_dir": r"E:\AI+Py\shopkeeper_brain\knowledge\processor\import_process\output_temp_dir",
        "file_title": "万用表RS-12的使用"
    }
    try:
        result_state = node.process(test_state)
        print("处理后的状态:", json.dumps(s=result_state, indent=2, ensure_ascii=False))
    except ValidationError as e:
        print("验证错误:", e)
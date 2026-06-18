import json
from pathlib import Path


from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    """
    实体节点
    位置：整个导入流程中的位置（第一段）
    作用：对上传的文件类型做判断（.pdf or .md）
    """
    name="entry_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
            处理文件类型的检测
        Args:
            state: ImportGraphState 图的状态

        Returns:
            处理之后的状态

        """
        #1.获取导入文件的路径以及文件的目录
        self.log_step("step1", "[获取文件路径]")
        import_file_path=state.get("import_file_path")
        file_dir=state.get("file_dir")

        #2.简单校验一下导入文件的路径以及文件的目录
        self.log_step("step2","[检测文件路径]")
        if not file_dir:
            raise ValidationError("文件目录不能为空",self.name)
        if not import_file_path:
            raise ValidationError("导入文件路径不能为空",self.name)
        #3.根据导入文件的路径判断文件类型，并更新状态
        self.log_step("step3","[判断文件类型]")
        #4.获取上传的后缀
        path=Path(import_file_path)
        suffix=path.suffix.lower()
        #5.判断后缀并赋值state
        if suffix==".pdf":
            state["is_pdf_read_enabled"]=True
            state["pdf_path"]=import_file_path
        elif suffix==".md":
            state["is_md_read_enabled"]=True
            state["md_path"]=import_file_path
        else:
            self.logger.debug(f"不支持的文件类型: {suffix}")
            raise ValidationError("不支持的文件类型，仅支持pdf和md",self.name,suffix)
        #6.赋值file_tile
        state["file_title"]=path.stem
        self.logger.debug(f"文件标题: {state['file_title']}")
        return state


if __name__ == "__main__":
    setup_logging()
    # 简单测试
    node = EntryNode()
    test_state = {
        "import_file_path": r"/path/to/document.md",
        "file_dir": r"/path/to"
    }
    try:
        updated_state = node.process(test_state)
        print("Updated State:", json.dumps(updated_state, ensure_ascii=False,indent=4))
    except ValidationError as e:
        print(f"Validation Error: {e}")



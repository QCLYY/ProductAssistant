"""
文档切分节点 — 融合层级追踪 + LangChain 递归切分 + 同宗同源合并

处理流程（7 步）：
  1. 获取输入（md_content、file_title、max_length）
  2. 按 Markdown 标题一级切分，parent_title 层级追踪发证
  3. 处理全文无标题情况（整体作为一个 chunk）
  4. 二次切分超长章节（LangChain RecursiveCharacterTextSplitter）+ 合并过短章节
  5. 组装最终 content（title + body）
  6. 日志统计
  7. 备份 + 状态更新
"""

import re
import os
import json
from typing import List, Tuple, Optional

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import DocumentSplitError


class DocumentSplitNode(BaseNode):
    """
    文档切分节点

    按 Markdown 标题结构切分文档，追踪层级血缘关系，
    对超长章节引入 LangChain 递归切分，对过短片段基于同宗同源合并。
    """

    name = "document_split"

    # ------------------------------------------------------------------ #
    #                           主流程                                     #
    # ------------------------------------------------------------------ #

    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = get_config()

        # Step 1: 获取输入
        content, file_title, max_length = self._get_inputs(state, config)
        if not content:
            raise DocumentSplitError("md_content 为空", node_name=self.name)

        # Step 1.5: HTML 表格降维转译（HTML table → 自然语言句子）
        content = self._normalize_tables(content)

        # Step 2: 按标题一级切分（带层级追踪，发放 parent_title 身份证）
        sections, has_title = self._split_by_headings(content, file_title)

        # Step 3: 处理全文无标题情况
        if not has_title:
            sections = [{
                "title": "无标题",
                "body": content,
                "file_title": file_title,
                "parent_title": file_title,
            }]
            self.logger.info("全文无标题，作为单个 chunk 处理")

        # Step 4: 二次切分 + 合并短章节
        sections = self._split_and_merge(
            sections, max_length, config.min_content_length,
            config.overlap_sentences,
        )

        # Step 5: 组装最终 content（title + body）
        sections = self._assemble_content(sections)

        # Step 6: 日志统计
        self._log_summary(content, sections, max_length)

        # Step 7: 备份 + 写入 state
        state["chunks"] = sections
        self._backup_chunks(state, sections)

        return state

    # ------------------------------------------------------------------ #
    #                       Step 1: 获取输入                               #
    # ------------------------------------------------------------------ #

    def _get_inputs(
        self, state: ImportGraphState, config
    ) -> Tuple[Optional[str], Optional[str], int]:
        self.log_step("step_1", "获取输入")

        content = state.get("md_content", "")
        if content:
            # 统一换行符，避免正则匹配出 Bug
            content = content.replace("\r\n", "\n").replace("\r", "\n")

        file_title = state.get("file_title", "")
        max_length = config.max_content_length

        return content, file_title, max_length

    # ------------------------------------------------------------------ #
    #              Step 1.5: HTML 表格降维转译                               #
    # ------------------------------------------------------------------ #

    def _normalize_tables(self, content: str) -> str:
        """
        将文档中的 HTML <table> 降维转译为自然语言句子，
        消除 HTML 标签对 RAG embedding 的噪声干扰。

        算法：2D 矩阵投影 → 意图嗅探 → 逐行语义重构。
        """
        soup = BeautifulSoup(content, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return content

        for table in tables:
            rows = table.find_all("tr")
            if not rows:
                continue

            # —— 第1步：构建 2D 矩阵（rowspan/colspan 全填充） ——
            grid = self._build_table_grid(rows)
            if not grid or not grid[0]:
                continue

            col_count = max(len(r) for r in grid)
            for r in grid:
                while len(r) < col_count:
                    r.append("")

            # —— 第2步：检测表格类型 + 合并多级表头 ——
            table_type, headers = self._detect_table_type(grid, col_count)

            # —— 第3步：语义重构 ——
            nl_lines = self._matrix_to_natural_language(
                grid, col_count, table_type, headers
            )

            # —— 第4步：替换原始 HTML table ——
            nl_block = "\n".join(nl_lines)
            tag = soup.new_string(f"\n{nl_block}\n")
            table.replace_with(tag)

        return str(soup)

    def _build_table_grid(self, rows) -> List[List[str]]:
        """
        构建 2D 矩阵：将每个 cell 的文本填充到它跨的所有 rowspan × colspan 位置。
        对标 markdown_utils 的 grid 构建方式，简洁且正确处理复杂 rowspan。
        """
        grid: List[List[str]] = []
        for _ in range(len(rows)):
            grid.append([])

        for row_idx, row in enumerate(rows):
            col_idx = 0
            for cell in row.find_all(["td", "th"]):
                # 跳过已被前面 cell 的 colspan/rowspan 占用的列
                while col_idx < len(grid[row_idx]) and grid[row_idx][col_idx] != "":
                    col_idx += 1

                rowspan = int(cell.get("rowspan", 1))
                colspan = int(cell.get("colspan", 1))
                text = cell.get_text(separator=" ", strip=True)

                # 将 cell 文本填到它跨的所有行和列中
                for r in range(row_idx, row_idx + rowspan):
                    while len(grid) <= r:
                        grid.append([])
                    # 补齐当前行到目标列数
                    while len(grid[r]) < col_idx + colspan:
                        grid[r].append("")
                    for c in range(col_idx, col_idx + colspan):
                        grid[r][c] = text

                col_idx += colspan

        return grid

    def _detect_table_type(
        self, grid: List[List[str]], col_count: int
    ) -> Tuple[str, List[str]]:
        """
        检测表格类型 + 合并多级表头，返回 (type, headers)。
        先判断类型（基于原始 grid），再输出合并表头。
        """
        if len(grid) < 2 or col_count < 2:
            return "normal", grid[0] if grid else []

        # —— 类型预判（基于原始 grid） ——
        # 交叉表：左上角空，首行其余列和首列其余行均有值
        is_cross = (
            grid[0][0] == ""
            and all(grid[0][c] for c in range(1, col_count))
            and all(grid[r][0] for r in range(1, len(grid)))
        )
        # 键值表：两列，左列全非空
        is_kv = (
            col_count == 2
            and all(grid[r][0] for r in range(len(grid)))
        )

        # —— 多级表头检测 ——
        header_end = 1
        has_colspan_artifact = any(
            grid[0][c] == grid[0][c - 1] for c in range(1, col_count)
        )
        if has_colspan_artifact:
            for r in range(1, min(len(grid), 5)):
                filled = sum(1 for cell in grid[r] if cell)
                if filled > col_count * 0.5:
                    header_end = r + 1
        # 最多 3 行表头（子表头的子表头的最终极限）；超过的视为数据行
        header_end = min(header_end, 3)

        # 合并表头行
        merged_headers: List[str] = []
        for c in range(col_count):
            parts: List[str] = []
            for r in range(header_end):
                val = grid[r][c]
                if val and val not in parts:
                    parts.append(val)
            # 保持原始空值（交叉表左上角），不替换为 "列N"
            merged_headers.append("/".join(parts) if parts else "")

        # 用合并表头替换原表头行
        data_start = header_end
        data_matrix = [merged_headers] + grid[data_start:] if data_start < len(grid) else [merged_headers]
        grid.clear()
        grid.extend(data_matrix)

        if is_cross:
            return "cross", merged_headers
        if is_kv:
            return "kv", merged_headers
        return "normal", merged_headers

    def _matrix_to_natural_language(
        self,
        matrix: List[List[str]],
        col_count: int,
        table_type: str,
        headers: List[str],
    ) -> List[str]:
        """
        逐行转译为自然语言句子。每行自包含完整表头信息，适合 RAG 检索。
        """
        lines: List[str] = []

        if table_type == "cross":
            row_headers = headers[1:]
            for r in range(1, len(matrix)):
                row_key = matrix[r][0]
                if not row_key:
                    continue
                parts = []
                for c in range(1, col_count):
                    val = matrix[r][c]
                    if not val:
                        continue
                    label = row_headers[c - 1] if c - 1 < len(row_headers) else f"列{c + 1}"
                    parts.append(f"{label}：{val}")
                if parts:
                    lines.append(f"【{row_key}】{'；'.join(parts)}。")

        elif table_type == "kv":
            for r in range(len(matrix)):
                key = matrix[r][0]
                val = matrix[r][1] if col_count > 1 else ""
                if key:
                    lines.append(f"- {key}：{val}。")

        else:
            # 普通表：每行自包含完整表头
            for r in range(1, len(matrix)):
                row_parts = []
                for c in range(col_count):
                    val = matrix[r][c]
                    if not val:
                        continue
                    hdr = headers[c] if c < len(headers) and headers[c] else ""
                    label = hdr if hdr else f"列{c + 1}"
                    row_parts.append(f"{label}：{val}")
                if row_parts:
                    lines.append(f"- {'；'.join(row_parts)}。")

        return lines

    # ------------------------------------------------------------------ #
    #                  Step 2: 按标题一级切分（带层级追踪）                    #
    # ------------------------------------------------------------------ #

    def _split_by_headings(
        self, content: str, file_title: str
    ) -> Tuple[List[dict], bool]:
        """
        按 Markdown 标题行切分，title 与 body 分开存储。
        通过 hierarchy 数组追踪层级，为每个 section 发放 parent_title 身份证。
        """
        self.log_step("step_2", "按标题切分并追踪层级")

        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")
        lines = content.split("\n")

        sections: List[dict] = []
        current_title = ""
        current_level = 0
        body_lines: List[str] = []
        has_title = False
        in_fence = False

        # 核心魔法：记录 1-6 级标题的最新足迹（索引 0 不用）
        hierarchy = [""] * 7

        def _flush():
            """将当前积累的内容保存为一个 section，并计算 parent_title"""
            body = "\n".join(body_lines).strip()
            if current_title or body:
                # 向上寻找最近的"长辈"作为 parent_title
                parent_title = ""
                for lvl in range(current_level - 1, 0, -1):
                    if hierarchy[lvl]:
                        parent_title = hierarchy[lvl]
                        break

                # 没找到长辈：统一归到 file_title 门下，同级 H1 间才可合并
                if not parent_title:
                    parent_title = file_title

                sections.append({
                    "title": current_title,
                    "body": body,
                    "file_title": file_title,
                    "parent_title": parent_title,
                })

        for line in lines:
            # 代码围栏检测：区分开启与关闭，防止不成对 toggle
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                if not in_fence:
                    in_fence = True
                elif stripped in ("```", "~~~"):
                    in_fence = False

            match = heading_re.match(line) if not in_fence else None

            if match:
                has_title = True
                _flush()  # 先把上一段结算落盘

                level = len(match.group(1))
                current_level = level
                current_title = line.strip()
                hierarchy[level] = current_title

                # 重点：出现新的上级标题，其下属的子标题足迹全清空
                for i in range(level + 1, 7):
                    hierarchy[i] = ""

                body_lines = []
            else:
                body_lines.append(line)

        _flush()  # 处理文档最后一段
        return sections, has_title

    # ------------------------------------------------------------------ #
    #                Step 4: 二次切分 + 合并短章节                          #
    # ------------------------------------------------------------------ #

    def _split_and_merge(
        self,
        sections: List[dict],
        max_length: int,
        min_length: int,
        overlap: int = 0,
    ) -> List[dict]:
        self.log_step("step_4", "二次切分和合并")

        if max_length <= 0:
            return sections

        # 4a: 对超长章节做二次切分
        split_result: List[dict] = []
        for section in sections:
            split_result.extend(self._split_long_section(section, max_length, overlap))

        # 4b: 合并过短的相邻章节（仅限同一 parent_title 下的子片段）
        return self._merge_short_sections(split_result, min_length, max_length)

    def _split_long_section(
        self, section: dict, max_length: int, overlap: int = 0
    ) -> List[dict]:
        """
        引入 LangChain 的 RecursiveCharacterTextSplitter 对超长 body 优雅降级切分。
        降级顺序：\\n\\n → \\n → 中文标点 → 英文标点 → 空格。

        overlap > 0 时，相邻 chunk 之间保持句子级重叠——
        将前一个 chunk 的最后 N 句复制到后一个 chunk 的开头，而非字符级截断。
        """
        title = section.get("title", "")
        body = section.get("body", "")
        file_title = section.get("file_title", "")
        parent_title = section.get("parent_title", title)

        # title 作为前缀会占用一部分空间
        title_prefix = f"{title}\n\n" if title else ""
        total = len(title_prefix) + len(body)

        if total <= max_length:
            return [section]
        #5.切分 #5.1对谁切(body) #5.2切多少(available) #5.3怎么切(RecursiveCharacterTextSplitter)  / 手写
        available = max_length - len(title_prefix)
        if available <= 0:
            return [section]
        #定义递归切分器，按照优先级依次切分，直到满足长度要求
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
            keep_separator=False,
        )
        #
        pieces = splitter.split_text(body)  #sp lit_text得到List字符串

        # 防御性代码：万一没切开，原样返回
        if len(pieces) <= 1:
            return [section]

        # 句子级 overlap：前一个 piece 的最后 N 句 → 后一个 piece 开头
        if overlap > 0:
            sentence_re = re.compile(r"(?<=[。！？!?\n])\s*")
            overlapped = [pieces[0]]
            for i in range(1, len(pieces)):
                prev_sentences = [
                    s for s in sentence_re.split(pieces[i - 1]) if s.strip()
                ]
                prefix = (
                    "".join(prev_sentences[-overlap:])
                    if len(prev_sentences) >= overlap
                    else pieces[i - 1]
                )
                overlapped.append(prefix + pieces[i])
            pieces = overlapped
        # 组装切片结果
        sub_sections = []
        for i, piece in enumerate(pieces):
            sub_sections.append({
                "title": f"{title}-{i + 1}" if title else f"chunk-{i + 1}",
                "body": piece.strip(),
                "file_title": file_title,
                "parent_title": parent_title,
                "part": i + 1,
            })

        return sub_sections

    def _merge_short_sections(
        self, sections: List[dict], min_length: int, max_length: int
    ) -> List[dict]:
        """
        合并过短的相邻子片段（仅限同一 parent_title 下的片段）。
        空 body section 始终合并。合并后若超过 max_length 则不合并。
        """
        if not sections:
            return []

        merged: List[dict] = []
        current = sections[0]

        for next_sec in sections[1:]:
            cur_body = current.get("body", "")
            cur_body_len = len(cur_body)
            next_body = next_sec.get("body", "")

            same_parent = (
                current.get("parent_title")
                and current["parent_title"] == next_sec.get("parent_title")
            )

            cur_empty = not cur_body.strip()
            next_empty = not next_body.strip()
            cur_short = cur_body_len < min_length
            next_short = len(next_body) < min_length

            # 任一为短/空且同宗同源则尝试合并
            should_merge = (cur_empty or cur_short or next_empty or next_short) and same_parent

            if should_merge:
                merged_len = cur_body_len + len(next_body) + 2
                if merged_len <= max_length or cur_empty or next_empty:
                    sep = "\n\n" if cur_body and next_body else ""
                    current["body"] = cur_body + sep + next_body
                    if not current.get("title") or current["title"] == current.get("file_title", ""):
                        current["title"] = next_sec.get("title", current.get("title", ""))
                    if "part" in next_sec:
                        current["part"] = next_sec["part"]
                    continue

            merged.append(current)
            current = next_sec

        merged.append(current)
        return merged

    # ------------------------------------------------------------------ #
    #               Step 5: 组装最终 content                               #
    # ------------------------------------------------------------------ #

    def _assemble_content(self, sections: List[dict]) -> List[dict]:
        """
        将 title + body 组装为最终的 content 字段，
        清理内部临时字段 body，保留 parent_title 和 part 供下游使用。
        """
        self.log_step("step_5", "组装 content")

        result: List[dict] = []
        for sec in sections:
            title = sec.get("title", "")
            body = sec.get("body", "").strip()

            # 跳过空 content（非首个）
            if not title and not body and result:
                continue

            if title and body:
                content = f"{title}\n\n{body}"
            else:
                content = title or body

            chunk = {
                "title": title,
                "content": content.strip(),
                "file_title": sec.get("file_title", ""),
            }

            # 保留二次切分产生的字段，供下游合并/溯源使用
            if "parent_title" in sec:
                chunk["parent_title"] = sec["parent_title"]
            if "part" in sec:
                chunk["part"] = sec["part"]

            result.append(chunk)

        return result

    # ------------------------------------------------------------------ #
    #                       日志 & 备份                                    #
    # ------------------------------------------------------------------ #

    def _log_summary(self, raw_content: str, sections: List[dict], max_length: int):
        self.log_step("step_6", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(sections)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if sections:
            self.logger.info("章节预览:")
            for i, sec in enumerate(sections[:5]):
                title = sec.get("title", "")[:50]
                content_len = len(sec.get("content", ""))
                self.logger.info(f"  {i + 1}. {title}... ({content_len} 字符)")
            if len(sections) > 5:
                self.logger.info(f"  ... 还有 {len(sections) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        self.log_step("step_7", "备份切片")

        local_dir = state.get("file_dir", state.get("local_dir", ""))
        if not local_dir:
            self.logger.debug("未设置 file_dir/local_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == "__main__":
    file_path = (
        r"D:\path\to\ProductAssistant\knowledge\processor\import_process\output_temp_dir\sample\hybrid_auto\sample.md"
    )
    with open(file_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    setup_logging()
    node = DocumentSplitNode()
    test_state = {
        "file_title": "万用表RS-12的使用",
        "md_content": md_content,
        "file_dir": os.path.dirname(file_path),
    }
    result = node.process(test_state)
    chunks = result.get("chunks", [])
    print(json.dumps(chunks, ensure_ascii=False, indent=2))
    print("\n" + "=" * 50)
    print(f"切分完成，共 {len(chunks)} 个 chunks")
    print("=" * 50)
    for c in chunks[:10]:
        print(f"title: {c.get('title', '')[:80]}")
        print(f"parent: {c.get('parent_title', 'N/A')[:60]}")
        print(f"content[{len(c.get('content', ''))}]: {c.get('content', '')[:120]}")
        print("---")
    if len(chunks) > 10:
        print(f"... 还有 {len(chunks) - 10} 个 chunk（详见 chunks.json）")

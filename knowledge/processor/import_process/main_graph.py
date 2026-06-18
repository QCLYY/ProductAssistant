import json

from langgraph.graph import StateGraph

from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state


def route_by_file_type(state: ImportGraphState) -> str:
    """根据文件类型路由到不同节点"""
    if state.get("is_md_read_enabled"):
        return "md_img_node"
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"
    return "__end__"


def create_import_graph() -> StateGraph:
    """创建导入流程图

    流程:
      __start__ -> entry_node
        -> (PDF)  pdf_to_md_node -> md_img_node -> document_split
        -> (MD)   md_img_node -> document_split
      -> item_name -> bge_embedding -> import_milvus -> knowledge_graph -> __end__
    """
    graph_workflow = StateGraph(ImportGraphState)  # type:ignore

    # 节点
    from knowledge.processor.import_process.nodes.entry_node import EntryNode
    from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
    from knowledge.processor.import_process.nodes.md_img_node import MarkDownImageNode

    from knowledge.processor.import_process.nodes.item_name_recognition import ItemNameRecognitionNode
    from knowledge.processor.import_process.nodes.bge_embedding import BgeEmbeddingNode
    from knowledge.processor.import_process.nodes.import_milvus import ImportMilvusNode
    from knowledge.processor.import_process.nodes.knowledge_graph import KnowledgeGraphNode

    entry_node = EntryNode()
    pdf_to_md_node = PdfToMdNode()
    md_img_node = MarkDownImageNode()
    document_split_node = DocumentSplitNode()
    item_name_node = ItemNameRecognitionNode()
    bge_embedding_node = BgeEmbeddingNode()
    import_milvus_node = ImportMilvusNode()
    knowledge_graph_node = KnowledgeGraphNode()

    all_nodes = [
        entry_node, pdf_to_md_node, md_img_node,
        document_split_node, item_name_node,
        bge_embedding_node, import_milvus_node, knowledge_graph_node,
    ]
    for node in all_nodes:
        graph_workflow.add_node(node.name, node)

    # 边：线性管道
    graph_workflow.add_edge("__start__", entry_node.name)
    graph_workflow.add_conditional_edges(
        entry_node.name, route_by_file_type,
        {"pdf_to_md_node": "pdf_to_md_node", "md_img_node": "md_img_node", "__end__": "__end__"},
    )
    graph_workflow.add_edge(pdf_to_md_node.name, md_img_node.name)
    graph_workflow.add_edge(md_img_node.name, document_split_node.name)
    graph_workflow.add_edge(document_split_node.name, item_name_node.name)
    graph_workflow.add_edge(item_name_node.name, bge_embedding_node.name)
    graph_workflow.add_edge(bge_embedding_node.name, import_milvus_node.name)
    graph_workflow.add_edge(import_milvus_node.name, knowledge_graph_node.name)
    graph_workflow.add_edge(knowledge_graph_node.name, "__end__")

    return graph_workflow.compile()

graph = create_import_graph()


def run_import_graph(import_file_path: str, file_dir: str):
    global graph
    state = create_default_state(
        import_file_path=import_file_path,
        file_dir=file_dir,
    )
    final_state = None
    for event in graph.stream(state):
        for node_name, node_state in event.items():
            print(f"运行节点: {node_name}，state: {node_state}")
            final_state = node_state
    return final_state


if __name__ == "__main__":
    setup_logging()
    # import_file_path = (
    #     r"D:\path\to\ProductAssistant\knowledge\processor\import_process"
    #     r"\import_temp_dir\万用表RS-12的使用.pdf"
    # )
    import_file_path = (
        r"D:\path\to\ProductAssistant\knowledge\processor\import_process\output_temp_dir\sample\hybrid_auto\sample.md"
    )

    # file_dir = (
    #     r"D:\path\to\ProductAssistant\knowledge\processor\import_process"
    #     r"\import_temp_dir"
    # )
    file_dir = (
        r"D:\path\to\ProductAssistant\knowledge\processor\import_process\output_temp_dir"
    )
    final_state = run_import_graph(import_file_path, file_dir)
    print(json.dumps(final_state, indent=4, ensure_ascii=False))

    print("=" * 50)
    graph.get_graph().print_ascii()


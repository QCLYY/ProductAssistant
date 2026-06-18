"""FastAPI 入口 — 挂载导入 + 查询路由 + 前端页面"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from knowledge.api.import_file_router import router as import_router
from knowledge.api.query_router import create_query_app

# ---- 文件导入路由（原版，挂在 /import 前缀下）----
app = FastAPI(title="掌柜智库 - 知识库")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(import_router)

# ---- 查询路由（挂载到同一实例）----
query_app = create_query_app()
for route in query_app.routes:
    app.routes.append(route)

# ---- 前端聊天页面 ----
FRONT_DIR = os.path.join(os.path.dirname(__file__), "..", "front")
if os.path.isdir(FRONT_DIR):
    app.mount("/front", StaticFiles(directory=FRONT_DIR, html=True), n

    ame="front")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("knowledge.api.app:app", host="0.0.0.0", port=8000, reload=True)

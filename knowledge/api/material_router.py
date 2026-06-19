"""Material management and runtime check routes."""

from fastapi import APIRouter, HTTPException, Query

from knowledge.services.material_service import material_service

router = APIRouter()


@router.get("/materials")
async def list_materials(limit: int = Query(5000, ge=1, le=20000)):
    return material_service.list_materials(limit=limit)


@router.delete("/materials")
async def delete_material(
    file_title: str = Query("", description="资料文件名"),
    material_name: str = Query("", description="资料识别名称"),
):
    try:
        return material_service.delete_material(
            file_title=file_title,
            material_name=material_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/config/check")
async def check_config():
    return material_service.check_config()

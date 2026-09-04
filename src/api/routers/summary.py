from fastapi import APIRouter

router = APIRouter(prefix="/summary", tags=["Summary"])

@router.get("/daily")
def daily_summary():
    return {"msg": "Generating daily summary..."}

from fastapi import APIRouter

router = APIRouter(prefix="/delivery", tags=["Delivery"])

@router.get("/track")
def track_delivery():
    return {"msg": "Tracking delivery..."}

@router.post("/update")
def update_delivery():
    return {"msg": "Updating delivery status..."}

@router.post("/archive")
def archive_delivery():
    return {"msg": "Archiving completed delivery..."}

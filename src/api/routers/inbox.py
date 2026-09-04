from fastapi import APIRouter

router = APIRouter(prefix="/inbox", tags=["Inbox"])

@router.get("/fetch")
def fetch_inbox():
    return {"msg": "Fetching inbox..."}

@router.post("/process")
def process_inbox():
    return {"msg": "Processing inbox..."}

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.db.database import get_db
from ..utils.inbox_processor import (
    MAX_UNDO_BATCHES,
    process_unprocessed_emails,
    undo_processing_batches,
)

router = APIRouter(prefix="/inbox", tags=["Inbox"])


@router.post("/process_unprocessed")
def process_unprocessed(db: Session = Depends(get_db)):
    batch_id, results, errors = process_unprocessed_emails(db)
    return {
        "batch_id": batch_id,
        "processed": len(results),
        "ner_processed": sum(result["ner_executed"] for result in results),
        "moved": sum("folder" in result for result in results),
        "failed": len(errors),
        "errors": errors,
    }


@router.post("/undo_last_processing")
def undo_processing(
    count: int = Query(1, ge=1, le=MAX_UNDO_BATCHES),
    db: Session = Depends(get_db),
):
    batch_ids, restored, errors = undo_processing_batches(db, count)
    return {
        "batch_ids": batch_ids,
        "restored": len(restored),
        "failed": len(errors),
        "errors": errors,
    }

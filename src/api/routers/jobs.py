"""Endpoints invoked by Cloud Scheduler."""

from datetime import date

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from src.scheduler.workflows import (
    run_daily_workflow,
    run_five_minute_workflow,
    run_gather_summary,
    run_retraining,
    run_send_summary,
)


router = APIRouter(prefix="/jobs", tags=["Scheduled jobs"])


@router.post("/five-minute")
def five_minute_job():
    result = run_five_minute_workflow()
    # Every account was processed and the database persisted; still let Cloud
    # Scheduler see a failed mailbox as a failed run.
    if any(
        "error" in outcome or (outcome.get("reply_status") or {}).get("stopped")
        for outcome in result.get("accounts", {}).values()
    ):
        return JSONResponse(status_code=500, content=jsonable_encoder(result))
    return result


@router.post("/daily")
def daily_job(summary_date: date | None = None):
    return run_daily_workflow(summary_date)


@router.post("/daily/gather")
def gather_daily_job(summary_date: date | None = None, force: bool = False):
    return run_gather_summary(summary_date, force=force)


@router.post("/daily/send")
def send_daily_job(summary_date: date | None = None):
    return run_send_summary(summary_date)


@router.post("/daily/retrain")
def retrain_daily_job(summary_date: date | None = None):
    return run_retraining(summary_date)
"""Endpoints invoked by Cloud Scheduler."""

from fastapi import APIRouter

from src.scheduler.workflows import run_daily_workflow, run_five_minute_workflow


router = APIRouter(prefix="/jobs", tags=["Scheduled jobs"])


@router.post("/five-minute")
def five_minute_job():
    return run_five_minute_workflow()


@router.post("/daily")
def daily_job():
    return run_daily_workflow()
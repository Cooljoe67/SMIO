"""Endpoints invoked by Cloud Scheduler."""

from datetime import date

from fastapi import APIRouter

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
    return run_five_minute_workflow()


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
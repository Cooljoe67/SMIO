"""Legacy local entry point: run both scheduled SMIO routines once."""

from src.scheduler.workflows import run_daily_workflow, run_five_minute_workflow


def run_workflow():
    return {
        "five_minute": run_five_minute_workflow(),
        "daily": run_daily_workflow(),
    }


if __name__ == "__main__":
    run_workflow()

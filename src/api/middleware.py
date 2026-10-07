from fastapi import Request
from fastapi.concurrency import run_in_threadpool

from src.db.database import persist_database


async def publish_database_changes(request: Request, call_next):
    """Upload the database once per request instead of after every commit."""
    response = await call_next(request)
    await run_in_threadpool(persist_database)
    return response

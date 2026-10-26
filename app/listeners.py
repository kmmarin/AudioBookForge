"""Listeners: the people who read along. Each keeps their own place and
checkpoints in every book, on the server, so they follow that person from
device to device. Requests say who's listening with ?listener=<id>."""
from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import db

router = APIRouter()
NAME_MAX = 40
GONE = {"message": "That listener was removed. Choose who’s listening.", "listener_gone": True}


def install(app: FastAPI) -> None:
    app.include_router(router)

    @app.exception_handler(db.ListenerGone)
    async def listener_gone(_: Request, __: db.ListenerGone):
        return JSONResponse({"detail": GONE}, status_code=404)


def resolve(listener: int | None) -> int:
    """Who's listening. No listener means the default one, so a reader page
    opened before listeners existed keeps working. An unknown one is an error,
    never someone else: a tab left open must not save over another's place."""
    if listener is None:
        return db.default_listener()
    if db.get_listener(listener) is None:
        raise db.ListenerGone(listener)
    return listener


def _name(raw: str) -> str:
    name = " ".join((raw or "").split())[:NAME_MAX]
    if not name:
        raise HTTPException(400, "Type a name.")
    return name


class ListenerBody(BaseModel):
    name: str


@router.get("/api/listeners")
def list_listeners():
    """Everyone, oldest first. The first is the default listener."""
    return db.list_listeners()


@router.post("/api/listeners")
def add_listener(body: ListenerBody):
    name = _name(body.name)
    try:
        new_id = db.add_listener(name)
    except db.NameTaken as taken:
        raise HTTPException(409, f"There’s already a listener called {taken}. Try adding a surname or an initial.")
    return db.get_listener(new_id)


@router.patch("/api/listeners/{listener_id}")
def rename_listener(listener_id: int, body: ListenerBody):
    name = _name(body.name)
    try:
        db.rename_listener(listener_id, name)
    except db.NameTaken as taken:
        raise HTTPException(409, f"There’s already a listener called {taken}. Pick another name.")
    return db.get_listener(listener_id)


@router.delete("/api/listeners/{listener_id}")
def delete_listener(listener_id: int):
    """Also deletes their place and checkpoints in every book."""
    try:
        db.delete_listener(listener_id)
    except ValueError:
        raise HTTPException(409, "Forge needs at least one listener, so the last one can’t be removed. Rename it instead.")
    return {"deleted": listener_id}

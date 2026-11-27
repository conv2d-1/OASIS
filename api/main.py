from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.factory import agent_catalog
from api.game_service import GameManager

ROOT = Path(__file__).resolve().parents[1]
manager = GameManager(ROOT)
app = FastAPI(title="OASIS API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlayerConfig(BaseModel):
    type: str
    simulations: int = Field(default=50, ge=1, le=2000)
    depth: int = Field(default=4, ge=1, le=32)
    checkpoint: str | None = None


class CreateGameRequest(BaseModel):
    board_size: Literal[6, 8] = 8
    black: PlayerConfig
    white: PlayerConfig


class MoveRequest(BaseModel):
    action: int = Field(ge=0)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/agents")
def list_agents():
    return {"agents": agent_catalog(ROOT)}


@app.post("/api/games")
def create_game(request: CreateGameRequest):
    players = {"black": request.black.model_dump(), "white": request.white.model_dump()}
    try:
        session = manager.create(request.board_size, players)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return manager.response(session)


@app.get("/api/games/{game_id}")
def get_game(game_id: str):
    try:
        return manager.response(manager.get(game_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/games/{game_id}/moves")
def apply_move(game_id: str, request: MoveRequest):
    try:
        session = manager.get(game_id)
        side = "black" if session.state.current_player == 1 else "white"
        if session.agents[side] is not None:
            raise ValueError("current player is an agent")
        manager.apply(session, request.action, "human")
        return manager.response(session)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/games/{game_id}/step")
def agent_step(game_id: str):
    try:
        session = manager.get(game_id)
        manager.step(session)
        return manager.response(session)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/games/{game_id}/end")
def end_game(game_id: str):
    try:
        session = manager.end(manager.get(game_id))
        return manager.response(session)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

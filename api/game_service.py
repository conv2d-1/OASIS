from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from agents.factory import build_agent, choose_action
from othello.game import OthelloGame
from othello.state import GameState


@dataclass
class GameSession:
    game_id: str
    game: OthelloGame
    state: GameState
    players: dict
    agents: dict
    moves: list[dict] = field(default_factory=list)

    @property
    def finished(self):
        return self.state.status == "finished"


class GameManager:
    def __init__(self, root: Path):
        self.root = root
        self.sessions: dict[str, GameSession] = {}

    def create(self, board_size: int, players: dict) -> GameSession:
        game = OthelloGame(board_size)
        agents = {
            side: build_agent(self.root, game, config)
            for side, config in players.items()
        }
        session = GameSession(
            game_id=uuid4().hex,
            game=game,
            state=game.getInitialState(),
            players=players,
            agents=agents,
        )
        self.sessions[session.game_id] = session
        return session

    def get(self, game_id: str) -> GameSession:
        try:
            return self.sessions[game_id]
        except KeyError as exc:
            raise KeyError(f"game not found: {game_id}") from exc

    def apply(self, session: GameSession, action: int, source: str, inference_ms: float | None = None) -> GameSession:
        if session.finished:
            raise ValueError("game is already finished")
        valid = session.game.getValidMoves(session.state.board, session.state.current_player)
        if action < 0 or action >= len(valid) or not valid[action]:
            raise ValueError("illegal action")
        player = session.state.current_player
        next_state = session.game.nextState(session.state, action)
        session.state = next_state
        session.moves.append({
            "step": len(session.moves) + 1,
            "player": player,
            "action": action,
            "source": source,
            "inference_ms": inference_ms,
            "board": next_state.board.tolist(),
        })
        return session

    def step(self, session: GameSession) -> GameSession:
        if session.finished:
            raise ValueError("game is already finished")
        side = "black" if session.state.current_player == 1 else "white"
        agent = session.agents[side]
        if agent is None:
            raise ValueError("current player is human")
        started_at = perf_counter()
        action = choose_action(agent, session.game.getCanonicalForm(session.state.board, session.state.current_player))
        inference_ms = (perf_counter() - started_at) * 1000
        return self.apply(session, action, "agent", inference_ms=inference_ms)

    def end(self, session: GameSession) -> GameSession:
        session.state = GameState(
            board=session.state.board,
            current_player=session.state.current_player,
            move_number=session.state.move_number,
            last_action=session.state.last_action,
            status="finished",
            result=session.state.result,
        )
        return session

    def response(self, session: GameSession) -> dict:
        board = session.state.board
        valid = [] if session.finished else [int(i) for i, value in enumerate(session.game.getValidMoves(board, session.state.current_player)) if value]
        side = "black" if session.state.current_player == 1 else "white"
        return {
            "id": session.game_id,
            "board_size": session.game.n,
            "board": board.tolist(),
            "current_player": session.state.current_player,
            "current_side": side,
            "legal_actions": valid,
            "move_number": session.state.move_number,
            "last_action": session.state.last_action,
            "status": session.state.status,
            "result": session.state.result,
            "players": session.players,
            "moves": session.moves,
        }

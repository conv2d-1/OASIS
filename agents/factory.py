"""Agent construction helpers for the API service."""

from pathlib import Path

from agents.alphabeta_agent import AlphaBetaAgent
from agents.alphazero_agent import AlphaZeroAgent
from agents.greedy_agent import GreedyOthelloPlayer
from agents.mcts_agent import MCTSAgent
from agents.random_agent import RandomPlayer

AGENT_TYPES = {
    "human": {"label": "真人", "requires_checkpoint": False, "available": True},
    "mcts": {"label": "MCTS", "requires_checkpoint": False, "available": True},
    "alphabeta": {"label": "Alpha-Beta", "requires_checkpoint": False, "available": True},
    "random": {"label": "随机", "requires_checkpoint": False, "available": True},
    "greedy": {"label": "贪心", "requires_checkpoint": False, "available": True},
    "alphazero": {"label": "AlphaZero", "requires_checkpoint": True, "available": False},
    "rgsc": {"label": "RGSC", "requires_checkpoint": True, "available": False},
}


def discover_checkpoints(root: Path) -> list[str]:
    paths = []
    for base in (root / "results", root / "models"):
        if base.exists():
            paths.extend(path.relative_to(root).as_posix() for path in base.rglob("*.pth.tar"))
    return sorted(set(paths))


def agent_catalog(root: Path) -> list[dict]:
    checkpoints = discover_checkpoints(root)
    return [
        {
            "type": agent_type,
            "label": meta["label"],
            "available": bool(meta["available"] or (agent_type == "alphazero" and checkpoints)),
            "requires_checkpoint": meta["requires_checkpoint"],
            "checkpoints": checkpoints if agent_type in {"alphazero", "rgsc"} else [],
        }
        for agent_type, meta in AGENT_TYPES.items()
    ]


def resolve_checkpoint(root: Path, checkpoint: str | None) -> Path:
    if not checkpoint:
        raise FileNotFoundError("checkpoint is required")
    candidate = Path(checkpoint)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    candidate = root / checkpoint
    if candidate.is_file():
        return candidate
    for base in (root / "results", root / "models"):
        if base.exists():
            matches = list(base.rglob(candidate.name))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"checkpoint not found: {checkpoint}")


def build_agent(root: Path, game, config: dict):
    agent_type = config["type"]
    if agent_type == "human":
        return None
    if agent_type == "random":
        return RandomPlayer(game)
    if agent_type == "greedy":
        return GreedyOthelloPlayer(game)
    if agent_type == "mcts":
        return MCTSAgent(game, simulations=config.get("simulations", 50))
    if agent_type == "alphabeta":
        return AlphaBetaAgent(game, depth=config.get("depth", 4))
    if agent_type == "alphazero":
        checkpoint = resolve_checkpoint(root, config.get("checkpoint"))
        return AlphaZeroAgent(game, checkpoint, config.get("simulations", 50))
    if agent_type == "rgsc":
        raise NotImplementedError("RGSC agent is not implemented yet")
    raise ValueError(f"unknown agent type: {agent_type}")


def choose_action(agent, board):
    if hasattr(agent, "choose_action"):
        return int(agent.choose_action(board))
    if hasattr(agent, "play"):
        return int(agent.play(board))
    return int(agent(board))

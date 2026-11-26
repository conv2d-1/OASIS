from dataclasses import dataclass, replace
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class GameState:
    """Serializable game state shared by future API and replay code."""

    board: np.ndarray
    current_player: int = 1
    move_number: int = 0
    last_action: Optional[int] = None
    status: str = "ongoing"
    result: Optional[float] = None

    def copy(self) -> "GameState":
        """Return a state copy with an independent board array."""
        return replace(self, board=np.copy(self.board))

    def to_dict(self) -> dict:
        """Return a JSON-friendly representation for API and replay code."""
        return {
            "board": self.board.tolist(),
            "current_player": self.current_player,
            "move_number": self.move_number,
            "last_action": self.last_action,
            "status": self.status,
            "result": self.result,
        }

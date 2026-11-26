from pathlib import Path
from types import SimpleNamespace

import numpy as np

from models.networks import NNetWrapper
from search.mcts import MCTS


class AlphaZeroAgent:
    """Saved-model player migrated from the original CLI implementation."""

    def __init__(self, game, checkpoint, simulations):
        self.game = game
        self.model = NNetWrapper(game)
        path = Path(checkpoint)
        self.model.load_checkpoint(path.parent, path.name)
        self.args = SimpleNamespace(numMCTSSims=simulations, cpuct=1.0)
        self.search = None

    def startGame(self):
        self.search = MCTS(self.game, self.model, self.args)

    def __call__(self, board):
        if self.search is None:
            self.startGame()
        return np.argmax(self.search.getActionProb(board, temp=0))

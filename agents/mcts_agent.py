from types import SimpleNamespace

import numpy as np

from search.mcts import MCTS


class HeuristicEvaluator:
    """Policy/value adapter used by the parameter-only MCTS baseline."""

    def __init__(self, game):
        self.game = game

    def predict(self, board):
        valid = self.game.getValidMoves(board, 1).astype(np.float64)
        policy = valid / valid.sum()
        value = np.tanh(self.game.getScore(board, 1) / max(1, self.game.n * self.game.n))
        return policy, float(value)


class MCTSAgent:
    def __init__(self, game, simulations=50, cpuct=1.0):
        self.game = game
        self.evaluator = HeuristicEvaluator(game)
        self.args = SimpleNamespace(numMCTSSims=max(1, int(simulations)), cpuct=float(cpuct))
        self.search = MCTS(game, self.evaluator, self.args)

    def choose_action(self, board):
        probabilities = self.search.getActionProb(board, temp=0)
        return int(np.argmax(probabilities))

    def __call__(self, board):
        return self.choose_action(board)

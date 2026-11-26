import numpy as np


class GreedyOthelloPlayer:
    """One-step piece-count player retained from the original CLI implementation."""

    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for action in np.flatnonzero(valids):
            next_board, _ = self.game.getNextState(board, 1, action)
            candidates.append((-self.game.getScore(next_board, 1), action))
        return min(candidates)[1]

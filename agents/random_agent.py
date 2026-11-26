import numpy as np


class RandomPlayer:
    """Random legal-action player retained from the original CLI implementation."""

    def __init__(self, game):
        self.game = game

    def play(self, board):
        return np.random.choice(np.flatnonzero(self.game.getValidMoves(board, 1)))

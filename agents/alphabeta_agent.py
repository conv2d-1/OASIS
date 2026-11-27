import math

import numpy as np


class AlphaBetaAgent:
    def __init__(self, game, depth=4):
        self.game = game
        self.depth = max(1, int(depth))

    def choose_action(self, board):
        valid = np.flatnonzero(self.game.getValidMoves(board, 1))
        best_action = int(valid[0])
        best_value = -math.inf
        for action in valid:
            next_board, next_player = self.game.getNextState(board, 1, int(action))
            canonical = self.game.getCanonicalForm(next_board, next_player)
            value = -self._search(canonical, self.depth - 1, -math.inf, math.inf)
            if value > best_value:
                best_value = value
                best_action = int(action)
        return best_action

    def _search(self, board, depth, alpha, beta):
        result = self.game.getGameEnded(board, 1)
        if result != 0:
            return float(result if abs(result) == 1 else 0)
        if depth <= 0:
            return float(np.tanh(self.game.getScore(board, 1) / max(1, self.game.n * self.game.n)))

        valid = np.flatnonzero(self.game.getValidMoves(board, 1))
        value = -math.inf
        for action in valid:
            next_board, next_player = self.game.getNextState(board, 1, int(action))
            canonical = self.game.getCanonicalForm(next_board, next_player)
            value = max(value, -self._search(canonical, depth - 1, -beta, -alpha))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    def __call__(self, board):
        return self.choose_action(board)

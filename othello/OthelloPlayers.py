import numpy as np


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        return np.random.choice(np.flatnonzero(self.game.getValidMoves(board, 1)))


class HumanOthelloPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        pass_action = self.game.getActionSize() - 1
        if valid[pass_action]:
            print("No legal placement. Passing.")
            return pass_action

        size = self.game.n
        print("Legal moves (row, column):", [divmod(int(action), size) for action in np.flatnonzero(valid)])
        while True:
            try:
                row, column = map(int, input("Enter row and column: ").split())
            except ValueError:
                print("Enter two integers.")
                continue
            if 0 <= row < size and 0 <= column < size:
                action = row * size + column
                if valid[action]:
                    return action
            print("Illegal move.")


class GreedyOthelloPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for action in np.flatnonzero(valids):
            next_board, _ = self.game.getNextState(board, 1, action)
            candidates.append((-self.game.getScore(next_board, 1), action))
        return min(candidates)[1]

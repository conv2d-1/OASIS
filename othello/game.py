import numpy as np

class Game:
    # TODO: Replace this legacy AlphaZero-General interface with a typed protocol.
    def __init__(self):
        pass

    def getInitBoard(self):
        pass

    def getBoardSize(self):
        pass

    def getActionSize(self):
        pass

    def getNextState(self, board, player, action):
        pass

    def getValidMoves(self, board, player):
        pass

    def getGameEnded(self, board, player):
        pass

    def getCanonicalForm(self, board, player):
        pass

    def getSymmetries(self, board, pi):
        pass

    def stringRepresentation(self, board):
        pass
from .board import Board
from .rules import apply_action, canonical_form, game_ended, score, symmetries, valid_moves
from .state import GameState


class OthelloGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    @staticmethod
    def getSquarePiece(piece):
        return OthelloGame.square_content[piece]

    def __init__(self, n):
        self.n = n

    def getInitialState(self) -> GameState:
        """Return the explicit state representation used by API and replay code."""
        return GameState(board=self.getInitBoard())

    def nextState(self, state: GameState, action: int) -> GameState:
        """Apply an action to an explicit state while preserving the legacy API."""
        board, player = self.getNextState(state.board, state.current_player, action)
        result = self.getGameEnded(board, player)
        return GameState(
            board=np.copy(board),
            current_player=player,
            move_number=state.move_number + 1,
            last_action=action,
            status="finished" if result != 0 else "ongoing",
            result=None if result == 0 else float(result),
        )

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        return apply_action(board, player, action)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        return valid_moves(board, player)

    def getGameEnded(self, board, player):
        """Return 0 for ongoing play, ±1 for a win/loss, or 1e-4 for a draw."""
        return game_ended(board, player)

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return canonical_form(board, player)

    def getSymmetries(self, board, pi):
        # mirror, rotational
        return symmetries(board, pi)

    def stringRepresentation(self, board):
        return board.tobytes()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        return score(board, player)

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(OthelloGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")


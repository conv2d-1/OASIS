"""Pure black-and-white board rules used by OthelloGame."""

import numpy as np

from .board import Board

DRAW_VALUE = 1e-4


def _board_copy(board: np.ndarray) -> Board:
    """Create the legacy Board helper from an array without mutating it."""
    result = Board(board.shape[0])
    result.pieces = np.copy(board)
    return result


def legal_moves(board: np.ndarray, player: int) -> list[tuple[int, int]]:
    """Return legal placement coordinates for a player."""
    return _board_copy(board).get_legal_moves(player)


def has_legal_moves(board: np.ndarray, player: int) -> bool:
    """Return whether a player can place a piece."""
    return _board_copy(board).has_legal_moves(player)


def apply_action(board: np.ndarray, player: int, action: int) -> tuple[np.ndarray, int]:
    """Apply a placement or pass action and return the next board and player."""
    board_size = board.shape[0]
    pass_action = board_size * board_size
    if action == pass_action:
        return board, -player

    result = _board_copy(board)
    result.execute_move(divmod(action, board_size), player)
    return result.pieces, -player


def valid_moves(board: np.ndarray, player: int) -> np.ndarray:
    """Return the fixed-size legal-action mask including the pass action."""
    board_size = board.shape[0]
    valids = [0] * (board_size * board_size + 1)
    moves = legal_moves(board, player)
    if not moves:
        valids[-1] = 1
        return np.array(valids)
    for row, column in moves:
        valids[board_size * row + column] = 1
    return np.array(valids)


def game_ended(board: np.ndarray, player: int) -> float:
    """Return zero for an ongoing game, a win or loss, or the draw value."""
    if has_legal_moves(board, player) or has_legal_moves(board, -player):
        return 0
    difference = _board_copy(board).countDiff(player)
    if difference == 0:
        return DRAW_VALUE
    return 1 if difference > 0 else -1


def canonical_form(board: np.ndarray, player: int) -> np.ndarray:
    """Return the board from the current player's perspective."""
    return player * board


def symmetries(board: np.ndarray, pi: list[float]) -> list[tuple[np.ndarray, list[float]]]:
    """Return rotational and mirror symmetries for training augmentation."""
    board_size = board.shape[0]
    assert len(pi) == board_size**2 + 1
    pi_board = np.reshape(pi[:-1], (board_size, board_size))
    forms = []
    for rotation in range(1, 5):
        for mirrored in [True, False]:
            new_board = np.rot90(board, rotation)
            new_pi = np.rot90(pi_board, rotation)
            if mirrored:
                new_board = np.fliplr(new_board)
                new_pi = np.fliplr(new_pi)
            forms.append((new_board, list(new_pi.ravel()) + [pi[-1]]))
    return forms


def score(board: np.ndarray, player: int) -> int:
    """Return the piece-count difference from a player's perspective."""
    return _board_copy(board).countDiff(player)

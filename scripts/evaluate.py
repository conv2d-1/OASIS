"""Play Othello in the terminal with humans, baselines, or saved models."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.alphazero_agent import AlphaZeroAgent
from agents.greedy_agent import GreedyOthelloPlayer
from agents.random_agent import RandomPlayer
from agents.base import HumanOthelloPlayer
from othello.game import OthelloGame
from training.evaluator import Arena


def make_player(spec, game, simulations):
    baselines = {
        "human": HumanOthelloPlayer,
        "random": RandomPlayer,
        "greedy": GreedyOthelloPlayer,
    }
    if spec in baselines:
        return baselines[spec](game).play

    return AlphaZeroAgent(game, Path(spec), simulations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("player1", nargs="?", default="greedy", help="human, random, greedy, or a checkpoint path")
    parser.add_argument("player2", nargs="?", default="random", help="human, random, greedy, or a checkpoint path")
    parser.add_argument("--board-size", type=int, choices=(6, 8), default=8)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--verbose", action="store_true", help="Print the board at each turn")
    options = parser.parse_args()
    if min(options.games, options.simulations) < 1:
        parser.error("games and simulations must be positive")

    game = OthelloGame(options.board_size)
    player1 = make_player(options.player1, game, options.simulations)
    player2 = make_player(options.player2, game, options.simulations)
    arena = Arena(player1, player2, game, display=game.display)
    verbose = options.verbose or "human" in (options.player1, options.player2)
    wins1, wins2, draws = arena.playGames(options.games, verbose)
    print(f"{options.player1} wins: {wins1} | {options.player2} wins: {wins2} | Draws: {draws}")


if __name__ == "__main__":
    main()

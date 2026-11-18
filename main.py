"""Train an Othello AlphaZero model through self-play."""

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

from Coach import Coach
from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-size", type=int, choices=(6, 8), default=8)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=100, help="Self-play games per iteration")
    parser.add_argument("--simulations", type=int, default=25, help="MCTS simulations per turn")
    parser.add_argument("--arena-games", type=int, default=40, help="Evaluation games per iteration")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per iteration")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--load-model", type=Path, help="Start training from an existing model")
    options = parser.parse_args()
    if min(options.iterations, options.episodes, options.simulations, options.epochs) < 1:
        parser.error("iterations, episodes, simulations and epochs must be positive")
    if options.arena_games < 2 or options.arena_games % 2:
        parser.error("arena-games must be a positive even number")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    game = OthelloGame(options.board_size)
    model = NNetWrapper(game, epochs=options.epochs)
    if options.load_model is not None:
        model.load_checkpoint(options.load_model.parent, options.load_model.name)

    args = SimpleNamespace(
        numIters=options.iterations,
        numEps=options.episodes,
        tempThreshold=15,
        updateThreshold=0.6,
        maxlenOfQueue=200000,
        numMCTSSims=options.simulations,
        arenaCompare=options.arena_games,
        cpuct=1.0,
        checkpoint=options.checkpoint_dir,
        numItersForTrainExamplesHistory=20,
    )
    Coach(game, model, args).learn()


if __name__ == "__main__":
    main()

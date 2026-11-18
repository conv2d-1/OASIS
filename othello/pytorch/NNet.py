from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from NeuralNet import NeuralNet
from .OthelloNNet import OthelloNNet

args = SimpleNamespace(lr=0.001, dropout=0.3, epochs=10, batch_size=64, num_channels=512)


class NNetWrapper(NeuralNet):
    def __init__(self, game, *, epochs=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nnet = OthelloNNet(game, args).to(self.device)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.epochs = args.epochs if epochs is None else epochs

    def train(self, examples):
        """Fit the policy and value heads on (board, policy, value) examples."""
        if not examples:
            raise ValueError("Training requires at least one example")
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)
        self.nnet.train()
        # The fully connected batch normalization layers require at least two samples.
        batch_size = max(2, min(args.batch_size, len(examples)))
        batch_count = max(1, len(examples) // batch_size)
        for epoch in range(self.epochs):
            progress = tqdm(range(batch_count), desc=f"Training epoch {epoch + 1}/{self.epochs}")
            for _ in progress:
                sample_ids = np.random.randint(len(examples), size=batch_size)
                boards, pis, vs = zip(*(examples[i] for i in sample_ids))
                boards = torch.as_tensor(np.array(boards), dtype=torch.float32, device=self.device)
                target_pis = torch.as_tensor(np.array(pis), dtype=torch.float32, device=self.device)
                target_vs = torch.as_tensor(np.array(vs), dtype=torch.float32, device=self.device)

                out_pi, out_v = self.nnet(boards)
                policy_loss = self.loss_pi(target_pis, out_pi)
                value_loss = self.loss_v(target_vs, out_v)
                progress.set_postfix(loss_pi=policy_loss.item(), loss_v=value_loss.item())
                optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                optimizer.step()

    def predict(self, board):
        board = torch.tensor(board.copy(), dtype=torch.float32, device=self.device)
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, value = self.nnet(board)
        return pi.exp().cpu().numpy()[0], value.item()

    def loss_pi(self, targets, outputs):
        return -(targets * outputs).sum(dim=1).mean()

    def loss_v(self, targets, outputs):
        return ((targets - outputs.view(-1)) ** 2).mean()

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = Path(folder) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=filepath.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with temporary.open("wb") as stream:
                torch.save({"state_dict": self.nnet.state_dict()}, stream)
            temporary.replace(filepath)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = Path(folder) / filename
        with filepath.open("rb") as stream:
            checkpoint = torch.load(stream, map_location=self.device, weights_only=True)
        self.nnet.load_state_dict(checkpoint["state_dict"])

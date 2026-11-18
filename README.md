# OASIS

**Othello Arena for Search and Interactive Self-play**

OASIS 是用于研究黑白棋强化学习的项目。当前实现提供 AlphaZero 训练和命令行对弈，作为后续网页人机对战和模型观战的基础。

## 安装

在项目目录下安装 NumPy、PyTorch 和 tqdm。PyTorch 会在检测到 CUDA 时使用 GPU，否则使用 CPU。

```powershell
python -m pip install -r requirements.txt
```

## 命令行对弈

```powershell
python pit.py greedy random --games 2 --verbose
python pit.py human greedy
```

棋盘默认 8×8，可以用 `--board-size 6` 切换到小棋盘。人类玩家输入以 0 开始的「行 列」，无合法落点时自动跳过。玩家参数支持 `human`、`random`、`greedy` 或 PyTorch 模型文件路径。指定两份模型文件即可对弈，使用 `--verbose` 查看每一步。模型所用的棋盘尺寸须与命令指定的尺寸一致。

```powershell
python pit.py human checkpoints/best.pth.tar
python pit.py checkpoints/checkpoint_1.pth.tar checkpoints/best.pth.tar --games 2 --verbose
```

## 自博弈训练

```powershell
python main.py --help
python main.py --board-size 6 --iterations 1 --episodes 1 --simulations 2 --arena-games 2 --epochs 1 --checkpoint-dir checkpoints/smoke
```

第二条命令用较小预算验证自博弈、训练和模型评估流程。候选模型通过评估后，才会写入该目录下的 `checkpoint_<迭代号>.pth.tar` 和 `best.pth.tar`。项目未附带预训练权重。可用 `--load-model` 从已有权重开始新的训练，并用 `--checkpoint-dir` 指定本次实验的输出目录。

## 代码结构

| 路径 | 职责 |
| --- | --- |
| `othello/OthelloGame.py`、`othello/OthelloLogic.py` | 落子、翻子、跳过回合、终局及棋盘对称变换 |
| `MCTS.py` | 用策略和价值网络引导蒙特卡洛树搜索 |
| `Coach.py` | 收集自博弈样本、训练候选模型和评估更新 |
| `Arena.py`、`othello/OthelloPlayers.py` | 运行对局和统计胜平负，提供人工与基线玩家 |
| `othello/pytorch/` | 策略与价值网络、训练、推理和模型权重读写 |

核心实现来自 [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general/tree/f1a78e0505c614205f8c60eaa82cd644a07a84b3)，沿用上游的 MIT 许可证，见 [LICENSE](LICENSE)。

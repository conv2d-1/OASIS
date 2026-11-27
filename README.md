# OASIS

**Othello Arena for Search and Interactive Self-play**

OASIS 用于在黑白棋上研究强化学习、搜索算法和自博弈方法。当前仓库保留可运行的 AlphaZero 训练、模型评估和命令行对战实现。RGSC、Web API、前端和 Docker 部分已建立目录，并在对应文件中标记 TODO。

## 安装

```powershell
python -m pip install -r requirements.txt
```

## 命令行对战

使用模块方式运行：

```powershell
python -m scripts.evaluate greedy random --games 2 --verbose
python -m scripts.evaluate human greedy
```

直接运行脚本也支持：

```powershell
python scripts/evaluate.py greedy random --games 2
```

玩家参数支持 human、random、greedy 或 PyTorch checkpoint 路径。棋盘默认是 8×8，可以用 --board-size 6 切换到 6×6。

```powershell
python -m scripts.evaluate human results/best.pth.tar
python -m scripts.evaluate results/checkpoint_1.pth.tar results/best.pth.tar --games 2 --verbose
```

## AlphaZero 自博弈训练

```powershell
python -m scripts.train --help
python -m scripts.train --board-size 6 --iterations 1 --episodes 1 --simulations 2 --arena-games 2 --epochs 1 --checkpoint-dir results/smoke
```

训练流程包括自博弈、对称数据增强、策略价值网络训练、新旧模型竞技和 checkpoint 保存。已有模型可以通过 --load-model 加载。

## 代码结构

```text
OASIS/
├── othello/
│   ├── board.py
│   ├── rules.py
│   ├── game.py
│   └── state.py
├── agents/
│   ├── base.py
│   ├── random_agent.py
│   ├── greedy_agent.py
│   ├── mcts_agent.py
│   ├── alphazero_agent.py
│   └── rgsc_agent.py
├── search/
│   └── mcts.py
├── training/
│   ├── self_play.py
│   ├── trainer.py
│   ├── coach.py
│   ├── replay_buffer.py
│   └── evaluator.py
├── models/
│   ├── networks.py
│   ├── registry.py
│   └── checkpoints.py
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── game_service.py
│   └── model_service.py
├── frontend/
├── configs/
├── tests/
├── scripts/
├── results/
├── LICENSE
├── README.md
└── requirements.txt
```

当前已经迁移的实现包括：

- othello/board.py 中的底层棋盘存储和翻子逻辑
- othello/rules.py 中的合法动作、状态转移、终局、对称和计分规则
- othello/state.py 中的统一棋局状态对象
- othello/game.py 中的 AlphaZero 游戏接口和黑白棋实现
- agents/ 中的人工、随机、贪心和 AlphaZero checkpoint 玩家
- search/mcts.py 中的 MCTS
- training/coach.py 和 training/evaluator.py 中的训练和竞技场
- models/networks.py 中的 PyTorch 策略价值网络
- scripts/train.py 和 scripts/evaluate.py 中的命令行入口

尚未实现的内容集中标记为 TODO，包括 RGSC、统一 Agent 接口、模型注册表、Web API、前端、实验配置、模型导出和自动化测试。

核心实现来自 [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general/tree/f1a78e0505c614205f8c60eaa82cd644a07a84b3)，沿用上游 MIT 许可证，见 LICENSE。

## API 与前端联调

安装依赖后启动 API：

```powershell
python -m pip install -r requirements.txt
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

另开终端启动前端：

```powershell
python -m http.server 4173 --bind 127.0.0.1 --directory frontend
```

访问 `http://127.0.0.1:4173/`。前端通过 API 选择真人、MCTS、Alpha-Beta、随机、贪心、AlphaZero 和 RGSC。AlphaZero 需要可发现的 checkpoint，RGSC 当前会返回未实现状态。

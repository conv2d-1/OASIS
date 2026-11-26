const SIZE = 8;
const PASS = SIZE * SIZE;

const state = {
  board: createInitialBoard(),
  currentPlayer: 1,
  mode: "human",
  gameStarted: false,
  paused: false,
  autoPlay: true,
  aiTimer: null,
  gameOver: false,
  lastAction: null,
  simulations: 50,
  players: {
    black: { type: "human", simulations: 50, depth: 4, checkpoint: "best.pth.tar" },
    white: { type: "alphazero", simulations: 50, depth: 4, checkpoint: "best.pth.tar" },
  },
  events: [],
};

const boardEl = document.querySelector("#board");
const eventListEl = document.querySelector("#eventList");
const setupOverlay = document.querySelector("#setupOverlay");
let squareElements = [];
let renderedBoard = null;

function createInitialBoard() {
  const board = Array.from({ length: SIZE }, () => Array(SIZE).fill(0));
  board[3][3] = -1;
  board[3][4] = 1;
  board[4][3] = 1;
  board[4][4] = -1;
  return board;
}

function directions() {
  return [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
}

function inside(row, col) {
  return row >= 0 && row < SIZE && col >= 0 && col < SIZE;
}

function captures(board, row, col, player) {
  if (!inside(row, col) || board[row][col] !== 0) return [];
  const captured = [];
  directions().forEach(function (direction) {
    const line = [];
    let r = row + direction[0];
    let c = col + direction[1];
    while (inside(r, c) && board[r][c] === -player) {
      line.push([r, c]);
      r += direction[0];
      c += direction[1];
    }
    if (line.length && inside(r, c) && board[r][c] === player) captured.push.apply(captured, line);
  });
  return captured;
}

function legalActions(board, player) {
  const actions = [];
  for (let row = 0; row < SIZE; row += 1) {
    for (let col = 0; col < SIZE; col += 1) {
      if (captures(board, row, col, player).length) actions.push(row * SIZE + col);
    }
  }
  return actions.length ? actions : [PASS];
}

function applyAction(board, player, action) {
  if (action === PASS) return { board: board, nextPlayer: -player };
  const row = Math.floor(action / SIZE);
  const col = action % SIZE;
  const next = board.map(function (line) { return line.slice(); });
  const captured = captures(next, row, col, player);
  next[row][col] = player;
  captured.forEach(function (position) { next[position[0]][position[1]] = player; });
  return { board: next, nextPlayer: -player };
}

function countPieces(board) {
  return board.flat().reduce(function (counts, value) {
    if (value === 1) counts.black += 1;
    if (value === -1) counts.white += 1;
    return counts;
  }, { black: 0, white: 0 });
}

function isFinished(board) {
  return legalActions(board, 1)[0] === PASS && legalActions(board, -1)[0] === PASS;
}

function actionName(action) {
  if (action === PASS) return "Pass";
  return String.fromCharCode(65 + (action % SIZE)) + (Math.floor(action / SIZE) + 1);
}

const MODEL_LABELS = {
  human: "真人",
  mcts: "MCTS",
  alphabeta: "Alpha-Beta",
  random: "随机",
  greedy: "贪心",
  alphazero: "AlphaZero",
  rgsc: "RGSC",
};

function playerKey(player) {
  return player === 1 ? "black" : "white";
}

function playerConfig(player) {
  return state.players[playerKey(player)];
}

function modelName(side) {
  return MODEL_LABELS[playerConfig(side).type];
}

function modelSummary(config) {
  const label = MODEL_LABELS[config.type];
  if (config.type === "mcts" || config.type === "alphabeta") return label + " · " + config.simulations + " sims / depth " + config.depth;
  if (config.type === "alphazero" || config.type === "rgsc") return label + " · " + config.checkpoint + " / " + config.simulations + " sims";
  return label;
}

function deriveMode() {
  const humanCount = [state.players.black, state.players.white].filter(function (config) { return config.type === "human"; }).length;
  return humanCount === 0 ? "ai" : "human";
}

function chooseAiAction(player) {
  const config = playerConfig(player);
  const actions = legalActions(state.board, player);
  if (actions.length === 1) return actions[0];
  if (config.type === "random") return actions[Math.floor(Math.random() * actions.length)];
  if (config.type === "greedy") return actions.slice().sort(function (a, b) {
    return captures(state.board, Math.floor(b / SIZE), b % SIZE, player).length - captures(state.board, Math.floor(a / SIZE), a % SIZE, player).length;
  })[0];

  // Local demo evaluator. Real MCTS, Alpha-Beta, AlphaZero, and RGSC will be called through api/ later.
  const corners = [0, SIZE - 1, SIZE * (SIZE - 1), PASS - SIZE];
  const preferred = actions.filter(function (action) { return corners.includes(action); });
  if (preferred.length) return preferred[0];
  return actions.slice().sort(function (a, b) {
    const score = function (action) {
      const row = Math.floor(action / SIZE);
      const col = action % SIZE;
      const captureScore = captures(state.board, row, col, player).length;
      const edgeScore = (row === 0 || row === SIZE - 1 ? 1 : 0) + (col === 0 || col === SIZE - 1 ? 1 : 0);
      const depthWeight = config.type === "alphabeta" ? config.depth : config.type === "mcts" ? Math.min(config.depth, 8) : 1;
      return captureScore * depthWeight + edgeScore;
    };
    return score(b) - score(a);
  })[0];
}

function configMarkup(side) {
  const config = state.players[side];
  if (["human", "random", "greedy"].includes(config.type)) {
    return '<div class="setup-config-empty">' + (config.type === "human" ? "由真人点击棋盘落子。" : "无需训练参数，直接使用本地规则。") + '</div>';
  }
  if (config.type === "mcts" || config.type === "alphabeta") {
    return '<div class="setup-param-grid"><div class="setup-param"><label>模拟次数<input type="number" min="10" max="500" step="10" data-config-key="simulations" value="' + config.simulations + '"></label></div><div class="setup-param"><label>搜索深度<input type="number" min="1" max="16" step="1" data-config-key="depth" value="' + config.depth + '"></label></div></div>';
  }
  return '<div class="setup-param-grid"><div class="setup-param"><label>Checkpoint<select data-config-key="checkpoint"><option>best.pth.tar</option><option>checkpoint_10.pth.tar</option><option>checkpoint_20.pth.tar</option></select></label></div><div class="setup-param"><label>MCTS 模拟<input type="number" min="10" max="500" step="10" data-config-key="simulations" value="' + config.simulations + '"></label></div></div>';
}

function renderSetupConfig(side) {
  const config = state.players[side];
  const configEl = document.querySelector(side === "black" ? "#blackConfig" : "#whiteConfig");
  configEl.innerHTML = configMarkup(side);
  const checkpoint = configEl.querySelector('[data-config-key="checkpoint"]');
  if (checkpoint) checkpoint.value = config.checkpoint;
  configEl.querySelectorAll("[data-config-key]").forEach(function (input) {
    input.addEventListener("input", function () {
      const value = input.type === "number" ? Number(input.value) : input.value;
      state.players[side][input.dataset.configKey] = value;
    });
  });
}

function renderSetup() {
  document.querySelector("#blackModelSetup").value = state.players.black.type;
  document.querySelector("#whiteModelSetup").value = state.players.white.type;
  renderSetupConfig("black");
  renderSetupConfig("white");
}

function openSetup() {
  renderSetup();
  setupOverlay.classList.remove("hidden");
}

function closeSetup() {
  setupOverlay.classList.add("hidden");
}

function buildBoard() {
  boardEl.innerHTML = "";
  squareElements = Array.from({ length: SIZE }, function () { return Array(SIZE); });
  for (let row = 0; row < SIZE; row += 1) {
    for (let col = 0; col < SIZE; col += 1) {
      const action = row * SIZE + col;
      const square = document.createElement("button");
      square.className = "square";
      square.type = "button";
      square.setAttribute("role", "gridcell");
      square.setAttribute("aria-label", actionName(action));
      square.addEventListener("click", function () { handleHumanMove(action); });
      boardEl.appendChild(square);
      squareElements[row][col] = square;
    }
  }
}

function createPiece(value, animated) {
  const piece = document.createElement("span");
  piece.className = "piece " + (value === 1 ? "black" : "white") + (animated ? " enter" : "");
  return piece;
}

function createFlipPiece(previousValue, nextValue) {
  const flip = document.createElement("span");
  flip.className = "piece-flip";
  const front = document.createElement("span");
  front.className = "flip-face front " + (previousValue === 1 ? "black" : "white");
  const back = document.createElement("span");
  back.className = "flip-face back " + (nextValue === 1 ? "black" : "white");
  flip.appendChild(front);
  flip.appendChild(back);
  return flip;
}

function updatePiece(square, previousValue, nextValue, initialRender) {
  if (previousValue === nextValue) return;
  const oldPiece = square.querySelector(".piece, .piece-flip");
  if (oldPiece) oldPiece.remove();
  if (nextValue === 0) return;
  if (!initialRender && previousValue !== 0 && previousValue !== nextValue) {
    square.appendChild(createFlipPiece(previousValue, nextValue));
  } else {
    square.appendChild(createPiece(nextValue, !initialRender));
  }
}

function renderBoard() {
  const initialRender = squareElements.length === 0;
  if (initialRender) buildBoard();
  const legal = state.gameOver || !state.gameStarted ? [] : legalActions(state.board, state.currentPlayer);
  for (let row = 0; row < SIZE; row += 1) {
    for (let col = 0; col < SIZE; col += 1) {
      const action = row * SIZE + col;
      const square = squareElements[row][col];
      const previousValue = renderedBoard ? renderedBoard[row][col] : 0;
      updatePiece(square, previousValue, state.board[row][col], initialRender);
      square.classList.toggle("legal", legal.includes(action) && action !== PASS);
      square.classList.toggle("last-move", state.lastAction === action);
    }
  }
  renderedBoard = state.board.map(function (line) { return line.slice(); });
}

function renderStats() {
  const counts = countPieces(state.board);
  const occupied = counts.black + counts.white;
  const legal = state.gameOver || !state.gameStarted ? [] : legalActions(state.board, state.currentPlayer);
  document.querySelector("#blackCount").textContent = counts.black;
  document.querySelector("#whiteCount").textContent = counts.white;
  document.querySelector("#emptyCount").textContent = SIZE * SIZE - occupied;
  document.querySelector("#leadMetric").textContent = Math.abs(counts.black - counts.white);
  document.querySelector("#turnMetric").textContent = String(Math.max(1, occupied - 3)).padStart(2, "0");
  document.querySelector("#progressText").textContent = occupied + " / 64";
  document.querySelector("#progressBar").style.width = Math.max(6, (occupied / 64) * 100) + "%";
  document.querySelector("#moveCounter").textContent = state.gameOver ? "对局已结束" : !state.gameStarted ? "等待开始 · 请先确认配置" : "第 " + Math.max(1, occupied - 3) + " 手 · " + (legal.length === 1 && legal[0] === PASS ? 0 : legal.length) + " 个合法位置";
  document.querySelector("#turnLabel").textContent = state.gameOver ? "对局结束" : !state.gameStarted ? "等待开始" : (state.currentPlayer === 1 ? "黑方回合" : "白方回合");
  document.querySelector("#boardStatus").textContent = state.gameOver ? "已结束" : !state.gameStarted ? "未开始" : (isHumanTurn() ? "等待落子" : "AI 思考中");
  document.querySelector("#blackPlayerChip").classList.toggle("active-player", state.gameStarted && !state.gameOver && state.currentPlayer === 1);
  document.querySelector("#whitePlayerChip").classList.toggle("active-player", state.gameStarted && !state.gameOver && state.currentPlayer === -1);
  document.querySelector("#startButton span").textContent = state.gameStarted && !state.gameOver ? "结束对局" : "开始对局";
  document.querySelector("#autoPlayToggle").checked = state.autoPlay;
  document.querySelector("#stepButton").disabled = !state.gameStarted || state.gameOver || state.autoPlay || state.paused || isHumanTurn();
}

function renderMode() {
  state.mode = deriveMode();
  document.querySelector("#matchTitle").textContent = "当前对局";
  document.querySelector("#blackName").textContent = modelName(1);
  document.querySelector("#whiteName").textContent = modelName(-1);
  const summary = document.querySelector("#selectedModelsSummary");
  if (summary) {
    summary.innerHTML = '<div class="selected-model-row"><b>黑方 · ' + modelName(1) + '</b><span>' + modelSummary(state.players.black) + '</span></div><div class="selected-model-row"><b>白方 · ' + modelName(-1) + '</b><span>' + modelSummary(state.players.white) + '</span></div>';
  }
}

function renderEvents() {
  eventListEl.innerHTML = state.events.length ? state.events.map(function (event) {
    return "<div class=\"event\"><time>" + event.time + "</time><span><b>" + event.text + "</b></span><span class=\"event-tag\">" + event.tag + "</span></div>";
  }).join("") : "<div class=\"event\"><time>--:--</time><span><b>等待第一步</b></span><span class=\"event-tag\">READY</span></div>";
}

function render() {
  renderBoard();
  renderStats();
  renderMode();
  renderEvents();
  const pauseButton = document.querySelector("#pauseButton");
  pauseButton.textContent = state.paused ? "继续" : "暂停";
  pauseButton.disabled = !state.gameStarted || state.gameOver;
}

function isHumanTurn() {
  if (!state.gameStarted || state.mode === "replay" || state.gameOver) return false;
  return playerConfig(state.currentPlayer).type === "human";
}

function addEvent(text, tag) {
  const now = new Date();
  const time = String(now.getHours()).padStart(2, "0") + ":" + String(now.getMinutes()).padStart(2, "0");
  state.events.unshift({ time: time, text: text, tag: tag });
  state.events = state.events.slice(0, 8);
}

function clearAiTimer() {
  if (state.aiTimer !== null) {
    window.clearTimeout(state.aiTimer);
    state.aiTimer = null;
  }
}

function scheduleAiTurn() {
  clearAiTimer();
  if (!state.autoPlay || !state.gameStarted || state.gameOver || state.paused || isHumanTurn()) return;
  state.aiTimer = window.setTimeout(function () {
    state.aiTimer = null;
    runAiTurn();
  }, 2000);
}

function finishIfNeeded() {
  if (!isFinished(state.board)) return false;
  const counts = countPieces(state.board);
  state.gameOver = true;
  clearAiTimer();
  const winner = counts.black === counts.white ? "和棋" : counts.black > counts.white ? "黑方获胜" : "白方获胜";
  addEvent(winner + " · " + counts.black + " : " + counts.white, "FINAL");
  return true;
}

function playAction(action, source) {
  if (!state.gameStarted) return;
  const legal = legalActions(state.board, state.currentPlayer);
  if (!legal.includes(action)) return;
  const player = state.currentPlayer;
  if (action === PASS) {
    addEvent((player === 1 ? "黑方" : "白方") + " 无合法落点，跳过", "PASS");
  } else {
    const applied = applyAction(state.board, player, action);
    state.board = applied.board;
    state.lastAction = action;
    addEvent((player === 1 ? "黑方" : "白方") + " 落子 " + actionName(action) + " · " + source, source === "你" ? "HUMAN" : "AI");
  }
  state.currentPlayer = -player;
  if (finishIfNeeded()) render();
  render();
  if (!state.gameOver && !isHumanTurn() && !state.paused && state.autoPlay) scheduleAiTurn();
}

function handleHumanMove(action) {
  if (!isHumanTurn() || state.paused) return;
  playAction(action, "你");
}

function runAiTurn() {
  if (!state.gameStarted || state.gameOver || state.paused || isHumanTurn()) return;
  const action = chooseAiAction(state.currentPlayer);
  playAction(action, modelName(state.currentPlayer));
}

function endGame() {
  clearAiTimer();
  state.gameStarted = false;
  state.gameOver = true;
  state.paused = true;
  addEvent("对局已结束", "END");
  render();
}

function resetGame() {
  clearAiTimer();
  state.mode = deriveMode();
  state.gameStarted = true;
  state.board = createInitialBoard();
  state.currentPlayer = 1;
  state.gameOver = false;
  state.lastAction = null;
  state.paused = false;
  state.events = [];
  addEvent(state.mode === "ai" ? "AI 对战已准备" : "对局已准备", "READY");
  render();
  if (!isHumanTurn() && !state.paused && state.autoPlay) scheduleAiTurn();
}

document.querySelectorAll(".mode-button").forEach(function (button) {
  button.addEventListener("click", function () {
    if (button.dataset.mode === "replay") {
      addEvent("对局记录将在 API 完成后启用", "TODO");
      renderEvents();
      document.querySelectorAll(".mode-button").forEach(function (item) { item.classList.toggle("active", item.dataset.mode === "match"); item.setAttribute("aria-selected", item.dataset.mode === "match" ? "true" : "false"); });
      return;
    }
    document.querySelectorAll(".mode-button").forEach(function (item) { item.classList.toggle("active", item === button); item.setAttribute("aria-selected", item === button ? "true" : "false"); });
  });
});

document.querySelector("#blackModelSetup").addEventListener("change", function (event) { state.players.black.type = event.target.value; renderSetupConfig("black"); });
document.querySelector("#whiteModelSetup").addEventListener("change", function (event) { state.players.white.type = event.target.value; renderSetupConfig("white"); });
document.querySelector("#confirmSetup").addEventListener("click", function () { closeSetup(); addEvent("对局配置已保存", "SETUP"); render(); });
document.querySelector("#openSetupButton").addEventListener("click", openSetup);
document.querySelector("#startButton").addEventListener("click", function () { if (state.gameStarted && !state.gameOver) endGame(); else resetGame(); });
document.querySelector("#pauseButton").addEventListener("click", function () { if (!state.gameStarted || state.gameOver) return; state.paused = !state.paused; if (state.paused) clearAiTimer(); render(); if (!state.paused && !isHumanTurn() && state.autoPlay) scheduleAiTurn(); });
document.querySelector("#autoPlayToggle").addEventListener("change", function (event) { state.autoPlay = event.target.checked; clearAiTimer(); render(); if (state.autoPlay && !isHumanTurn()) scheduleAiTurn(); });
document.querySelector("#stepButton").addEventListener("click", function () { if (state.autoPlay || state.paused || state.gameOver || !state.gameStarted || isHumanTurn()) return; runAiTurn(); });
document.querySelector("#clearLog").addEventListener("click", function () { state.events = []; renderEvents(); });
document.querySelector("#themeButton").addEventListener("click", function () { document.body.classList.toggle("light"); });

addEvent("等待配置双方模型", "SETUP");
render();
renderSetup();

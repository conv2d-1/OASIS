# OASIS 前端

当前版本是一个无需构建工具的静态演示页面。

## 本地运行

在仓库根目录执行：

~~~powershell
python -m http.server 4173 --bind 127.0.0.1 --directory frontend
~~~

然后打开 `http://127.0.0.1:4173/`，并保持运行服务的终端窗口打开。

也可以直接双击 `start.bat`，再打开同一个地址。

如果页面仍无法打开，可以直接双击 `index.html` 作为本地演示页面。当前页面不依赖后端接口。

当前页面在浏览器中运行黑白棋规则和演示 AI，方便先完成棋盘交互、模式切换和回放界面。棋盘采用增量 DOM 更新，落子和翻子只更新受影响的格子，翻子使用 3D 翻转动画。`api/` 完成后，将 `app.js` 中的本地控制器替换为对局服务请求。

## 连接真实 API

启动后端：

~~~powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
~~~

再启动前端静态服务：

~~~powershell
python -m http.server 4173 --bind 127.0.0.1 --directory frontend
~~~

前端会通过 API 创建对局、提交真人动作和请求 Agent 下一步。若没有 checkpoint，AlphaZero 会在 Agent 列表中标记为不可用，RGSC 在当前版本也会返回未实现状态。

播放控制使用 0、0.5、1、1.5、2 秒五档间隔。Agent 会立即开始推理，实际落子等待时间为推理耗时和所选间隔中的较大值。事件记录会显示每次 Agent 的推理时长。

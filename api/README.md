# OASIS API

## 运行

在仓库根目录执行：

~~~powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
~~~

前端默认请求 `http://127.0.0.1:8000`。如需修改地址，可以在页面加载前设置 `window.OASIS_API_BASE`。

## 接口

- `GET /api/health` 检查服务状态
- `GET /api/agents` 返回可用 Agent 和 checkpoint
- `POST /api/games` 创建对局
- `GET /api/games/{id}` 获取局面
- `POST /api/games/{id}/moves` 提交真人落子
- `POST /api/games/{id}/step` 执行当前 Agent 一步
- `POST /api/games/{id}/end` 结束对局

当前对局保存在内存中。服务重启后，对局会丢失。模型 checkpoint 从 `results/` 和 `models/` 自动发现。

# 模型契约与异步视频任务治理设计

## 目标

修复图片 `size`/`quality` 参数解析回归，将 Gemini `predictLongRunning` 改造成真正的本地异步 operation，统一八个公开模型与 OpenAI 兼容接口契约，并同步测试及 README。

## 范围

本次改造包含四部分：

1. 图片模型参数优先级。
2. 视频异步提交、持久化、轮询和失败语义。
3. 公开模型与协议入口契约。
4. 测试、依赖说明和用户文档。

不包含：自动恢复服务重启前尚未提交到 Flow 的图片输入；新增视频模型；删除旧模型兼容别名；删除 `/v1/chat/completions`。

## 图片参数契约

输出分辨率优先级固定为：

```text
显式 imageSize / image_size / resolution
    > quality
    > size 中明确的 1k / 2k / 4k 档位
```

像素尺寸值（如 `1024x1024`、`1024x1792`）只用于推导画幅，不能覆盖 `quality`。因此：

- `size=1024x1024, quality=high` 解析为 `square + 4k`。
- `size=1024x1792, quality=medium` 解析为 `portrait + 2k`。
- `imageSize=2k, quality=high` 保持 `2k`。
- `size=4k` 可直接选择 `4k`。

该规则应用于 `imageConfig`、`generationConfig`、`extra_body`、OpenAI 顶层字段和查询参数。

## 异步视频 operation 契约

### 提交流程

`predictLongRunning` 完成同步输入校验后执行以下步骤：

1. 快速选择一个可用视频账号并登记 pending 占用。
2. 生成本地 operation UUID。
3. 写入 `tasks` 表，状态为 `submitting`。
4. 创建后台提交任务。
5. 立即返回 `200`：

```json
{"name":"operations/<local-id>","done":false}
```

后台任务负责校验 Token、准备项目、上传参考图、向 Flow 提交视频，并更新本地任务。

### 数据模型

继续复用 `tasks` 表：

- `task_id`：本地 operation ID，对客户端稳定。
- `upstream_operation_id`：新增可空列，保存 Flow operation ID。
- `error_code`：新增可空列，保存失败的 HTTP/Gemini 语义码。
- `token_id`：创建本地 operation 前已经完成账号选择，继续保持非空。
- `status`：支持 `submitting`、`processing`、`completed`、`failed`。

数据库迁移只新增可空列，不删除或重命名现有列。

### 后台状态转换

```text
submitting
  ├─ Flow 接受请求 → processing
  └─ 提交失败      → failed

processing
  ├─ Flow 完成     → completed
  └─ Flow 失败     → failed
```

成功提交后，`upstream_operation_id`、`project_id`、`media_name`、`scene_id` 和请求日志信息写入任务。pending 账号占用无论成功或失败都必须在 `finally` 中释放。

### 轮询语义

- `submitting`：HTTP 200，`done:false`。
- `processing`：轮询 `upstream_operation_id`，HTTP 200，未完成时 `done:false`。
- `completed`：HTTP 200，`done:true`，返回 `generateVideoResponse`。
- `failed`：HTTP 200，`done:true`，返回 `error.code/status/message`。
- operation 不存在：HTTP 404。
- 同步阶段无可用账号：HTTP 503，不创建 operation。

错误映射：

- 模型权限拒绝：403 / `PERMISSION_DENIED`。
- 暂无可用账号、Token 无效或账号层级不支持：503 / `UNAVAILABLE`，其中明确的模型层级拒绝可映射为 403。
- 上游请求或协议失败：502 / `UNAVAILABLE`。
- 未分类后台错误：500 / `INTERNAL`。

### 服务重启

- `processing` 且存在 `upstream_operation_id` 的任务继续正常轮询。
- `submitting` 且没有 `upstream_operation_id` 的任务在首次轮询时标记为失败，消息说明提交被服务重启中断。
- 不自动重提，避免重复生成和重复扣点。

### 后台任务生命周期

`GenerationHandler` 维护本进程的提交任务集合，并通过完成回调清理引用。客户端断开不取消已经返回 operation 的任务，因为客户端可继续轮询。

## Keep-alive 边界

`predictLongRunning` 不再使用 whitespace keep-alive，因为本地 operation 会立即返回。

同步 `generateContent` 仍保留 JSON whitespace keep-alive，以兼容现有网关。辅助函数改为明确的同步生成命名，并在响应生成器结束或取消时清理本地等待任务。视频客户端文档优先推荐 `predictLongRunning`。

## 公开 API 契约

三个发现入口返回相同的八个规范模型 ID：

1. `Nano Banana Pro`
2. `Nano Banana 2.1`
3. `Nano Banana 2 Lite`
4. `Imagen 4`
5. `Omni 1.1 Flash`
6. `Veo 3.1 - Lite`
7. `Veo 3.1 - Fast`
8. `Veo 3.1 - Quality`

`Nano Banana 2`、`Nano Banana2` 及旧长模型 ID 保持生成兼容，但不进入公开发现列表。

保留 `/v1/chat/completions` 作为 OpenAI 兼容生成入口。README 不再声明项目只有 Gemini 生成协议。

## 测试策略

采用红—绿—重构：

- 为像素 `size` 与 `quality` 优先级增加失败测试。
- 为本地 operation 的立即返回、后台成功、后台权限失败、后台上游失败增加测试。
- 为旧 SQLite 任务表迁移新增列增加测试。
- 为重启中断的 `submitting` operation 增加测试。
- 更新公开模型集合为八个，避免散落硬编码数量。
- 将 `/v1/chat/completions` 测试改为存在性和鉴权/校验测试。
- 修复验证码测试绕过构造函数和扩展同步测试 patch 路径的问题。
- 最终运行完整 `unittest` 套件、`compileall` 和 `pip check`。

## 回滚

- 数据库仅增加可空列，旧数据继续可读。
- 保留现有 Flow 提交和轮询核心代码，将其拆为本地 operation 编排与上游提交两个阶段。
- 如需回滚路由，可恢复同步调用路径；新增列无需删除。
- 旧模型兼容别名不变，因此调用方无需回滚模型名称。

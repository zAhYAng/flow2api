# 模型契约与异步视频治理实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 subagent-driven-development（推荐）或 executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 修复图片参数解析回归，将 `predictLongRunning` 改为持久化本地 operation，统一八个公开模型、OpenAI 入口、测试和 README。

**架构：** 图片解析器将像素 `size` 与输出档位分离。视频提交拆为“快速创建本地任务”和“后台提交 Flow”两阶段，`tasks.task_id` 作为公开 operation ID，新增字段保存上游 operation 和错误码。路由立即返回 operation，轮询负责暴露后台提交及生成结果。

**技术栈：** Python 3.10、FastAPI、Pydantic、aiosqlite、unittest、SQLite。

---

## 文件职责

- `src/core/model_resolver.py`：统一图片画幅和输出分辨率参数优先级。
- `src/core/models.py`：扩展持久化任务模型。
- `src/core/database.py`：迁移和读写异步任务新增字段。
- `src/services/generation_handler.py`：本地 operation 创建、后台 Flow 提交、轮询状态转换。
- `src/api/routes.py`：`predictLongRunning` 路由契约和同步生成 keep-alive 边界。
- `tests/test_model_resolver_2k.py`：图片参数组合回归测试。
- `tests/test_gemini_video_operations.py`：异步 operation API、持久化、失败和重启测试。
- `tests/test_gemini_only_routes.py`：八个公开模型和 OpenAI 路由契约。
- `tests/test_api_captcha_fingerprint.py`：使用完整初始化的 FlowClient 测试夹具。
- `tests/test_extension_account_sync_guard.py`：修复配置属性 patch。
- `README.md`：同步八个公开模型、OpenAI/Gemini 接口和异步视频推荐流程。

### 任务 1：修复图片 size/quality 参数优先级

**文件：**
- 修改：`src/core/model_resolver.py:613-930`
- 测试：`tests/test_model_resolver_2k.py`
- 测试：`tests/test_veo_lite_support.py:39-52`

- [ ] **步骤 1：增加像素尺寸和显式档位的失败测试**

```python
def test_openai_pixel_size_does_not_override_quality(self):
    request = SimpleNamespace(__pydantic_extra__={"size": "1024x1024", "quality": "high"})
    self.assertEqual(
        resolve_model_name("nano-banana-2", request, MODEL_CONFIG),
        "gemini-3.1-flash-image-square-4k",
    )

def test_explicit_image_size_wins_over_quality(self):
    request = SimpleNamespace(
        generationConfig={"imageSize": "2k", "size": "1024x1024", "quality": "high"}
    )
    self.assertEqual(
        resolve_model_name("Nano Banana 2.1", request, MODEL_CONFIG),
        "gemini-3.1-flash-image-square-2k",
    )
```

- [ ] **步骤 2：运行目标测试确认失败**

运行：`.venv\Scripts\python.exe -m unittest tests.test_model_resolver_2k tests.test_veo_lite_support.VeoLiteModelResolverTests -v`

预期：像素 `size + quality` 用例失败，实际解析为 `1k`。

- [ ] **步骤 3：实现档位 size 识别与统一优先级**

新增只接受 `1k/2k/4k/1080p` 档位的辅助函数；像素尺寸只调用 `_aspect_from_openai_size`。在五个参数入口统一使用：显式分辨率、quality、档位 size。

- [ ] **步骤 4：运行图片解析测试确认通过**

运行：`.venv\Scripts\python.exe -m unittest tests.test_model_resolver_2k tests.test_veo_lite_support.VeoLiteModelResolverTests -v`

预期：全部通过。

- [ ] **步骤 5：提交**

```bash
git add src/core/model_resolver.py tests/test_model_resolver_2k.py tests/test_veo_lite_support.py
git commit -m "fix(model): 修复 size 与 quality 的分辨率优先级"
```

### 任务 2：扩展本地 operation 持久化模型

**文件：**
- 修改：`src/core/models.py:94-111`
- 修改：`src/core/database.py:793-813,1301-1344`
- 测试：`tests/test_gemini_video_operations.py`

- [ ] **步骤 1：增加旧任务表迁移失败测试**

创建缺少新字段的旧版 `tasks` 表，执行 `init_db()` 后断言：

```python
columns = {row[1] for row in connection.execute("PRAGMA table_info(tasks)")}
self.assertIn("upstream_operation_id", columns)
self.assertIn("error_code", columns)
```

- [ ] **步骤 2：运行迁移测试确认失败**

运行：`.venv\Scripts\python.exe -m unittest tests.test_gemini_video_operations.GeminiVideoPersistenceTests.test_old_database_task_schema_migrates -v`

预期：缺少新增列。

- [ ] **步骤 3：扩展 Task 和数据库迁移**

`Task` 新增：

```python
upstream_operation_id: Optional[str] = None
error_code: Optional[int] = None
```

`init_db()` 对已有表执行幂等 `ALTER TABLE`。`create_task()` 和 `update_task()` 支持新字段。

- [ ] **步骤 4：运行数据库相关测试确认通过**

运行：`.venv\Scripts\python.exe -m unittest tests.test_gemini_video_operations.GeminiVideoPersistenceTests -v`

- [ ] **步骤 5：提交**

```bash
git add src/core/models.py src/core/database.py tests/test_gemini_video_operations.py
git commit -m "feat(video): 扩展本地 operation 持久化字段"
```

### 任务 3：实现本地 operation 与后台 Flow 提交

**文件：**
- 修改：`src/services/generation_handler.py:1183-1465`
- 测试：`tests/test_gemini_video_operations.py`

- [ ] **步骤 1：编写后台生命周期失败测试**

覆盖：

```python
name = await handler.enqueue_gemini_video(model=MODEL, prompt="cat", images=[])
self.assertTrue(name.startswith("operations/"))
self.assertEqual((await handler.get_gemini_video_operation(name))["done"], False)
await handler.wait_for_gemini_video_submission(name)
self.assertEqual((await db.get_task(local_id)).upstream_operation_id, "op-123")
```

另写权限拒绝和普通上游失败测试，等待后台提交后轮询得到 `done:true` 与对应 `error.code`。

- [ ] **步骤 2：运行生命周期测试确认失败**

运行：`.venv\Scripts\python.exe -m unittest tests.test_gemini_video_operations.GeminiVideoPersistenceTests -v`

预期：`enqueue_gemini_video` 尚不存在。

- [ ] **步骤 3：拆分提交实现**

实现：

```python
async def enqueue_gemini_video(...) -> str
async def _submit_gemini_video_operation(local_operation_id, token, ...)
async def wait_for_gemini_video_submission(name: str) -> None
```

`enqueue` 选择账号、写 `submitting` 任务并创建后台 task；后台函数复用现有上传和 Flow 提交逻辑，成功后更新为 `processing`，失败后写 `failed/error_code/error_message`，最后释放 pending。

- [ ] **步骤 4：调整轮询使用本地 ID 和上游 ID**

`get_gemini_video_operation()`：

- `submitting` 且后台任务仍存在：`done:false`。
- `submitting` 且任务不存在：标记重启中断失败。
- `processing`：使用 `upstream_operation_id` 调 Flow。
- `failed`：返回持久化的错误码和 Gemini status。

- [ ] **步骤 5：运行持久化和生命周期测试确认通过**

运行：`.venv\Scripts\python.exe -m unittest tests.test_gemini_video_operations.GeminiVideoPersistenceTests -v`

- [ ] **步骤 6：提交**

```bash
git add src/services/generation_handler.py tests/test_gemini_video_operations.py
git commit -m "feat(video): 使用本地 operation 后台提交 Flow"
```

### 任务 4：切换 predictLongRunning 路由并收敛 keep-alive

**文件：**
- 修改：`src/api/routes.py:1035-1232`
- 测试：`tests/test_gemini_video_operations.py`

- [ ] **步骤 1：更新 API 失败测试为异步轮询契约**

提交响应必须立即为：

```json
{"name":"operations/local-123","done":false}
```

后台失败由 operation 轮询返回：

```json
{"name":"operations/local-123","done":true,"error":{"code":403,"status":"PERMISSION_DENIED"}}
```

- [ ] **步骤 2：运行 API 测试确认旧实现失败**

运行：`.venv\Scripts\python.exe -m unittest tests.test_gemini_video_operations.GeminiVideoApiTests -v`

- [ ] **步骤 3：路由改用 enqueue 并移除异步提交 keep-alive**

`predict_video_long_running()` 调用 `enqueue_gemini_video()` 并直接返回 operation 字典。同步校验和无可用账号继续返回真实 HTTP 4xx/5xx。

- [ ] **步骤 4：限制 keep-alive 只服务 generateContent**

将 `_whitespace_keep_alive` 重命名为 `_stream_json_with_keep_alive`，加入 `finally` 清理未完成 task。只有 `generateContent` 使用该函数。

- [ ] **步骤 5：运行 API 测试确认通过**

运行：`.venv\Scripts\python.exe -m unittest tests.test_gemini_video_operations.GeminiVideoApiTests -v`

- [ ] **步骤 6：提交**

```bash
git add src/api/routes.py tests/test_gemini_video_operations.py
git commit -m "refactor(api): 将视频提交切换为本地异步 operation"
```

### 任务 5：统一公开模型与 OpenAI 路由契约

**文件：**
- 修改：`tests/test_gemini_only_routes.py`
- 修改：`README.md`

- [ ] **步骤 1：更新公开模型集合测试**

```python
public_models = {
    "Nano Banana Pro", "Nano Banana 2.1", "Nano Banana 2 Lite", "Imagen 4",
    "Omni 1.1 Flash", "Veo 3.1 - Lite", "Veo 3.1 - Fast", "Veo 3.1 - Quality",
}
```

数量断言统一使用 `len(self.public_models)`。

- [ ] **步骤 2：更新 OpenAI 路由测试**

从 404 列表移除 `/v1/chat/completions`，新增空请求返回 422 或缺少 prompt 返回 400 的存在性断言。

- [ ] **步骤 3：运行模型发现测试确认通过**

运行：`.venv\Scripts\python.exe -m unittest tests.test_gemini_only_routes -v`

- [ ] **步骤 4：同步 README**

更新为八个模型；主示例使用 `Nano Banana 2.1`；注明旧名兼容；说明 `/v1/chat/completions` 保留；视频优先推荐 `predictLongRunning`。

- [ ] **步骤 5：提交**

```bash
git add tests/test_gemini_only_routes.py README.md
git commit -m "docs(api): 统一公开模型与协议入口契约"
```

### 任务 6：修复失效测试夹具并完成全量验证

**文件：**
- 修改：`tests/test_api_captcha_fingerprint.py`
- 修改：`tests/test_extension_account_sync_guard.py`

- [ ] **步骤 1：修复 FlowClient 初始化测试夹具**

使用正常构造函数创建 `FlowClient`，并断言 fake session 被调用两次。

- [ ] **步骤 2：修复 Config 属性 patch**

```python
patch.object(
    Config,
    "extension_account_sync_enabled",
    new_callable=PropertyMock,
    return_value=False,
)
```

- [ ] **步骤 3：运行两个目标测试**

运行：`.venv\Scripts\python.exe -m unittest tests.test_api_captcha_fingerprint tests.test_extension_account_sync_guard -v`

- [ ] **步骤 4：运行完整验证**

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m compileall -q src tests
.venv\Scripts\python.exe -m pip check
git diff --check
```

预期：所有测试通过、编译通过、依赖无冲突、无空白错误。

- [ ] **步骤 5：提交**

```bash
git add tests/test_api_captcha_fingerprint.py tests/test_extension_account_sync_guard.py
git commit -m "test: 修复验证码与扩展同步测试夹具"
```

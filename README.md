# Flow2API Fork

将 Google Flow 的图片和视频生成能力封装为 Gemini 请求体兼容 API，并通过仓库自带的 Chrome 扩展同步当前浏览器账号、刷新 ST/AT 和处理 reCAPTCHA。

本仓库是 [zAhYAng/flow2api](https://github.com/zAhYAng/flow2api) 的维护版本，保留上游核心能力，并针对原生运行、浏览器插件同步、公开模型名和管理后台做了调整。

## 主要变化

- 内置 Chrome 扩展同时负责账号导入、定时同步和 reCAPTCHA，不需要另外安装 Token Updater。
- 推荐原生 Python 运行，方便直接使用真实 Chrome 登录态。
- 对外只展示 8 个简洁规范模型名，历史旧模型名与底层长模型 ID 仍保持调用兼容。
- 支持 Gemini `generateContent` / `streamGenerateContent` 与视频异步任务 `predictLongRunning`。
- 保留 `POST /v1/chat/completions` 与 `GET /v1/models`，保持 OpenAI 客户端与外部 Agent 接入兼容。
- 管理后台显示账号积分；图片不扣点，视频按 Flow 规则统计点数。
- 图片生成失败可自动切换其他账号重试，默认兜底 1 次，可在后台关闭或调整。
- 支持企业微信 Webhook 通知：账号失效实时告警、定时每日汇报生成情况（图片/视频成败统计、各账号调用排行与余额）。
- 移除了上游 README 中与本 fork 使用方式无关的推广内容。

## 支持能力

- 文生图、图生图、连续图片编辑
- 文生视频、图生视频、首尾帧视频、多参考图视频
- Nano Banana Pro / Nano Banana 2.1 的 2K、4K 输出，以及轻量版 Nano Banana 2 Lite
- Omni 1.1 Flash 的 4、6、8、10 秒路由，支持首帧和首尾帧
- 多账号 Token 管理、并发控制和负载均衡
- 浏览器账号自动导入与定时刷新
- Chrome 扩展 route key 绑定，多浏览器 Profile 对应多账号
- Web 管理后台、请求日志、余额和生成统计
- 企业微信 Webhook 通知与每日定时统计汇报

## 快速开始

### 1. 原生运行

要求 Python 3.10+，推荐使用虚拟环境：

```bash
git clone https://github.com/zAhYAng/flow2api.git
cd flow2api

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Windows 激活虚拟环境：

```powershell
.venv\Scripts\activate
```

服务默认监听：

```text
http://127.0.0.1:8000
```

首次登录管理后台：

```text
用户名：admin
密码：admin
```

首次登录后请立即修改密码和 API Key。

### 2. 加载浏览器扩展

1. 在 Chrome 打开 `chrome://extensions/`。
2. 开启“开发者模式”。
3. 点击“加载已解压的扩展程序”。
4. 选择仓库里的 `extension` 目录。
5. 打开扩展设置页并填写：

```text
WebSocket URL: ws://127.0.0.1:8000/captcha_ws
Flow2API API Key: 管理后台中的 API Key
```

6. 在同一 Chrome Profile 登录 `https://labs.google/fx/tools/flow`。
7. 点击“导入当前 Google 账号”。
8. 开启“定时自动导入当前 Google 账号”。

扩展会读取当前 Profile 的 Labs Session Token 和 Google 登录 Cookie，导入后台并定时更新。后台验证码方式应设置为 `extension`。

扩展版本显示在插件设置页标题和 Chrome 扩展详情页中。每次插件代码更新都会递增 `extension/manifest.json` 的版本号，并同步记录在 `extension/CHANGELOG.md`。

详细说明见 [浏览器插件配置](docs/captcha-worker-setup.md)。

### 3. 多账号

不同账号必须使用不同 Chrome Profile，不能只开同一 Profile 的多个窗口。

每个 Profile 单独加载扩展。插件会自动生成不同的内部实例标识：

```text
账号 A: 自动生成
账号 B: 自动生成
账号 C: 自动生成
```

导入后，后台 Token 会自动绑定当前浏览器 Profile，生成时验证码请求只会发给匹配的 Profile。API Key 是后台唯一的全局鉴权 Key，不需要为每个插件生成新的 Key。

## API 接入

Flow2API 提供 **OpenAI 兼容** 与 **Gemini 原生** 两类协议入口，满足不同客户端与 Agent 框架的集成需求。所有接口均支持 API Key 认证。

支持的认证方式：

```text
Authorization: Bearer <api_key>
x-goog-api-key: <api_key>
?key=<api_key>
```

### 1. 模型发现

- **OpenAI 格式发现**：`GET /v1/models`，返回标准 `{"object": "list", "data": [...]}` 结构，列出 8 个公开规范模型。
- **Gemini 格式发现**：`GET /models` 或 `GET /v1beta/models`，返回标准 `{"models": [...]}` 结构，列出 8 个公开规范模型并携带支持的能力标注（如视频模型声明 `predictLongRunning`）。

### 2. Gemini 原生协议接入

#### 图片生成主示例（Nano Banana 2.1）

```bash
curl -X POST "http://127.0.0.1:8000/models/Nano%20Banana%202.1:generateContent?key=$FLOW2API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [{"text": "一个透明玻璃苹果，白底产品摄影"}]
      }
    ],
    "generationConfig": {
      "imageConfig": {
        "aspectRatio": "1:1",
        "imageSize": "2k"
      }
    }
  }'
```

#### 视频生成：优先推荐 `predictLongRunning` 异步长轮询

视频生成涉及上游排队与长时间渲染，强烈建议外部客户端优先使用异步长轮询入口 `predictLongRunning`，提交任务后立即获得 operation 标识并进行轮询，避免长时间保持 HTTP 连接因网关超时或网络波动而中断：

```bash
# 提交视频生成任务
curl -X POST "http://127.0.0.1:8000/v1beta/models/Veo%203.1%20-%20Lite:predictLongRunning" \
  -H "x-goog-api-key: $FLOW2API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"prompt":"一只橘猫在窗边看雨"}],"parameters":{"aspectRatio":"16:9"}}'

# 轮询任务状态
curl "http://127.0.0.1:8000/v1beta/operations/OPERATION_ID" \
  -H "x-goog-api-key: $FLOW2API_KEY"
```

提交响应包含 `name: "operations/OPERATION_ID"` 和 `done: false`。轮询返回 HTTP 200 时，仍需检查任务结果：

- `done: false`：仍在提交或生成，稍后继续轮询。
- `done: true` 且有 `response`：生成成功，视频链接位于 `response.generateVideoResponse.generatedSamples[0].video.uri`。
- `done: true` 且有 `error`：任务失败，读取 `error.code`、`error.status` 和 `error.message`，停止轮询。例如上游上传失败会返回：

```json
{
  "name": "operations/OPERATION_ID",
  "done": true,
  "error": {
    "code": 502,
    "status": "UNAVAILABLE",
    "message": "Project-scoped image upload failed via /flow/uploadImage"
  }
}
```

本地任务仍处于 `submitting`、尚未保存上游 operation ID 时，如果服务重启，首次轮询会将任务标记为失败（`done: true`，`error.code: 503`，`error.status: "UNAVAILABLE"`，消息为 `Service restart interrupted submission`）。服务不会自动重新提交，以免重复生成和扣点。已保存上游 operation ID 的 `processing` 任务可在重启后继续轮询。

支持一个 `instances`；单张首帧可放在 `instances[0].image.bytesBase64Encoded` 并设置 `mimeType`。

#### 视频生成：同步阻塞 `generateContent`（备用）

如需同步阻塞等待视频完成，可直接调用 `generateContent`，响应中的 `candidates[0].content.parts[0].fileData.fileUri` 即为生成视频直链：

```bash
curl -X POST "http://127.0.0.1:8000/models/Veo%203.1%20-%20Fast:generateContent" \
  -H "x-goog-api-key: $FLOW2API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"role":"user","parts":[{"text":"一只橘猫在窗边看雨"}]}],"generationConfig":{"aspectRatio":"16:9"}}'
```

视频续写可使用 `veo-extend` 模型，并在 `contents[].parts[]` 中传入 `{ "fileData": { "mimeType": "video/mp4", "fileUri": "extend://原视频mediaGenerationId" } }`。

### 3. OpenAI 兼容协议接入

保留 `POST /v1/chat/completions`，允许各类标准 OpenAI 客户端、聊天应用和 Agent 框架直接调用：

```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Authorization: Bearer $FLOW2API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Nano Banana 2.1",
    "messages": [
      {
        "role": "user",
        "content": "一个透明玻璃苹果，白底产品摄影"
      }
    ]
  }'
```

OpenAI 请求返回包含图片或视频链接的标准 `choices[0].message.content` Markdown 输出，支持流式 `stream: true` 与非流式请求。

## 公开模型名

公开模型发现（`/v1/models`、`/models`、`/v1beta/models`）返回以下 8 个规范模型名：

| 模型 | 类型 | 主要参数 |
| --- | --- | --- |
| `Nano Banana Pro` | 图片 | 5 种比例，默认/2K/4K |
| `Nano Banana 2.1` | 图片 | 5 种比例，默认/2K/4K |
| `Nano Banana 2 Lite` | 图片 | 5 种比例，默认尺寸 |
| `Imagen 4` | 图片 | 16:9、9:16 |
| `Omni 1.1 Flash` | 视频 | 4/6/8/10 秒；1 张首帧、2 张首尾帧、3 张以上参考图 |
| `Veo 3.1 - Lite` | 视频 | 按图片数量自动选择 T2V/I2V/首尾帧 |
| `Veo 3.1 - Fast` | 视频 | 按图片数量自动选择 T2V/I2V/R2V |
| `Veo 3.1 - Quality` | 视频 | T2V/I2V，支持 1080p/4K 放大 |

**历史旧名与内部模型兼容说明**：
- 历史旧名称（如 `Nano Banana 2`、`Nano Banana2`、`Nano Banana Lite`）仍保持调用兼容（例如传入 `Nano Banana 2` 会自动映射至 `Nano Banana 2.1` 对应的底层实现），但仅用于请求调用兼容，不再展示在公开模型列表中。
- `Veo 3.1 - Fast`、`Lite` 的公开规范名使用默认时长；若需精确指定 4/6/8 秒，可直接调用内部时长模型名（如 `veo_3_1_t2v_fast_4s`）。完整内部模型键见 [模型路由规则](docs/model-aliases.md)。

## 视频积分

图片生成不消耗账号点数。当前管理后台按以下规则统计成功视频请求：

| 模型 | 模式 / 时长 | 点数 |
| --- | --- | --- |
| `Veo 3.1 - Lite` | 默认 | 10 |
| `Omni 1.1 Flash` | 【帧模式】4 秒 | 7 |
| `Omni 1.1 Flash` | 【帧模式】6 秒 | 10 |
| `Omni 1.1 Flash` | 【帧模式】8 秒 | 12 |
| `Omni 1.1 Flash` | 【帧模式】10 秒 | 15 |
| `Omni 1.1 Flash` | 【素材模式】随视频素材而定 (最高 10 秒) | 20 |

Token 列表里的余额来自上游账号 Credits；“今日视频点数/账号余额”是统计值，不会在本地重复扣款。

## 图片与视频失败兜底

管理后台“系统配置 -> 生成配置”中的“图片失败兜底次数”与“视频失败兜底次数”控制账号切换重试：

```text
0: 关闭兜底
1: 首个账号失败后换另一个账号重试一次（默认）
2-5: 最多切换其他账号重试对应次数
```

兜底适用于图片与视频生成，不会重复使用已经尝试过的账号。提示词安全策略拒绝、参数错误等请求级失败也会按配置尝试其他账号，但不会因此自动禁用原账号。

每次实际切换账号都会单独写入一条“图片兜底”或“视频兜底”请求日志，主日志保留整次请求的最终结果，不会被备用账号尝试覆盖。

## 企业微信 Webhook 通知

支持通过企业微信群机器人接收实时告警与每日统计汇报：

- **通知消息类型可选**：支持 `Markdown 格式`（企业微信客户端排版最佳）与 `纯文本格式`（微信内查看更直观清晰，避免特殊排版符号）。
- **账号失效实时告警**：当账号 Token 过期、刷新失败或触发连续错误禁用时，实时向群机器人发送告警，提示对应账号与失效原因。
- **每日生成情况汇报**：可单独配置开关与每日定时推送时间（默认 22:00）：
  - 今日生成概览（图片/视频成功数、失败数、成功率、视频积分消耗）
  - 账号整体状态（活跃数、异常数、总积分）
  - 各账号今日生成排行榜（按调用量降序列出各个账号生成的图片与视频成败明细、余额）
  - 异常账号警示（列出被禁用或过期的账号及原因）
- **管理后台配置**：“系统配置 -> 企业微信通知配置”中可一键配置 Webhook 地址、消息类型、测试通知、或点击“立即汇报”即刻生成并发送报告。

## 常用地址

```text
管理后台: http://127.0.0.1:8000/manage
模型测试: http://127.0.0.1:8000/test
健康检查: http://127.0.0.1:8000/health
模型列表: http://127.0.0.1:8000/models
Prometheus: http://127.0.0.1:8000/metrics
```

## Docker

仓库仍保留上游 Docker 配置，但浏览器扩展需要连接真实 Chrome 登录态，本 fork 更推荐原生运行。使用 Docker 时，需要确保扩展所在设备可以访问容器的 `8000` 端口，并把 WebSocket URL 改成宿主机可访问地址。

## 排障

### Token 显示过期

- 确认扩展已重新加载且开启定时自动导入。
- 确认同一 Chrome Profile 能正常打开 Google Flow。
- 打开扩展设置查看最近自动导入状态。
- 手动点击“导入当前 Google 账号”，再刷新后台 Token 列表。

### `Failed to obtain reCAPTCHA token`

- 确认扩展 WebSocket 已连接。
- 确认账号是在当前 Chrome Profile 中导入的，不要混用其他 Profile 的账号配置。
- 保持 Google Flow 页面可正常打开。
- 并发请求会排队等待插件生成验证码，先用单请求验证。

### 外部设备调用

外部设备不能使用 `127.0.0.1`，应改成运行 Flow2API 设备的局域网 IP，并确认防火墙已开放 `8000` 端口。

## 文档

- [浏览器插件配置](docs/captcha-worker-setup.md)
- [模型路由规则](docs/model-aliases.md)
- [本项目仓库](https://github.com/zAhYAng/flow2api)
- [上游原作者仓库](https://github.com/TheSmallHanCat/flow2api)

## 许可证

本项目沿用上游 MIT License。请遵守 Google 服务条款，并自行承担账号、网络和 API 使用风险。

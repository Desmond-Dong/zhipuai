<p align="center">
  <img src="https://github.com/user-attachments/assets/f8ff7a6c-4449-496a-889a-d205469a84df" alt="ZhipuAI" width="700" height="400" />
</p>

<h1 align="center">智谱清言 ZhipuAI · Home Assistant 集成</h1>

<p align="center">
  <a href="https://github.com/Desmond-Dong/zhipuai/releases"><img src="https://img.shields.io/github/v/release/knoop7/zhipuai" alt="GitHub Version"></a>
  <a href="https://github.com/Desmond-Dong/zhipuai/issues"><img src="https://img.shields.io/github/issues/knoop7/zhipuai" alt="GitHub Issues"></a>
  <img src="https://img.shields.io/github/forks/knoop7/zhipuai?style=social" alt="GitHub Forks">
  <img src="https://img.shields.io/github/stars/knoop7/zhipuai?style=social" alt="GitHub Stars">
</p>

---

> **智谱清言（zhipuai）** 是 Home Assistant 的自定义集成，实现与智谱大模型的无缝对接，主要包括：
>
> - **对话助手（Conversation）** ｜ 作为 Assist 对话代理，支持流式回答与Home Assistant工具调用。
> - **AI 任务（AI Task）** ｜ 生成结构化数据、生成图片（CogView 系列）。


---

## 🚀 功能特色

### 🤖 对话助手（Conversation）
- **流式输出**：对话内容实时、增量呈现。
- **工具调用**：集成 Home Assistant LLM API，模型可控制设备、执行脚本与自动化、查询状态等。
- **上下文记忆**：可自定义历史消息上限，有效平衡上下文关联与性能。
- **语音多轮**：支持前端/设备唤醒后的连续语音会话。
- **联网搜索**：高级模式下可启用“web_search”参数，提升信息查全率。

### 🛠️ AI 任务（AI Task）
- **结构化数据生成**：支持模型按需输出 JSON 并自动解析，结果可用于自动化/前端。
- **图片生成**：对接智谱图片 API（cogview-3系），输出PNG图片数据。

### ⚙️ 模型与参数
- **聊天模型**：默认 GLM-4-Flash-250414，高级可选多型号（部分付费）。
- **图片模型**：默认 cogview-3-flash，可选 cogview-3、cogview-3-plus。
- **采样与长度**：支持 temperature、top_p、top_k、max_tokens 等调优参数。
- **历史消息上限**：自定义连续对话最大长度，平衡体验和资源。

---

## 📦 安装与升级

### 方式一：HACS 安装 <sup style="color:#b9b;">（推荐）</sup>
1. 进入 HACS 中搜索安装 “zhipuai”。
2. 重启 Home Assistant。

### 方式二：手动安装
- 将 `custom_components/zhipuai` 目录放入配置目录下的 `custom_components/`。
- 重启 Home Assistant。

> ⚠️ **版本兼容：**
> 本集成依赖 Home Assistant 新版 Conversation / AI Task / 子条目等框架能力，建议使用最新版 Home Assistant。

---

## 🛠️ 配置向导（Config Flow）

1. 前往 **设置 → 设备与服务 → 集成 → 添加集成**，搜索“智谱清言（zhipuai）”添加。
2. 输入 [API Key](https://open.bigmodel.cn/usercenter/apikeys)。
3. 首次添加会自动生成两个“子条目（subentry）”：
    - 智谱对话助手（conversation）
    - 智谱AI任务（ai_task_data）
4. 参数修改，请在对应子条目详情页面点击“配置”按钮（父条目不可直接配置）。

> **配置校验：** 系统自动校验证权有效性，API Key 无效或网络异常会有提示。

---

## 📝 使用指南

### A. 对话助手（Assist 对话代理）

- **切换代理**：在 Assist（对话）页面，将当前代理设为“智谱对话助手”。
- **发起对话**：输入任意自然语言指令，如：
  > “打开客厅顶灯，亮度设为60%”
- **工具调用**：若子条目启用 LLM API（默认开启），模型可控制家居、读取状态。
- **历史设置**：通过子条目配置“最大历史消息条数”。
- **联网搜索**：高级模式启用 web_search 字段，发送请求时由服务端判定是否联网。

### B. AI 任务（生成数据）

- “AI 任务”卡片/接口调用会复用对话消息格式，取助手最终回复。
- 若请求 JSON 结构，系统自动解析助手消息为 JSON，若解析失败返回错误信息。

### C. AI 任务（生成图片）

支持`generate_image`服务（`domain: zhipuai`），主要参数如下：

| 参数   | 类型     | 必选 | 说明                                 | 备注                    |
| ------ | -------- | ---- | ------------------------------------ | ----------------------- |
| prompt | string   | ✅   | 描述生成的图片内容                   |                         |
| size   | string   | ❌   | 图片尺寸，默认 `1024x1024`           | 支持：1024x1024, 768x1344, 864x1152, 1344x768, 1152x864, 1440x720, 720x1440 |
| model  | string   | ❌   | 图片模型，默认 `cogview-3-flash`     | 可选 cogview-3、cogview-3-plus |


---

## ⚙️ 参数说明（子条目配置）

- **推荐模式**（默认，适合新手）
    - 对话助手：GLM-4-Flash-250414，temperature 0.3，top_p 0.5，top_k 1，max_tokens 250，历史30，可联网搜索
    - AI 任务：文本GLM-4-Flash-250414（temperature 0.95, top_p 0.7, max_tokens 2000），图片 cogview-3-flash（1024x1024）
- **高级模式**（关闭推荐后，自定义）：
    - 可灵活配置模型、温度、采样相关参数及最大历史条数，web_search 开关等

> 💡 **提示：高温度、长输出、更多历史上下文会显著增加延迟和token消耗，请谨慎权衡。**

---

## ❓ 常见问题（FAQ）

<details>
  <summary>配置时报 401 或“API 密钥无效”？</summary>
  请确认 API Key 正确且未过期，可在控制台重新生成。
</details>

<details>
  <summary>对话没有流式输出？</summary>
  请确保 Assist 已切换到本集成为当前代理，同时确认日志无相关报错。
</details>

<details>
  <summary>图片生成失败？</summary>
  检查 HASS 返回的错误文本，确认模型和尺寸参数受支持，如遇故障稍后重试。
</details>

<details>
  <summary>实体/子条目未出现？</summary>
  请确保 Home Assistant 版本较新，并尝试重启或升级系统。
</details>

---

## 🔒 隐私与安全

- 集成仅在请求阶段使用你配置的 API Key（仅运行时存储，不会默认持久化）。
- 对话内容与图片提示词默认不做本地持久化（自动化记录除外）。
- 对话中上传图片附件会转换为 base64 的 data URL 传递，请注意图片大小和隐私脱敏。

---

## 🤝 参与贡献

欢迎提交 Issue 与 PR，帮助完善文档与集成功能！

- 文档/代码：[https://github.com/Desmond-Dong/zhipuai](https://github.com/Desmond-Dong/zhipuai)
- Issue：[https://github.com/Desmond-Dong/zhipuai/issues](https://github.com/Desmond-Dong/zhipuai/issues)

---

## 📄 许可协议

本项目遵循仓库内 [LICENSE](./LICENSE) 协议发布。

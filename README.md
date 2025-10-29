<h1 align="center">智谱清言 ZhipuAI · Home Assistant 集成</h1>
<p align="center">
  <img src="https://github.com/user-attachments/assets/f8ff7a6c-4449-496a-889a-d205469a84df" alt="ZhipuAI" width="700" height="400" />
</p>


<p align="center">
  <a href="https://github.com/Desmond-Dong/zhipuai/releases"><img src="https://img.shields.io/github/v/release/Desmond-Dong/zhipuai" alt="GitHub Version"></a>
  <a href="https://github.com/Desmond-Dong/zhipuai/issues"><img src="https://img.shields.io/github/issues/Desmond-Dong/zhipuai" alt="GitHub Issues"></a>
  <img src="https://img.shields.io/github/forks/Desmond-Dong/zhipuai?style=social" alt="GitHub Forks">
  <img src="https://img.shields.io/github/stars/Desmond-Dong/zhipuai?style=social" alt="GitHub Stars"> <a href="README_EN.md">English</a>
</p>

---

智谱清言（ZhipuAI）是 Home Assistant 的自定义集成，提供与智谱大模型的原生对接：
- 对话助手（Conversation）：作为 Assist 对话代理，支持流式回答、家居控制工具调用、图片理解。
- AI 任务（AI Task）：生成结构化数据、生成图片（CogView 系列）。
- 语音合成（TTS）：将文本合成为语音（支持流式返回）。
- 语音识别（STT）：将语音转写为文本（支持流式返回）。
- 集成服务：图片分析、图片生成、TTS 播放、STT 转写。
- AI 自动化（实验特性）：用自然语言生成并写入 automations.yaml，自动重载。


## 功能特色

### 对话助手（Conversation）
- 流式输出：实时显示模型回复。
- 控制家居：对接 Home Assistant LLM API，模型可调用“控制/查询”等工具。
- 图片理解：消息携带图片时自动切换到视觉模型（优先使用免费 glm-4v-flash）。
- 上下文记忆：可配置历史消息条数，平衡效果与性能。
- 可选联网搜索：高级模式可开启 web_search 工具。

### AI 任务（AI Task）
- 结构化数据生成：可指定 JSON 结构（失败有错误提示）。
- 图片生成：对接 images/generations，支持 URL 或 base64 返回，统一转为 PNG。
- 附件支持：复用对话消息格式，便于多模态任务。

### 语音合成（TTS）
- 支持多音色（tongtong、xiaochen、chuichui、jam、kazi、douji、luodo）。
- 支持流式/非流式：自动解析分片并合并为 WAV 音频。
- 语速/音量/编码：可配置 speed、volume、response_format（pcm/wav）、encode_format（base64/raw）。

### 语音识别（STT）
- 输入 WAV 优先：自动将音频转换/标准化为 WAV（16k/16bit/单声道兼容）。
- 支持流式/非流式：逐行 data: JSON 拼接识别结果。
- 支持中文/英文等常见语言代码。

### 配置与管理
- 推荐/高级双模式：默认推荐参数即用即走；高级模式开放模型与调参。
- 子条目（Subentry）：在一个集成下，分别管理 Conversation / AI Task / TTS / STT 的参数。
- 版本迁移：自动从旧版本迁移到新结构。


## 安装

### 方式一：HACS（推荐）
1. 在 HACS 中搜索并安装“zhipuai”。
2. 重启 Home Assistant。

### 方式二：手动安装
1. 将仓库中的 `custom_components/zhipuai` 目录复制到 Home Assistant 配置目录下的 `custom_components/`。
2. 重启 Home Assistant。

提示：本集成依赖新版的 Conversation/AI Task/子条目框架，建议使用较新的 Home Assistant 版本。


## 快速开始（配置向导）
1. 进入 设置 → 设备与服务 → 集成 → 添加集成，搜索“智谱清言（zhipuai）”。
2. 按指引输入智谱 API Key（可在 [此处](https://open.bigmodel.cn/usercenter/apikeys) 获取）。
   - 如果还没有智谱的账号，可以先进行[注册](https://www.bigmodel.cn/invite?icode=NWiYEUi2tleEV8cplkb1Z%2BZLO2QH3C0EBTSr%2BArzMw4%3D/)。
3. 成功后会自动创建四个“子条目”：
   - 对话助手 conversation
   - AI 任务 ai_task_data
   - 文本转语音 tts
   - 语音转文本 stt
4. 若需调整参数，在对应子条目点击“配置”进入推荐/高级模式配置。

配置校验：系统会即时校验 API Key 与网络，如失败会提示“密钥无效/无法连接/未知错误”。


## 使用指南

### A. 对话助手（Assist 对话代理）
- 在 Assist 对话页面，将当前代理切换为“智谱对话助手”。
- 直接说/输：“打开客厅灯到 60%”、“帮我总结今天日程”。
- 携带图片：在支持的前端上传或引用图片，模型会自动进行视觉分析并回答。
- 工具调用：在子条目启用 LLM Hass API 后，模型可调用 Home Assistant 工具以控制设备/查询状态。

### B. AI 任务（结构化数据）
- 在卡片或自动化中调用 AI Task 生成数据：
  - 文本→结构化 JSON：任务定义结构后，系统会尝试将回复解析为 JSON。
  - 若解析失败，会返回错误，便于调参或重试。

### C. AI 任务（生成图片）
- 使用服务 zhipuai.generate_image 生成图片（CogView）：
  - 参数：prompt（必填）、size（默认 1024x1024）、model（默认 cogview-3-flash）。
  - 返回：image_url 或 image_base64（内部自动转为 PNG 以便前端/卡片使用）。

### D. 图片分析服务（图像理解）
- 使用服务 zhipuai.analyze_image：
  - 参数：image_file 或 image_entity 二选一、message（分析说明）、model（默认 glm-4v-flash）。
  - 支持 stream（流式）返回增量内容。

### E. 文本转语音（TTS 实体与服务）
- 作为 TTS 实体在前端选择“智谱 TTS”，输入文本播放。
- 或调用服务 zhipuai.tts_speech：
  - 参数：text、voice（默认 tongtong）、speed（0.25~4.0）、volume（0.1~2.0）、response_format、encode_format、stream、media_player_entity（可选直播放）。
  - 非流式：直接返回完整音频；流式：内部合并分块再播放/返回 WAV 数据。

### F. 语音转文本（STT 实体与服务）
- 作为 STT 实体使用：前端/麦克风采集的音频将被标准化为 WAV 后上传到智谱 STT。
- 或调用服务 zhipuai.stt_transcribe：
  - 参数：audio_file（WAV）、model（默认 glm-asr）、temperature、language（默认 zh）、stream。
  - 返回：识别到的文本（流式会持续拼接，直至 [DONE]）。


## 常用服务参数示例（YAML）

1) 生成图片

```yaml
service: zhipuai.generate_image
data:
  prompt: "一只穿宇航服在月球上奔跑的猫"
  size: "1024x1024"
  model: "cogview-3-flash"
```

2) 图片分析

```yaml
service: zhipuai.analyze_image
data:
  image_file: "/config/www/cats/cat.jpg"
  message: "请描述图片的场景和主体"
  model: "glm-4v-flash"
  stream: false
```

3) 文本转语音并在播放器播放

```yaml
service: zhipuai.tts_speech
data:
  text: "欢迎回家，已为你开启客厅灯。"
  voice: "tongtong"
  speed: 1.0
  volume: 1.0
  response_format: "wav"
  encode_format: "base64"
  stream: false
  media_player_entity: media_player.living_room
```

4) 语音转文本

```yaml
service: zhipuai.stt_transcribe
data:
  audio_file: "/config/www/records/command.wav"
  model: "glm-asr"
  temperature: 0.95
  language: "zh"
  stream: true
```


## 参数与模型（默认与推荐）
- 对话（Conversation）默认：
  - 模型：GLM-4-Flash-250414
  - temperature: 0.3，top_p: 0.5，top_k: 1，max_tokens: 250，history: 30
- AI 任务默认：
  - 文本模型：GLM-4-Flash-250414（temperature 0.95 / top_p 0.7 / max_tokens 2000）
  - 图片模型：cogview-3-flash（size 默认 1024x1024）
- 视觉模型优先级：glm-4v-flash（免费优先）、glm-4v、glm-4v-plus
- TTS 默认：模型 cogtts，voice tongtong，response_format pcm，encode_format base64，speed 1.0，volume 1.0，stream true
- STT 默认：模型 glm-asr，temperature 0.95，language zh，stream true

提示：更高温度、更多历史与更长输出会增加耗时与 token 消耗。


## AI 自动化（实验特性）
- 在对话中输入“帮我创建一个自动化…”、“当…就…”，系统会尝试生成标准的 Home Assistant automation YAML：
  - 将 YAML 直接写入 automations.yaml，并尝试自动 reload。
  - 写入前会对 automations.yaml 做时间戳备份。
- 安全提示：该操作会修改你的 Home Assistant 配置文件，请谨慎使用并确认有备份。


## 常见问题（FAQ）
- 401 或“API 密钥无效”：请在 https://open.bigmodel.cn/usercenter/apikeys 重新生成并粘贴正确值。
- 对话不流式：确认已将 Assist 代理切换为本集成，查看日志是否有 API 错误。
- 图片生成失败：检查模型与尺寸是否受支持，稍后重试或更换模型。
- 未看到子条目/实体：请升级 Home Assistant 至较新版本并重启。
- TTS 播放无声：检查 media_player 是否支持 WAV，或将响应格式改为 wav 并保持 encode 为 base64。


## 参与贡献
欢迎提交 Issue 与 PR，帮助完善功能与文档：
- 代码与文档：https://github.com/Desmond-Dong/zhipuai
- 问题反馈：https://github.com/Desmond-Dong/zhipuai/issues


## 许可协议
本项目遵循仓库内 LICENSE 协议发布。

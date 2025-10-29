<h1 align="center">ZhipuAI · Home Assistant Integration</h1>
<p align="center">
  <img src="https://github.com/user-attachments/assets/f8ff7a6c-4449-496a-889a-d205469a84df" alt="ZhipuAI" width="700" height="400" />
</p>

<p align="center">
  <a href="https://github.com/Desmond-Dong/zhipuai/releases"><img src="https://img.shields.io/github/v/release/Desmond-Dong/zhipuai" alt="GitHub Version"></a>
  <a href="https://github.com/Desmond-Dong/zhipuai/issues"><img src="https://img.shields.io/github/issues/Desmond-Dong/zhipuai" alt="GitHub Issues"></a>
  <img src="https://img.shields.io/github/forks/Desmond-Dong/zhipuai?style=social" alt="GitHub Forks">
  <img src="https://img.shields.io/github/stars/Desmond-Dong/zhipuai?style=social" alt="GitHub Stars">
  <a href="README.md">中文</a>
</p>

---

ZhipuAI is a custom Home Assistant integration providing native connectivity with Zhipu's large language models:
- **Conversation Assistant**: Acts as an Assist conversation agent, supporting streaming responses, home control tools, and image understanding.
- **AI Task**: Generate structured data and images (CogView series).
- **Text-to-Speech (TTS)**: Convert text to speech (supports streaming).
- **Speech-to-Text (STT)**: Transcribe speech to text (supports streaming).
- **Integrated Services**: Image analysis, image generation, TTS playback, and STT transcription.
- **AI Automation (experimental)**: Generate and write automations.yaml using natural language and auto-reload.

## Features

### Conversation Assistant
- **Streaming Output**: Real-time model replies.
- **Home Control**: Connects to Home Assistant LLM API, enabling the model to call tools such as "control/query".
- **Image Understanding**: Automatically switches to a vision model when messages contain images (prioritizes free glm-4v-flash).
- **Context Memory**: Configurable message history count to balance effectiveness and performance.
- **Optional Web Search**: Enable web_search tool in advanced mode.

### AI Task
- **Structured Data Generation**: Specify a JSON structure (provides error hints on failure).
- **Image Generation**: Integrates with images/generations, supports URL or base64 output, always returns PNG.
- **Attachment Support**: Reuses message format for easy multimodal AI tasks.

### Text-to-Speech (TTS)
- **Multiple Voices**: (tongtong, xiaochen, chuichui, jam, kazi, douji, luodo).
- **Streaming/Non-Streaming**: Automatically parses fragments and merges them into WAV audio.
- **Configurable Speed/Volume/Encoding**: Set `speed`, `volume`, `response_format` (pcm/wav), `encode_format` (base64/raw).

### Speech-to-Text (STT)
- **WAV Input Preferred**: Automatically converts/normalizes audio to compatible WAV (16k/16bit/mono).
- **Streaming/Non-Streaming**: Returns recognition results line-by-line as data: JSON.
- **Language Support**: Common language codes such as Chinese and English.

### Configuration & Management
- **Recommended/Advanced Modes**: Default recommended parameters for quick start; advanced mode opens model and parameter settings.
- **Subentries**: Manage Conversation / AI Task / TTS / STT separately within one integration.
- **Version Migration**: Automatically migrate old versions to the new structure.

## Installation

### Method 1: HACS (Recommended)
1. Search for and install "zhipuai" in HACS.
2. Restart Home Assistant.

### Method 2: Manual Installation
1. Copy the `custom_components/zhipuai` directory from this repo to your Home Assistant `custom_components/` config directory.
2. Restart Home Assistant.

*Note: This integration depends on the newer Conversation/AI Task/Subentry frameworks. It is recommended to use a recent HA version.*

## Quick Start (Configuration Wizard)
1. Go to Settings → Devices & Services → Integrations → Add Integration, search for "ZhipuAI".
2. Enter your Zhipu API Key as prompted (get it [here](https://open.bigmodel.cn/usercenter/apikeys)).
   - If you don't have a Zhipu account, you can [register here](https://www.bigmodel.cn/invite?icode=NWiYEUi2tleEV8cplkb1Z%2BZLO2QH3C0EBTSr%2BArzMw4%3D/).
3. Four subentries will be automatically created:
   - Conversation Assistant (`conversation`)
   - AI Task (`ai_task_data`)
   - Text-to-Speech (`tts`)
   - Speech-to-Text (`stt`)
4. To adjust parameters, click "Configure" in the desired subentry and choose recommended or advanced settings.

*Parameter Verification*: The system will verify your API Key and network in real time. If it fails, you'll see errors like "invalid key/unreachable/unknown error".

## User Guide

### A. Conversation Assistant (Assist Agent)
- In the Assist page, set the agent to "ZhipuAI Conversation Assistant".
- Say/type things like: "Turn living room lights to 60%", "Summarize today’s schedule".
- Attach images: Upload or reference images on supported frontends for visual analysis and answers.
- Tool calls: Enable LLM Hass API in subentries so the model can operate Home Assistant tools for control/query.

### B. AI Task (Structured Data)
- Call AI Task in cards or automations to generate data:
  - Text → Structured JSON: Once you define the structure, replies are parsed as JSON.
  - On parse failure, errors are returned for easier tuning/retries.

### C. AI Task (Image Generation)
- Use service `zhipuai.generate_image` to create images (CogView):
  - Params: prompt (required), size (default 1024x1024), model (default cogview-3-flash).
  - Output: `image_url` or `image_base64` (auto-converted to PNG for frontend/card use).

### D. Image Analysis Service (Visual Understanding)
- Use service `zhipuai.analyze_image`:
  - Params: `image_file` or `image_entity`, `message` (analysis notes), model (default glm-4v-flash).
  - Streaming (stream) supported for incremental results.

### E. Text-to-Speech (TTS Entities & Services)
- Choose "Zhipu TTS" as a TTS entity in the frontend and input text for playback.
- Or call `zhipuai.tts_speech` service:
  - Params: `text`, `voice` (default tongtong), `speed` (0.25~4.0), `volume` (0.1~2.0), `response_format`, `encode_format`, `stream`, `media_player_entity` (optional direct playback).
  - Non-streaming: returns complete audio; streaming: internal merging then plays/returns WAV.

### F. Speech-to-Text (STT Entities & Services)
- As an STT entity: Audio from frontend/mic is normalized to WAV and uploaded to Zhipu STT.
- Or call `zhipuai.stt_transcribe` service:
  - Params: `audio_file` (WAV), `model` (default glm-asr), `temperature`, `language` (default zh), `stream`.
  - Output: Transcribed text (streaming will concatenate until [DONE]).

## YAML Service Parameter Examples

1) Image Generation

```yaml
service: zhipuai.generate_image
data:
  prompt: "A cat in a spacesuit running on the moon"
  size: "1024x1024"
  model: "cogview-3-flash"
```

2) Image Analysis

```yaml
service: zhipuai.analyze_image
data:
  image_file: "/config/www/cats/cat.jpg"
  message: "Describe the scene and main subject of the image"
  model: "glm-4v-flash"
  stream: false
```

3) Text-to-Speech with Media Player

```yaml
service: zhipuai.tts_speech
data:
  text: "Welcome home, the living room light has been turned on for you."
  voice: "tongtong"
  speed: 1.0
  volume: 1.0
  response_format: "wav"
  encode_format: "base64"
  stream: false
  media_player_entity: media_player.living_room
```

4) Speech-to-Text

```yaml
service: zhipuai.stt_transcribe
data:
  audio_file: "/config/www/records/command.wav"
  model: "glm-asr"
  temperature: 0.95
  language: "zh"
  stream: true
```

## Parameters & Models (Default & Recommendations)
- **Conversation** (default):
  - Model: GLM-4-Flash-250414
  - temperature: 0.3, top_p: 0.5, top_k: 1, max_tokens: 250, history: 30
- **AI Task** (default):
  - Text model: GLM-4-Flash-250414 (temperature 0.95 / top_p 0.7 / max_tokens 2000)
  - Image model: cogview-3-flash (size default 1024x1024)
- **Vision model priority**: glm-4v-flash (free preferred), glm-4v, glm-4v-plus
- **TTS defaults**: model cogtts, voice tongtong, response_format pcm, encode_format base64, speed 1.0, volume 1.0, stream true
- **STT defaults**: model glm-asr, temperature 0.95, language zh, stream true

*Note: Higher temperature, longer history, and longer outputs consume more time/tokens.*

## AI Automation (Experimental)
- In conversation, input things like "Help me create an automation…" or "When... then...", and the system will try to generate standard Home Assistant automation YAML:
  - YAML is written directly to automations.yaml and auto-reloaded.
  - Backup with timestamp is made before writing.
- *Security Warning*: This will modify your Home Assistant config files. Please use with caution and ensure you have backups.

## Frequently Asked Questions (FAQ)
- **401 or "API key invalid"**: Please re-generate your key at https://open.bigmodel.cn/usercenter/apikeys and paste the correct value.
- **Conversation replies are not streaming**: Make sure Assist agent is set to this integration and check the log for API issues.
- **Image generation failed**: Check if the model and size are supported, retry later or switch models.
- **Cannot see subentries/entities**: Please upgrade Home Assistant to a recent version and restart.
- **No sound on TTS playback**: Confirm your media_player supports WAV, or set response_format to wav and keep encode as base64.

## Contributing
Contributions are welcome! Open issues and PRs to help improve features and docs:
- Code & docs: https://github.com/Desmond-Dong/zhipuai
- Issues: https://github.com/Desmond-Dong/zhipuai/issues

## License
This project is released under the LICENSE found in this repository.

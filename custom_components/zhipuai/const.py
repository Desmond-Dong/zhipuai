"""Constants for the 智谱清言 integration."""

from __future__ import annotations

import logging
from typing import Final

# Import llm for API constants
try:
    from homeassistant.helpers import llm
    LLM_API_ASSIST = llm.LLM_API_ASSIST
    DEFAULT_INSTRUCTIONS_PROMPT = llm.DEFAULT_INSTRUCTIONS_PROMPT
except ImportError:
    # Fallback values if llm module is not available
    LLM_API_ASSIST = "assist"
    DEFAULT_INSTRUCTIONS_PROMPT = "你是一个有用的AI助手，请根据用户的问题提供准确、有帮助的回答。"

_LOGGER = logging.getLogger(__name__)
LOGGER = _LOGGER  # 为了向后兼容，提供不带下划线的版本

# Domain
DOMAIN: Final = "zhipuai"

# API Configuration
ZHIPUAI_API_BASE: Final = "https://open.bigmodel.cn/api/paas/v4"
ZHIPUAI_CHAT_URL: Final = f"{ZHIPUAI_API_BASE}/chat/completions"
ZHIPUAI_IMAGE_GEN_URL: Final = f"{ZHIPUAI_API_BASE}/images/generations"
ZHIPUAI_WEB_SEARCH_URL: Final = f"{ZHIPUAI_API_BASE}/web_search"
ZHIPUAI_TTS_URL: Final = f"{ZHIPUAI_API_BASE}/audio/speech"
ZHIPUAI_STT_URL: Final = f"{ZHIPUAI_API_BASE}/audio/transcriptions"

# Timeout
DEFAULT_REQUEST_TIMEOUT: Final = 30000  # milliseconds
TIMEOUT_SECONDS: Final = 30

# Configuration Keys
CONF_API_KEY: Final = "api_key"
CONF_CHAT_MODEL: Final = "chat_model"
CONF_IMAGE_MODEL: Final = "image_model"
CONF_MAX_TOKENS: Final = "max_tokens"
CONF_PROMPT: Final = "prompt"
CONF_TEMPERATURE: Final = "temperature"
CONF_TOP_P: Final = "top_p"
CONF_TOP_K: Final = "top_k"
CONF_LLM_HASS_API: Final = "llm_hass_api"
CONF_RECOMMENDED: Final = "recommended"
CONF_WEB_SEARCH: Final = "web_search"
CONF_MAX_HISTORY_MESSAGES: Final = "max_history_messages"

# Recommended Values for Conversation
RECOMMENDED_CHAT_MODEL: Final = "GLM-4-Flash-250414"
RECOMMENDED_TEMPERATURE: Final = 0.3
RECOMMENDED_TOP_P: Final = 0.5
RECOMMENDED_TOP_K: Final = 1
RECOMMENDED_MAX_TOKENS: Final = 250
RECOMMENDED_MAX_HISTORY_MESSAGES: Final = 30  # Keep last 30 messages for continuous conversation

# Recommended Values for AI Task
RECOMMENDED_AI_TASK_MODEL: Final = "GLM-4-Flash-250414"
RECOMMENDED_AI_TASK_TEMPERATURE: Final = 0.95
RECOMMENDED_AI_TASK_TOP_P: Final = 0.7
RECOMMENDED_AI_TASK_MAX_TOKENS: Final = 2000

# Image Analysis
RECOMMENDED_IMAGE_ANALYSIS_MODEL: Final = "glm-4v-flash"

# Image Generation
RECOMMENDED_IMAGE_MODEL: Final = "cogview-3-flash"

# TTS Configuration
RECOMMENDED_TTS_MODEL: Final = "cogtts"
ZHIPUAI_TTS_MODELS: Final = [
    "cogtts",  # 智谱 TTS 模型
]

# STT Configuration
RECOMMENDED_STT_MODEL: Final = "glm-asr"
ZHIPUAI_STT_MODELS: Final = [
    "glm-asr",  # 智谱 STT 模型
]

# TTS Voice Options
ZHIPUAI_TTS_VOICES: Final = [
    "yongxi",      # 勇熙 - 女声
    "xiuyi",       # 叙怡 - 女声
    "catherine",   # 凯瑟琳 - 女声
    "emma",        # 艾玛 - 女声
    "aaron",       # 亚伦 - 男声
    "brian",       # 布莱恩 - 男声
    "daniel",      # 丹尼尔 - 男声
    "edward",      # 爱德华 - 男声
]

# TTS Audio Formats
ZHIPUAI_TTS_RESPONSE_FORMATS: Final = [
    "pcm",     # PCM 格式 (默认)
    "wav",     # WAV 格式
]

ZHIPUAI_TTS_ENCODE_FORMATS: Final = [
    "base64",  # Base64 编码 (默认)
    "raw",     # 原始数据
]

# TTS Configuration Keys
CONF_TTS_VOICE: Final = "tts_voice"
CONF_TTS_SPEED: Final = "tts_speed"
CONF_TTS_VOLUME: Final = "tts_volume"
CONF_TTS_RESPONSE_FORMAT: Final = "tts_response_format"
CONF_TTS_ENCODE_FORMAT: Final = "tts_encode_format"
CONF_TTS_STREAM: Final = "tts_stream"

# TTS Default Parameters
TTS_DEFAULT_VOICE: Final = "yongxi"  # 默认使用勇熙女声
TTS_DEFAULT_RESPONSE_FORMAT: Final = "pcm"
TTS_DEFAULT_ENCODE_FORMAT: Final = "base64"
TTS_DEFAULT_SPEED: Final = 1.0
TTS_DEFAULT_VOLUME: Final = 1.0
TTS_DEFAULT_STREAM: Final = False

# TTS Parameter Ranges
TTS_SPEED_MIN: Final = 0.25
TTS_SPEED_MAX: Final = 4.0
TTS_SPEED_STEP: Final = 0.1

TTS_VOLUME_MIN: Final = 0.1
TTS_VOLUME_MAX: Final = 2.0
TTS_VOLUME_STEP: Final = 0.1

# STT Configuration
# STT Configuration Keys
CONF_STT_FILE: Final = "audio_file"
CONF_STT_MODEL: Final = "stt_model"
CONF_STT_TEMPERATURE: Final = "stt_temperature"
CONF_STT_LANGUAGE: Final = "stt_language"
CONF_STT_STREAM: Final = "stt_stream"

# STT Default Parameters
STT_DEFAULT_TEMPERATURE: Final = 0.95
STT_DEFAULT_STREAM: Final = False

# STT Parameter Ranges
STT_TEMPERATURE_MIN: Final = 0.0
STT_TEMPERATURE_MAX: Final = 1.0
STT_TEMPERATURE_STEP: Final = 0.05

# STT Audio Formats
ZHIPUAI_STT_AUDIO_FORMATS: Final = [
    "wav",  # WAV 格式
    "mp3",  # MP3 格式
]

# STT File Size Limits
STT_MAX_FILE_SIZE_MB: Final = 25  # 最大文件大小 25MB
STT_MAX_DURATION_SECONDS: Final = 60  # 最大时长 60秒
IMAGE_SIZES: Final = [
    "1024x1024",
    "768x1344",
    "864x1152",
    "1344x768",
    "1152x864",
    "1440x720",
    "720x1440",
]

# Available Models
ZHIPUAI_CHAT_MODELS: Final = [
    "GLM-4-Flash",          # GLM-4-Flash - 免费通用，128K/16K，免费
    "glm-4.5-flash",        # GLM-4.5-Flash - 免费通用模型，128K/16K，免费使用，解码速度20-25tokens/秒
    "GLM-4-Flash-250414",   # GLM-4-Flash-250414 - 免费通用，128K/16K，免费
    "GLM-Z1-Flash",         # GLM-Z1-Flash - 免费推理，128K/32K，免费
    "GLM-4-FlashX-250414",  # GLM-4-FlashX-250414 - 高速低价，128K/4K，0.1元/百万tokens，支持0.05元优惠价
    "GLM-4-Long",           # GLM-4-Long - 超长输入，1M/4K，1元/百万tokens，批量调用0.5元/ 百万Tokens
    "GLM-4-Air",            # GLM-4-Air - 高性价比，128K/16K，0.5元/百万tokens，支持0.25元优惠价
    "GLM-4-Air-250414",     # GLM-4-Air-250414 - 高性价比，128K/16K，0.5元/百万tokens，支持0.25元优惠价
    "GLM-4-AirX",           # GLM-4-AirX - 极速推理，8K/4K，10元/百万tokens
    "GLM-Z1-Air",           # GLM-Z1-Air - 轻量推理，128K/32K，0.5元/百万tokens
    "GLM-Z1-AirX",          # GLM-Z1-AirX - 极速推理，32K/30K，5元/百万tokens
    "GLM-Z1-FlashX-250414", # GLM-Z1-FlashX-250414 - 低价推理，128K/32K，0.5元/百万tokens
    "glm-4.5",              # GLM-4.5 - 通用最强大模型，输入长度[0,32]/输出[0,0.2]：1元，输出[0.2+]：1.5元，长文本[32,128]：2元，解码速度30-50tokens/秒
    "glm-4.5-x",            # GLM-4.5-X - 高性能大模型，输入长度[0,32]/输出[0,0.2]：4元，输出[0.2+]：6元，长文本[32,128]：8元，解码速度60-100tokens/秒
    "glm-4.5-air",          # GLM-4.5-Air - 轻量级模型，输入长度[0,32]/输出[0,0.2]：0.4元，输出[0.2+]：0.4元，长文本[32,128]：0.6元，解码速度30-50tokens/秒
    "glm-4.5-airx",         # GLM-4.5-AirX - 快速推理模型，输入长度[0,32]/输出[0,0.2]：2元，输出[0.2+]：2元，长文本[32,128]：4元，解码速度60-100tokens/秒
    "GLM-4-Plus",           # GLM-4-Plus - 旧智能旗舰，128K/4K，5元/百万tokens，批量2.5元/ 百万Tokens
    "GLM-4-0520",           # GLM-4-0520 - 稳定版本，128K/4K，100元/百万tokens
    "GLM-4-AllTools",       # GLM-4-AllTools - 全能工具，128K/32K，1元/百万tokens
    "GLM-4-Assistant",      # GLM-4-Assistant - 全智能体，128K/4K，5元/百万tokens
    "GLM-4-CodeGeex-4",     # GLM-4-CodeGeex - 代码生成，128K/32K，0.1元/百万Tokens
    "GLM-4V-Flash",         # GLM-4V-Flash - （无官方定价说明，支持多模态/图像处理）
    "GLM-4V-Plus",          # GLM-4V-Plus - （无官方定价说明，支持多模态/图像处理）
    "CharGLM-4",            # CharGLM-4 - 拟人对话，8K/4K，1元/百万tokens
    "glm-zero-preview",     # glm-zero-preview - （无官方定价说明/暂未公开）
]

ZHIPUAI_IMAGE_MODELS: Final = [
    "cogview-3-flash",      # CogView-3 Flash (免费)
    "cogview-3-plus",       # CogView-3 Plus
    "cogview-3",            # CogView-3

]

# Vision Models (支持图像分析) - 优先使用免费模型
VISION_MODELS: Final = [
    "glm-4v-flash",      # GLM-4V-Flash - 免费视觉模型（推荐）
    "glm-4v",            # GLM-4V - 收费视觉模型
    "glm-4v-plus",        # GLM-4V-Plus - 收费视觉模型
]

# Default Names
DEFAULT_TITLE: Final = "智谱清言"
DEFAULT_CONVERSATION_NAME: Final = "智谱对话助手"
DEFAULT_AI_TASK_NAME: Final = "智谱AI任务"
DEFAULT_TTS_NAME: Final = "智谱TTS语音"
DEFAULT_STT_NAME: Final = "智谱STT语音"

# Services
SERVICE_GENERATE_IMAGE: Final = "generate_image"
SERVICE_ANALYZE_IMAGE: Final = "analyze_image"
SERVICE_TTS_SPEECH: Final = "tts_speech"
SERVICE_STT_TRANSCRIBE: Final = "stt_transcribe"

# Error Messages
ERROR_GETTING_RESPONSE: Final = "获取响应时出错"
ERROR_INVALID_API_KEY: Final = "API密钥无效"
ERROR_CANNOT_CONNECT: Final = "无法连接到智谱AI服务"

# Web Search Tool
WEB_SEARCH_TOOL: Final = {
    "type": "web_search",
    "web_search": {
        "enable": False,
        "search_query": ""
    }
}

# Recommended Options
RECOMMENDED_CONVERSATION_OPTIONS: Final = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: LLM_API_ASSIST,
    CONF_PROMPT: DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
    CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
    CONF_TOP_P: RECOMMENDED_TOP_P,
    CONF_TOP_K: RECOMMENDED_TOP_K,
    CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
    CONF_MAX_HISTORY_MESSAGES: RECOMMENDED_MAX_HISTORY_MESSAGES,
    CONF_WEB_SEARCH: False,
}

RECOMMENDED_AI_TASK_OPTIONS: Final = {
    CONF_RECOMMENDED: True,
    CONF_CHAT_MODEL: RECOMMENDED_AI_TASK_MODEL,
    CONF_TEMPERATURE: RECOMMENDED_AI_TASK_TEMPERATURE,
    CONF_TOP_P: RECOMMENDED_AI_TASK_TOP_P,
    CONF_MAX_TOKENS: RECOMMENDED_AI_TASK_MAX_TOKENS,
    CONF_IMAGE_MODEL: RECOMMENDED_IMAGE_MODEL,
}

RECOMMENDED_TTS_OPTIONS: Final = {
    CONF_RECOMMENDED: True,
    CONF_CHAT_MODEL: RECOMMENDED_TTS_MODEL,
    CONF_TTS_VOICE: TTS_DEFAULT_VOICE,
    CONF_TTS_SPEED: TTS_DEFAULT_SPEED,
    CONF_TTS_VOLUME: TTS_DEFAULT_VOLUME,
    CONF_TTS_RESPONSE_FORMAT: TTS_DEFAULT_RESPONSE_FORMAT,
    CONF_TTS_ENCODE_FORMAT: TTS_DEFAULT_ENCODE_FORMAT,
    CONF_TTS_STREAM: TTS_DEFAULT_STREAM,
}

RECOMMENDED_STT_OPTIONS: Final = {
    CONF_RECOMMENDED: True,
    CONF_CHAT_MODEL: RECOMMENDED_STT_MODEL,
    CONF_STT_TEMPERATURE: STT_DEFAULT_TEMPERATURE,
    CONF_STT_LANGUAGE: "zh",  # 默认中文
    CONF_STT_STREAM: STT_DEFAULT_STREAM,
}

"""Constants for the 智谱清言 integration."""

from __future__ import annotations

import logging
from typing import Final

from homeassistant.core import HomeAssistant

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


def get_localized_name(hass: HomeAssistant, zh_name: str, en_name: str) -> str:
    """根据Home Assistant语言设置返回本地化名称."""
    language = hass.config.language

    # 中文语言代码列表
    chinese_languages = ["zh", "zh-cn", "zh-hans", "zh-hant", "zh-tw", "zh-hk"]

    if language and language.lower() in chinese_languages:
        return zh_name
    else:
        return en_name

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
RECOMMENDED_CHAT_MODEL: Final = "glm-4.7-flash"
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
RECOMMENDED_IMAGE_ANALYSIS_MODEL: Final = "glm-4.6v-flash"

# Image Generation
RECOMMENDED_IMAGE_MODEL: Final = "cogview-3-flash"

# TTS Configuration
RECOMMENDED_TTS_MODEL: Final = "glm-tts"
ZHIPUAI_TTS_MODELS: Final = [
    "glm-tts",  # GLM-TTS - 新一代语音合成模型（推荐）
    "cogtts",   # CogTTS - 旧版 TTS 模型
]

# STT Configuration
RECOMMENDED_STT_MODEL: Final = "glm-asr-2512"
ZHIPUAI_STT_MODELS: Final = [
    "glm-asr-2512",  # GLM-ASR-2512 - 新一代高精度语音识别模型（推荐）
    "glm-asr",       # GLM-ASR - 旧版语音识别模型
]

# TTS Voice Options
ZHIPUAI_TTS_VOICES: Final = [
    "tongtong",    # 彤彤 - 女声（默认）
    "xiaochen",    # 小陈 - 女声
    "chuichui",    # 锤锤 - 女声
    "jam",         # jam - 女声
    "kazi",        # kazi - 女声
    "douji",       # douji - 女声
    "luodo",       # luodo - 女声
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
TTS_DEFAULT_VOICE: Final = "tongtong"  # 默认使用彤彤女声
TTS_DEFAULT_RESPONSE_FORMAT: Final = "pcm"
TTS_DEFAULT_ENCODE_FORMAT: Final = "base64"
TTS_DEFAULT_SPEED: Final = 1.0
TTS_DEFAULT_VOLUME: Final = 1.0
TTS_DEFAULT_STREAM: Final = True

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
STT_DEFAULT_STREAM: Final = True

# STT Parameter Ranges
STT_TEMPERATURE_MIN: Final = 0.0
STT_TEMPERATURE_MAX: Final = 1.0
STT_TEMPERATURE_STEP: Final = 0.05

# STT Audio Formats
ZHIPUAI_STT_AUDIO_FORMATS: Final = [
    "wav",  # WAV 格式 (智谱AI官方支持)
    # 注意：虽然官方支持MP3，但Home Assistant的STT组件对MP3处理比较复杂
    # 建议使用WAV格式以获得最佳兼容性
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

# Available Models (based on https://docs.bigmodel.cn/cn/guide/start/model-overview)
ZHIPUAI_CHAT_MODELS: Final = [
    # 免费文本模型
    "glm-4.7-flash",        # GLM-4.7-Flash - 免费文本模型，200K上下文（推荐）
    "glm-4.5-flash",        # GLM-4.5-Flash - 免费文本模型，128K上下文
    "GLM-4-Flash-250414",   # GLM-4-Flash-250414 - 免费文本模型，128K上下文
    # 旗舰/最新文本模型
    "glm-5.3",              # GLM-5.3 - 最新旗舰，1M上下文（始终开启思考）
    "glm-5.2",              # GLM-5.2 - 旗舰模型，1M上下文
    "glm-5.1",              # GLM-5.1 - 200K上下文
    "glm-5",                # GLM-5 - 200K上下文
    "glm-5-turbo",          # GLM-5-Turbo - 长任务优化，200K上下文
    "glm-4.7",              # GLM-4.7 - 通用对话/推理/智能体，200K上下文
    "glm-4.7-flashx",       # GLM-4.7-FlashX - 轻量高速，200K上下文
    "glm-4.6",              # GLM-4.6 - 高级编码/复杂推理/工具调用，200K上下文
    "glm-4.5-air",          # GLM-4.5-Air - 轻量模型，128K上下文
    "glm-4.5-airx",         # GLM-4.5-AirX - 极速版本，128K上下文
    "GLM-4-Long",           # GLM-4-Long - 超长文本，1M上下文
    "GLM-4-FlashX-250414",  # GLM-4-FlashX-250414 - 高速版本，128K上下文
    # 原生多模态模型（支持图片/视频/文件理解）
    "glm-5.3-flash",        # GLM-5.3-Flash - 原生多模态，1M上下文（始终开启思考）
]

ZHIPUAI_IMAGE_MODELS: Final = [
    "cogview-3-flash",      # CogView-3-Flash (免费)
    "glm-image",            # GLM-Image - 旗舰图像生成模型
    "cogView-4-250304",     # CogView-4 - 支持汉字生成
]

# Vision Models (支持图像分析) - 优先使用免费模型
VISION_MODELS: Final = [
    "glm-4.6v-flash",           # GLM-4.6V-Flash - 免费视觉推理模型，128K上下文（推荐）
    "glm-4.1v-thinking-flash",  # GLM-4.1V-Thinking-Flash - 免费视觉推理模型，64K上下文
    "glm-4v-flash",             # GLM-4V-Flash - 免费图像理解模型，16K上下文
    "glm-4.6v",                 # GLM-4.6V - 视觉模型，128K上下文
    "glm-4.1v-thinking-flashx", # GLM-4.1V-Thinking-FlashX - 高并发视觉推理，64K上下文
    "glm-5v-turbo",             # GLM-5V-Turbo - 多模态Coding基座，200K上下文
]

# Default Names
DEFAULT_TITLE: Final = "智谱清言"
DEFAULT_CONVERSATION_NAME: Final = "智谱对话助手"
DEFAULT_AI_TASK_NAME: Final = "智谱AI任务"
DEFAULT_TTS_NAME: Final = "智谱TTS语音"
DEFAULT_TTS_NAME_EN: Final = "ZhipuAI TTS"
DEFAULT_STT_NAME: Final = "智谱STT语音"
DEFAULT_STT_NAME_EN: Final = "ZhipuAI STT"
DEFAULT_CONVERSATION_NAME_EN: Final = "ZhipuAI Assistant"
DEFAULT_AI_TASK_NAME_EN: Final = "ZhipuAI Task"

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

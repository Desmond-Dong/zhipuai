"""Constants for the 智谱清言 integration."""

from __future__ import annotations

from typing import Final

# Domain
DOMAIN: Final = "zhipuai"

# API Configuration
ZHIPUAI_API_BASE: Final = "https://open.bigmodel.cn/api/paas/v4"
ZHIPUAI_CHAT_URL: Final = f"{ZHIPUAI_API_BASE}/chat/completions"
ZHIPUAI_IMAGE_GEN_URL: Final = f"{ZHIPUAI_API_BASE}/images/generations"
ZHIPUAI_WEB_SEARCH_URL: Final = f"{ZHIPUAI_API_BASE}/web_search"

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
RECOMMENDED_CHAT_MODEL: Final = "GLM-4-Flash"
RECOMMENDED_TEMPERATURE: Final = 0.3
RECOMMENDED_TOP_P: Final = 0.5
RECOMMENDED_TOP_K: Final = 1
RECOMMENDED_MAX_TOKENS: Final = 250
RECOMMENDED_MAX_HISTORY_MESSAGES: Final = 30  # Keep last 30 messages for continuous conversation

# Recommended Values for AI Task
RECOMMENDED_AI_TASK_MODEL: Final = "GLM-4-Flash"
RECOMMENDED_AI_TASK_TEMPERATURE: Final = 0.95
RECOMMENDED_AI_TASK_TOP_P: Final = 0.7
RECOMMENDED_AI_TASK_MAX_TOKENS: Final = 2000

# Image Generation
RECOMMENDED_IMAGE_MODEL: Final = "cogview-3-flash"
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
    "GLM-4-Flash",
    "GLM-4-Plus",
    "GLM-4-Air",
    "GLM-4-AirX",
    "GLM-4-Long",
    "GLM-4-AllTools",
    "GLM-4-0520",
    "GLM-4V-Flash",
    "GLM-4V-Plus",
    "GLM-Z1-Flash",
    "GLM-Z1-Air",
    "GLM-Z1-AirX",
]

ZHIPUAI_IMAGE_MODELS: Final = [
    "cogview-3-flash",
    "cogview-3",
    "cogview-3-plus",
]

# Vision Models (支持图像分析)
VISION_MODELS: Final = [
    "GLM-4V-Flash",
    "GLM-4V-Plus",
    "GLM-4V",
]

# Default Names
DEFAULT_TITLE: Final = "智谱清言"
DEFAULT_CONVERSATION_NAME: Final = "智谱对话助手"
DEFAULT_AI_TASK_NAME: Final = "智谱AI任务"

# Services
SERVICE_GENERATE_IMAGE: Final = "generate_image"

# Error Messages
ERROR_GETTING_RESPONSE: Final = "获取响应时出错"
ERROR_INVALID_API_KEY: Final = "API密钥无效"
ERROR_CANNOT_CONNECT: Final = "无法连接到智谱AI服务"

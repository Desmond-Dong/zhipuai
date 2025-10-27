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

# Services
SERVICE_GENERATE_IMAGE: Final = "generate_image"
SERVICE_ANALYZE_IMAGE: Final = "analyze_image"

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

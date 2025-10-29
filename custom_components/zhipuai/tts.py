"""Text to speech support for 智谱清言."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any

import aiohttp
from propcache.api import cached_property

from homeassistant.components.tts import (
    ATTR_VOICE,
    TextToSpeechEntity,
    TtsAudioType,
    Voice,
)
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_API_KEY,
    CONF_CHAT_MODEL,
    CONF_TTS_VOICE,
    CONF_TTS_SPEED,
    CONF_TTS_VOLUME,
    CONF_TTS_RESPONSE_FORMAT,
    CONF_TTS_ENCODE_FORMAT,
    CONF_TTS_STREAM,
    DEFAULT_REQUEST_TIMEOUT,
    LOGGER,
    RECOMMENDED_TTS_MODEL,
    TTS_DEFAULT_ENCODE_FORMAT,
    TTS_DEFAULT_RESPONSE_FORMAT,
    TTS_DEFAULT_SPEED,
    TTS_DEFAULT_STREAM,
    TTS_DEFAULT_VOICE,
    TTS_DEFAULT_VOLUME,
    TTS_SPEED_MAX,
    TTS_SPEED_MIN,
    TTS_VOLUME_MAX,
    TTS_VOLUME_MIN,
    ZHIPUAI_TTS_ENCODE_FORMATS,
    ZHIPUAI_TTS_MODELS,
    ZHIPUAI_TTS_RESPONSE_FORMATS,
    ZHIPUAI_TTS_URL,
    ZHIPUAI_TTS_VOICES,
)
from .entity import ZhipuAIEntityBase
from .helpers import decode_base64_audio, parse_streaming_response

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up TTS entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "tts":
            continue

        async_add_entities(
            [ZhipuaiTextToSpeechEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class ZhipuaiTextToSpeechEntity(TextToSpeechEntity, ZhipuAIEntityBase):
    """智谱 AI text-to-speech entity."""

    _attr_supported_options = [
        ATTR_VOICE,
        "speed",
        "volume",
        "response_format",
        "encode_format",
        "stream"
    ]

    # 智谱 AI TTS 支持中文
    _attr_supported_languages = [
        "zh-CN",  # 中文（主要支持）
        "en-US",  # 英文（部分支持）
    ]

    _attr_default_language = "zh-CN"

    # 支持的语音
    _supported_voices = [
        Voice("tongtong", "彤彤"),
        Voice("xiaochen", "小陈"),
        Voice("chuichui", "锤锤"),
        Voice("jam", "jam"),
        Voice("kazi", "kazi"),
        Voice("douji", "douji"),
        Voice("luodo", "luodo"),
    ]

    def __init__(self, config_entry: ConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the TTS entity."""
        super().__init__(config_entry, subentry, RECOMMENDED_TTS_MODEL)

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice]:
        """Return a list of supported voices for a language."""
        return self._supported_voices

    @cached_property
    def default_options(self) -> Mapping[str, Any]:
        """Return a mapping with the default options."""
        return {
            ATTR_VOICE: TTS_DEFAULT_VOICE,
            "speed": TTS_DEFAULT_SPEED,
            "volume": TTS_DEFAULT_VOLUME,
            "response_format": TTS_DEFAULT_RESPONSE_FORMAT,
            "encode_format": TTS_DEFAULT_ENCODE_FORMAT,
            "stream": TTS_DEFAULT_STREAM,
        }

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Load tts audio file from the engine."""
        if not message or not message.strip():
            raise HomeAssistantError("文本内容不能为空")

        # 获取参数
        voice = options.get(ATTR_VOICE, options.get(CONF_TTS_VOICE, TTS_DEFAULT_VOICE))
        speed = float(options.get("speed", options.get(CONF_TTS_SPEED, TTS_DEFAULT_SPEED)))
        volume = float(options.get("volume", options.get(CONF_TTS_VOLUME, TTS_DEFAULT_VOLUME)))
        response_format = options.get("response_format", options.get(CONF_TTS_RESPONSE_FORMAT, TTS_DEFAULT_RESPONSE_FORMAT))
        encode_format = options.get("encode_format", options.get(CONF_TTS_ENCODE_FORMAT, TTS_DEFAULT_ENCODE_FORMAT))
        stream = options.get("stream", options.get(CONF_TTS_STREAM, TTS_DEFAULT_STREAM))

        # 验证参数
        if voice not in ZHIPUAI_TTS_VOICES:
            raise HomeAssistantError(f"不支持的语音: {voice}")

        if response_format not in ZHIPUAI_TTS_RESPONSE_FORMATS:
            raise HomeAssistantError(f"不支持的响应格式: {response_format}")

        if encode_format not in ZHIPUAI_TTS_ENCODE_FORMATS:
            raise HomeAssistantError(f"不支持的编码格式: {encode_format}")

        if not TTS_SPEED_MIN <= speed <= TTS_SPEED_MAX:
            raise HomeAssistantError(f"语速必须在 {TTS_SPEED_MIN} 到 {TTS_SPEED_MAX} 之间")

        if not TTS_VOLUME_MIN <= volume <= TTS_VOLUME_MAX:
            raise HomeAssistantError(f"音量必须在 {TTS_VOLUME_MIN} 到 {TTS_VOLUME_MAX} 之间")

        # 构建请求
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": RECOMMENDED_TTS_MODEL,
            "input": message,
            "voice": voice,
            "response_format": response_format,
            "encode_format": encode_format,
            "stream": stream,
            "speed": speed,
            "volume": volume,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=DEFAULT_REQUEST_TIMEOUT / 1000)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    ZHIPUAI_TTS_URL,
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        _LOGGER.error(
                            "智谱AI TTS API 错误: %s - %s",
                            response.status,
                            error_text
                        )
                        raise HomeAssistantError(
                            f"智谱AI TTS API 请求失败: {response.status}"
                        )

                    if stream:
                        # 处理流式响应
                        response_text = await response.text()
                        audio_chunks = parse_streaming_response(response_text)

                        if not audio_chunks:
                            raise HomeAssistantError("未从流式响应中获取到音频数据")

                        # 合并音频块
                        combined_audio = audio_chunks[0]  # 对于 TTS，通常第一个块就包含完整数据

                        # 如果有多个块，尝试合并
                        if len(audio_chunks) > 1:
                            try:
                                from .helpers import combine_audio_chunks
                                combined_audio = combine_audio_chunks(audio_chunks)
                            except Exception as exc:
                                _LOGGER.warning("音频合并失败，使用第一个音频块: %s", exc)

                        # 解码音频
                        wav_data = decode_base64_audio(combined_audio)
                        return "wav", wav_data
                    else:
                        # 处理非流式响应
                        response_data = await response.json()

                        if "choices" not in response_data or not response_data["choices"]:
                            raise HomeAssistantError("API 响应格式错误")

                        # 从非流式响应中提取音频数据
                        choice = response_data["choices"][0]
                        if "audio" in choice:
                            audio_data = choice["audio"]["content"]
                        elif "message" in choice and "content" in choice["message"]:
                            audio_data = choice["message"]["content"]
                        else:
                            raise HomeAssistantError("无法从响应中提取音频数据")

                        # 解码音频
                        wav_data = decode_base64_audio(audio_data)
                        return "wav", wav_data

        except aiohttp.ClientError as exc:
            _LOGGER.error("智谱AI TTS 网络请求失败: %s", exc)
            raise HomeAssistantError(f"网络请求失败: {exc}") from exc
        except asyncio.TimeoutError as exc:
            _LOGGER.error("智谱AI TTS 请求超时: %s", exc)
            raise HomeAssistantError("请求超时") from exc
        except Exception as exc:
            _LOGGER.error("智谱AI TTS 生成失败: %s", exc, exc_info=True)
            raise HomeAssistantError(f"TTS 生成失败: {exc}") from exc
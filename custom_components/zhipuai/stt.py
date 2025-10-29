"""Speech to Text support for 智谱清言."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any

import aiohttp
from propcache.api import cached_property
from homeassistant.components import stt
from homeassistant.components.stt import SpeechToTextEntity
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_API_KEY,
    CONF_CHAT_MODEL,
    CONF_RECOMMENDED,
    DEFAULT_REQUEST_TIMEOUT,
    LOGGER,
    RECOMMENDED_STT_MODEL,
    STT_DEFAULT_TEMPERATURE,
    STT_DEFAULT_STREAM,
    STT_TEMPERATURE_MAX,
    STT_TEMPERATURE_MIN,
    STT_TEMPERATURE_STEP,
    ZHIPUAI_STT_MODELS,
    ZHIPUAI_STT_URL,
)
from .entity import ZhipuAIEntityBase

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up STT entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "stt":
            continue

        async_add_entities(
            [ZhipuaiSpeechToTextEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class ZhipuaiSpeechToTextEntity(SpeechToTextEntity, ZhipuAIEntityBase):
    """智谱 AI speech-to-text entity."""

    _attr_supported_options = [
        "model",
        "temperature",
        "language",
        "stream",
    ]

    def __init__(self, config_entry: ConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the STT entity."""
        super().__init__(config_entry, subentry, RECOMMENDED_STT_MODEL)
        self._api_key = config_entry.data[CONF_API_KEY]

    
    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return ["zh-CN", "zh-TW", "en-US"]

    @property
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return a list of supported audio formats."""
        return [stt.AudioFormats.WAV, stt.AudioFormats.MP3]

    @property
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return a list of supported codecs."""
        return [stt.AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return a list of supported bit rates."""
        return [stt.AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return a list of supported sample rates."""
        return [stt.AudioSampleRates.SAMPLERATE_16000]

    @cached_property
    def default_options(self) -> Mapping[str, Any]:
        """Return a mapping with the default options."""
        return {
            "model": RECOMMENDED_STT_MODEL,
            "temperature": STT_DEFAULT_TEMPERATURE,
            "stream": STT_DEFAULT_STREAM,
        }

    async def async_process_audio(
        self, audio_data: bytes, metadata: dict[str, Any]
    ) -> str:
        """Process the audio data and return the transcribed text."""
        if not audio_data:
            raise ValueError("音频数据不能为空")

        # 获取配置选项
        options = self.options
        model = options.get("model", RECOMMENDED_STT_MODEL)
        temperature = float(options.get("temperature", STT_DEFAULT_TEMPERATURE))
        stream = options.get("stream", STT_DEFAULT_STREAM)

        # 验证参数
        if model not in ZHIPUAI_STT_MODELS:
            raise ValueError(f"不支持的模型: {model}")

        if not STT_TEMPERATURE_MIN <= temperature <= STT_TEMPERATURE_MAX:
            raise ValueError(f"温度参数必须在 {STT_TEMPERATURE_MIN} 到 {STT_TEMPERATURE_MAX} 之间")

        # 构建请求
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }

        # 准备文件上传
        form_data = aiohttp.FormData()
        form_data.add_field(
            "file",
            audio_data,
            filename="audio.wav",
            content_type="audio/wav"
        )
        form_data.add_field("model", model)
        form_data.add_field("temperature", str(temperature))
        form_data.add_field("stream", str(stream).lower())

        try:
            timeout = aiohttp.ClientTimeout(total=DEFAULT_REQUEST_TIMEOUT / 1000)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    ZHIPUAI_STT_URL,
                    headers=headers,
                    data=form_data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        _LOGGER.error(
                            "智谱AI STT API 错误: %s - %s",
                            response.status,
                            error_text
                        )
                        raise ValueError(f"STT API 请求失败: {response.status}")

                    if stream:
                        # 处理流式响应
                        full_text = ""
                        async for line in response.content:
                            if line:
                                line_text = line.decode('utf-8').strip()
                                if line_text.startswith('data: '):
                                    try:
                                        data_str = line_text[6:]  # Remove 'data: ' prefix
                                        data_dict = json.loads(data_str)

                                        if "text" in data_dict:
                                            full_text += data_dict["text"]
                                    except (json.JSONDecodeError, KeyError) as exc:
                                        _LOGGER.warning("解析流式响应失败: %s", exc)
                                        continue

                        return full_text.strip()
                    else:
                        # 处理非流式响应
                        response_data = await response.json()

                        if "text" not in response_data:
                            _LOGGER.error("STT API 响应格式错误: %s", response_data)
                            raise ValueError("API 响应格式错误")

                        return response_data["text"]

        except aiohttp.ClientError as exc:
            _LOGGER.error("智谱AI STT 网络请求失败: %s", exc)
            raise ValueError(f"网络请求失败: {exc}") from exc
        except asyncio.TimeoutError as exc:
            _LOGGER.error("智谱AI STT 请求超时: %s", exc)
            raise ValueError("请求超时") from exc
        except Exception as exc:
            _LOGGER.error("智谱AI STT 转录失败: %s", exc, exc_info=True)
            raise ValueError(f"STT 转录失败: {exc}") from exc
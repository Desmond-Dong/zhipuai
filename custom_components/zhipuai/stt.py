"""Speech to Text support for 智谱清言."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterable, Mapping
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
from .helpers import convert_to_wav

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

    _attr_has_entity_name = False
    _attr_supported_options = [
        "model",
        "temperature",
        "language",
        "stream",
    ]

    def __init__(self, config_entry: ConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the STT entity."""
        super().__init__(config_entry, subentry, RECOMMENDED_STT_MODEL)

    
    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return ["zh-CN", "zh-TW", "en-US"]

    @property
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return a list of supported audio formats."""
        # 智谱AI官方只支持WAV格式（MP3格式需要特殊处理）
        return [stt.AudioFormats.WAV]

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

    @property
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return a list of supported channels."""
        return [stt.AudioChannels.CHANNEL_MONO]

    @cached_property
    def default_options(self) -> Mapping[str, Any]:
        """Return a mapping with the default options."""
        return {
            "model": RECOMMENDED_STT_MODEL,
            "temperature": STT_DEFAULT_TEMPERATURE,
            "stream": STT_DEFAULT_STREAM,
        }

    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Process an audio stream and return the transcribed text."""
        # 获取配置选项
        options = self.subentry.data
        model = options.get("model", RECOMMENDED_STT_MODEL)
        temperature = float(options.get("temperature", STT_DEFAULT_TEMPERATURE))
        stream_response = options.get("stream", STT_DEFAULT_STREAM)

        # 验证参数
        if model not in ZHIPUAI_STT_MODELS:
            return stt.SpeechResult(
                "",
                stt.SpeechResultState.ERROR
            )

        if not STT_TEMPERATURE_MIN <= temperature <= STT_TEMPERATURE_MAX:
            return stt.SpeechResult(
                "",
                stt.SpeechResultState.ERROR
            )

        try:
            # 收集音频数据
            audio_data = b""
            async for chunk in stream:
                audio_data += chunk

            if not audio_data:
                return stt.SpeechResult(
                    "",
                    stt.SpeechResultState.ERROR
                )

            # 转换音频为WAV格式（确保兼容性）
            if metadata.format != stt.AudioFormats.WAV:
                _LOGGER.info("Converting audio from %s to WAV format", metadata.format.value)
                try:
                    mime_type = f"audio/L{metadata.bit_rate.value};rate={metadata.sample_rate.value}"
                    audio_data = convert_to_wav(audio_data, mime_type)
                    _LOGGER.info("Successfully converted audio to WAV, new size: %d bytes", len(audio_data))
                except Exception as exc:
                    _LOGGER.error("Failed to convert audio to WAV: %s", exc, exc_info=True)
                    # 如果转换失败，继续使用原始数据
                    pass
            else:
                # 即使是WAV格式，也重新生成标准WAV头部以确保兼容性
                _LOGGER.info("Re-generating WAV file for compatibility")
                try:
                    mime_type = f"audio/L{metadata.bit_rate.value};rate={metadata.sample_rate.value}"
                    audio_data = convert_to_wav(audio_data, mime_type)
                    _LOGGER.info("Re-generated WAV file for compatibility, new size: %d bytes", len(audio_data))
                except Exception as exc:
                    _LOGGER.error("Failed to re-generate WAV file: %s", exc, exc_info=True)

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
            form_data.add_field("stream", str(stream_response).lower())

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
                        return stt.SpeechResult(
                            "",
                            stt.SpeechResultState.ERROR
                        )

                    if stream_response:
                        # 处理流式响应
                        full_text = ""
                        async for line in response.content:
                            if line:
                                line_text = line.decode('utf-8').strip()

                                if line_text.startswith('data: '):
                                    try:
                                        data_str = line_text[6:]  # Remove 'data: ' prefix

                                        # 检查是否是结束标记
                                        if data_str == "[DONE]":
                                            break

                                        data_dict = json.loads(data_str)

                                        if "text" in data_dict:
                                            full_text += data_dict["text"]
                                    except (json.JSONDecodeError, KeyError) as exc:
                                        _LOGGER.warning("解析流式响应失败: %s", exc)
                                        continue

                        return stt.SpeechResult(
                            full_text.strip(),
                            stt.SpeechResultState.SUCCESS
                        )
                    else:
                        # 处理非流式响应
                        response_data = await response.json()

                        if "text" not in response_data:
                            _LOGGER.error("STT API 响应格式错误: %s", response_data)
                            return stt.SpeechResult(
                                "",
                                stt.SpeechResultState.ERROR
                            )

                        return stt.SpeechResult(
                            response_data["text"],
                            stt.SpeechResultState.SUCCESS
                        )

        except aiohttp.ClientError as exc:
            _LOGGER.error("智谱AI STT 网络请求失败: %s", exc)
            return stt.SpeechResult(
                "",
                stt.SpeechResultState.ERROR
            )
        except asyncio.TimeoutError as exc:
            _LOGGER.error("智谱AI STT 请求超时: %s", exc)
            return stt.SpeechResult(
                "",
                stt.SpeechResultState.ERROR
            )
        except Exception as exc:
            _LOGGER.error("智谱AI STT 转录失败: %s", exc, exc_info=True)
            return stt.SpeechResult(
                "",
                stt.SpeechResultState.ERROR
            )
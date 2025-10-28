"""Services for 智谱清言 integration."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import aiohttp
import voluptuous as vol
from homeassistant.components import camera
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_API_KEY,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    DEFAULT_REQUEST_TIMEOUT,
    DOMAIN,
    ERROR_GETTING_RESPONSE,
    IMAGE_SIZES,
    RECOMMENDED_IMAGE_ANALYSIS_MODEL,
    RECOMMENDED_IMAGE_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    SERVICE_ANALYZE_IMAGE,
    SERVICE_GENERATE_IMAGE,
    SERVICE_TTS_SPEECH,
    TTS_DEFAULT_ENCODE_FORMAT,
    TTS_DEFAULT_RESPONSE_FORMAT,
    TTS_DEFAULT_SPEED,
    TTS_DEFAULT_STREAM,
    TTS_DEFAULT_VOICE,
    TTS_DEFAULT_VOLUME,
    ZHIPUAI_CHAT_URL,
    ZHIPUAI_IMAGE_GEN_URL,
    ZHIPUAI_TTS_URL,
    ZHIPUAI_TTS_ENCODE_FORMATS,
    ZHIPUAI_TTS_RESPONSE_FORMATS,
    ZHIPUAI_TTS_VOICES,
)

_LOGGER = logging.getLogger(__name__)

# Schema for image analysis service
IMAGE_ANALYZER_SCHEMA = {
    vol.Optional("image_file"): cv.string,
    vol.Optional("image_entity"): cv.entity_id,
    vol.Required("message"): cv.string,
    vol.Optional("model", default=RECOMMENDED_IMAGE_ANALYSIS_MODEL): cv.string,
    vol.Optional("temperature", default=RECOMMENDED_TEMPERATURE): vol.Coerce(float),
    vol.Optional("max_tokens", default=RECOMMENDED_MAX_TOKENS): cv.positive_int,
    vol.Optional("stream", default=False): cv.boolean,
}

# Schema for image generation service
IMAGE_GENERATOR_SCHEMA = {
    vol.Required("prompt"): cv.string,
    vol.Optional("size", default="1024x1024"): vol.In(IMAGE_SIZES),
    vol.Optional("model", default=RECOMMENDED_IMAGE_MODEL): cv.string,
}

# Schema for TTS service
TTS_SCHEMA = {
    vol.Required("text"): cv.string,
    vol.Optional("voice", default=TTS_DEFAULT_VOICE): vol.In(ZHIPUAI_TTS_VOICES),
    vol.Optional("speed", default=TTS_DEFAULT_SPEED): vol.Coerce(float),
    vol.Optional("volume", default=TTS_DEFAULT_VOLUME): vol.Coerce(float),
    vol.Optional("response_format", default=TTS_DEFAULT_RESPONSE_FORMAT): vol.In(ZHIPUAI_TTS_RESPONSE_FORMATS),
    vol.Optional("encode_format", default=TTS_DEFAULT_ENCODE_FORMAT): vol.In(ZHIPUAI_TTS_ENCODE_FORMATS),
    vol.Optional("stream", default=TTS_DEFAULT_STREAM): cv.boolean,
    vol.Optional("media_player_entity"): cv.entity_id,
}


async def async_setup_services(hass: HomeAssistant, config_entry) -> None:
    """Set up services for 智谱清言 integration."""

    api_key = config_entry.runtime_data

    async def handle_analyze_image(call: ServiceCall) -> dict:
        """Handle image analysis service call."""
        try:
            image_data = None

            # Get image from file
            if image_file := call.data.get("image_file"):
                image_data = await _load_image_from_file(hass, image_file)

            # Get image from camera entity
            elif image_entity := call.data.get("image_entity"):
                image_data = await _load_image_from_camera(hass, image_entity)

            if not image_data:
                raise ServiceValidationError("必须提供 image_file 或 image_entity 参数")

            # Resize and convert image to save bandwidth
            processed_image_data = await _process_image(image_data)
            base64_image = base64.b64encode(processed_image_data).decode()

            # Prepare API request
            model = call.data.get("model", RECOMMENDED_IMAGE_ANALYSIS_MODEL)
            message = call.data["message"]
            temperature = call.data.get("temperature", RECOMMENDED_TEMPERATURE)
            max_tokens = call.data.get("max_tokens", RECOMMENDED_MAX_TOKENS)
            stream = call.data.get("stream", False)

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            # Try exact format from ZhipuAI official documentation
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": message
                            }
                        ]
                    }
                ]
            }

            # Only add non-problematic parameters
            if stream:
                payload["stream"] = True

            # Make API call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    ZHIPUAI_CHAT_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        _LOGGER.error("API request failed: %s", error_text)
                        raise HomeAssistantError(f"{ERROR_GETTING_RESPONSE}: {error_text}")

                    if stream:
                        return await _handle_stream_response(hass, response)
                    else:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]
                        return {
                            "success": True,
                            "content": content,
                            "model": model,
                        }

        except Exception as err:
            _LOGGER.error("Error analyzing image: %s", err)
            return {
                "success": False,
                "error": str(err)
            }

    async def handle_generate_image(call: ServiceCall) -> dict:
        """Handle image generation service call."""
        try:
            prompt = call.data["prompt"]
            size = call.data.get("size", "1024x1024")
            model = call.data.get("model", RECOMMENDED_IMAGE_MODEL)

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": model,
                "prompt": prompt,
                "size": size,
            }

            # Make API call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    ZHIPUAI_IMAGE_GEN_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),  # Image generation takes longer
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        _LOGGER.error("Image generation API request failed: %s", error_text)
                        raise HomeAssistantError(f"{ERROR_GETTING_RESPONSE}: {error_text}")

                    result = await response.json()

                    # Process the response based on ZhipuAI's actual API response format
                    if "data" in result and len(result["data"]) > 0:
                        image_data = result["data"][0]
                        image_url = image_data.get("url", "")
                        if image_url:
                            return {
                                "success": True,
                                "image_url": image_url,
                                "prompt": prompt,
                                "size": size,
                                "model": model,
                            }
                        else:
                            # If base64 image is returned instead of URL
                            b64_json = image_data.get("b64_json", "")
                            if b64_json:
                                return {
                                    "success": True,
                                    "image_base64": b64_json,
                                    "prompt": prompt,
                                    "size": size,
                                    "model": model,
                                }

                    raise HomeAssistantError("无法获取生成的图像")

        except Exception as err:
            _LOGGER.error("Error generating image: %s", err)
            return {
                "success": False,
                "error": str(err)
            }

    async def handle_tts_speech(call: ServiceCall) -> dict:
        """Handle TTS service call."""
        try:
            text = call.data["text"]
            voice = call.data.get("voice", TTS_DEFAULT_VOICE)
            speed = float(call.data.get("speed", TTS_DEFAULT_SPEED))
            volume = float(call.data.get("volume", TTS_DEFAULT_VOLUME))
            response_format = call.data.get("response_format", TTS_DEFAULT_RESPONSE_FORMAT)
            encode_format = call.data.get("encode_format", TTS_DEFAULT_ENCODE_FORMAT)
            stream = call.data.get("stream", TTS_DEFAULT_STREAM)
            media_player_entity = call.data.get("media_player_entity")

            # 验证参数
            if not text or not text.strip():
                raise ServiceValidationError("文本内容不能为空")

            if voice not in ZHIPUAI_TTS_VOICES:
                raise ServiceValidationError(f"不支持的语音类型: {voice}")

            if response_format not in ZHIPUAI_TTS_RESPONSE_FORMATS:
                raise ServiceValidationError(f"不支持的响应格式: {response_format}")

            if encode_format not in ZHIPUAI_TTS_ENCODE_FORMATS:
                raise ServiceValidationError(f"不支持的编码格式: {encode_format}")

            if not 0.25 <= speed <= 4.0:
                raise ServiceValidationError("语速必须在 0.25 到 4.0 之间")

            if not 0.1 <= volume <= 2.0:
                raise ServiceValidationError("音量必须在 0.1 到 2.0 之间")

            # 构建 TTS API 请求
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": "cogtts",
                "input": text,
                "voice": voice,
                "response_format": response_format,
                "encode_format": encode_format,
                "stream": stream,
                "speed": speed,
                "volume": volume,
            }

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
                        return {
                            "success": False,
                            "error": f"TTS API 请求失败: {response.status}"
                        }

                    if stream:
                        # 处理流式响应
                        response_text = await response.text()
                        from .helpers import parse_streaming_response, combine_audio_chunks

                        audio_chunks = parse_streaming_response(response_text)

                        if not audio_chunks:
                            return {"success": False, "error": "未从流式响应中获取到音频数据"}

                        # 合并音频块
                        combined_audio = audio_chunks[0]  # 对于 TTS，通常第一个块就包含完整数据

                        # 如果有多个块，尝试合并
                        if len(audio_chunks) > 1:
                            try:
                                combined_audio = combine_audio_chunks(audio_chunks)
                            except Exception as exc:
                                _LOGGER.warning("音频合并失败，使用第一个音频块: %s", exc)

                        audio_base64 = combined_audio
                    else:
                        # 处理非流式响应
                        response_data = await response.json()

                        if "choices" not in response_data or not response_data["choices"]:
                            return {"success": False, "error": "API 响应格式错误"}

                        # 从非流式响应中提取音频数据
                        choice = response_data["choices"][0]
                        if "audio" in choice:
                            audio_base64 = choice["audio"]["content"]
                        elif "message" in choice and "content" in choice["message"]:
                            audio_base64 = choice["message"]["content"]
                        else:
                            return {"success": False, "error": "无法从响应中提取音频数据"}

                    # 解码音频为 WAV 格式
                    from .helpers import decode_base64_audio
                    wav_audio_data = decode_base64_audio(audio_base64)

                    # 如果指定了媒体播放器实体，直接播放
                    if media_player_entity:
                        try:
                            # 将音频数据保存为临时文件
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                                temp_file.write(wav_audio_data)
                                temp_file_path = temp_file.name

                            # 调用媒体播放器的播放服务
                            await hass.services.async_call(
                                "media_player",
                                "play_media",
                                {
                                    "entity_id": media_player_entity,
                                    "media_content_id": f"file://{temp_file_path}",
                                    "media_content_type": "audio/wav",
                                },
                                blocking=True,
                            )

                            # 延迟删除临时文件
                            await asyncio.sleep(1)
                            try:
                                os.unlink(temp_file_path)
                            except OSError:
                                pass  # 文件可能已被系统删除

                            return {
                                "success": True,
                                "message": "语音播放成功",
                                "media_player": media_player_entity,
                            }

                        except Exception as exc:
                            _LOGGER.error("媒体播放失败: %s", exc)
                            return {
                                "success": False,
                                "error": f"媒体播放失败: {exc}",
                                "audio_data": audio_base64,
                            }

                    # 返回音频数据供其他用途
                    return {
                        "success": True,
                        "audio_data": audio_base64,
                        "audio_format": "wav",
                        "voice": voice,
                        "speed": speed,
                        "volume": volume,
                    }

        except ServiceValidationError as exc:
            _LOGGER.error("TTS service validation error: %s", exc)
            return {"success": False, "error": str(exc)}
        except aiohttp.ClientError as exc:
            _LOGGER.error("TTS service network error: %s", exc)
            return {"success": False, "error": f"网络请求失败: {exc}"}
        except asyncio.TimeoutError as exc:
            _LOGGER.error("TTS service timeout: %s", exc)
            return {"success": False, "error": "请求超时"}
        except Exception as exc:
            _LOGGER.error("TTS service error: %s", exc, exc_info=True)
            return {"success": False, "error": f"TTS 生成失败: {exc}"}

    # Register services
    hass.services.async_register(
        DOMAIN,
        SERVICE_ANALYZE_IMAGE,
        handle_analyze_image,
        schema=vol.Schema(IMAGE_ANALYZER_SCHEMA),
        supports_response=True
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_IMAGE,
        handle_generate_image,
        schema=vol.Schema(IMAGE_GENERATOR_SCHEMA),
        supports_response=True
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_TTS_SPEECH,
        handle_tts_speech,
        schema=vol.Schema(TTS_SCHEMA),
        supports_response=True
    )


async def _load_image_from_file(hass: HomeAssistant, image_file: str) -> bytes:
    """Load image from file path."""
    try:
        # Handle relative paths
        if not os.path.isabs(image_file):
            image_file = os.path.join(hass.config.config_dir, image_file)

        if not os.path.exists(image_file):
            raise ServiceValidationError(f"图像文件不存在: {image_file}")

        if os.path.isdir(image_file):
            raise ServiceValidationError(f"提供的路径是一个目录: {image_file}")

        with open(image_file, "rb") as f:
            return f.read()

    except IOError as err:
        raise ServiceValidationError(f"读取图像文件失败: {err}")


async def _load_image_from_camera(hass: HomeAssistant, entity_id: str) -> bytes:
    """Load image from camera entity."""
    try:
        if not entity_id.startswith("camera."):
            raise ServiceValidationError(f"无效的摄像头实体ID: {entity_id}")

        if not hass.states.get(entity_id):
            raise ServiceValidationError(f"摄像头实体不存在: {entity_id}")

        # Get image from camera
        image = await camera.async_get_image(hass, entity_id, timeout=10)

        if not image or not image.content:
            raise ServiceValidationError(f"无法从摄像头获取图像: {entity_id}")

        return image.content

    except (camera.CameraEntityImageError, TimeoutError) as err:
        raise ServiceValidationError(f"获取摄像头图像失败: {err}")


async def _process_image(image_data: bytes, max_size: int = 1024, quality: int = 85) -> bytes:
    """Process image: resize and compress to optimize for API."""
    try:
        # Open image
        from PIL import Image
        img = Image.open(io.BytesIO(image_data))

        # Convert to RGB if necessary (for JPEG compatibility)
        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            if img.mode == "RGBA":
                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
            else:
                background.paste(img)
            img = background

        # Resize if too large
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Compress to JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue()

    except Exception as err:
        _LOGGER.warning("Failed to process image: %s, using original", err)
        return image_data


async def _handle_stream_response(hass: HomeAssistant, response: aiohttp.ClientResponse) -> dict:
    """Handle streaming response from API."""
    event_id = f"zhipuai_image_analysis_{int(time.time())}"

    try:
        hass.bus.async_fire(f"{DOMAIN}_image_analysis_start", {"event_id": event_id})

        accumulated_text = ""
        async for line in response.content:
            if line:
                try:
                    line_text = line.decode('utf-8').strip()
                    if line_text.startswith('data: '):
                        line_text = line_text[6:]  # Remove 'data: ' prefix

                    if line_text == '[DONE]':
                        break

                    if line_text:
                        json_data = json.loads(line_text)
                        if 'choices' in json_data and len(json_data['choices']) > 0:
                            content = json_data['choices'][0].get('delta', {}).get('content', '')
                            if content:
                                accumulated_text += content
                                hass.bus.async_fire(
                                    f"{DOMAIN}_image_analysis_token",
                                    {
                                        "event_id": event_id,
                                        "content": content,
                                        "full_content": accumulated_text
                                    }
                                )
                except json.JSONDecodeError:
                    continue

        return {
            "success": True,
            "content": accumulated_text,
            "stream_event_id": event_id,
        }

    except Exception as err:
        _LOGGER.error("Error handling stream response: %s", err)
        return {
            "success": False,
            "error": str(err)
        }
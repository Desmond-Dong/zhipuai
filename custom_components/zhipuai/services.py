"""Services for 智谱清言 integration."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
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
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    DOMAIN,
    ERROR_GETTING_RESPONSE,
    RECOMMENDED_IMAGE_ANALYSIS_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    SERVICE_ANALYZE_IMAGE,
    ZHIPUAI_CHAT_URL,
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

    # Register service
    hass.services.async_register(
        DOMAIN,
        SERVICE_ANALYZE_IMAGE,
        handle_analyze_image,
        schema=vol.Schema(IMAGE_ANALYZER_SCHEMA),
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
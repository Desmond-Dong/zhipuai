"""AI Task support for 智谱清言."""

from __future__ import annotations

import base64
import io
import logging
from json import JSONDecodeError, loads as json_loads

import aiohttp
from PIL import Image

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_IMAGE_MODEL,
    ERROR_GETTING_RESPONSE,
    IMAGE_SIZES,
    RECOMMENDED_AI_TASK_MODEL,
    RECOMMENDED_IMAGE_MODEL,
)
from .entity import ZhipuAIBaseLLMEntity

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up AI task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "ai_task_data":
            continue

        async_add_entities(
            [ZhipuAITaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class ZhipuAITaskEntity(
    ai_task.AITaskEntity,
    ZhipuAIBaseLLMEntity,
):
    """智谱清言 AI Task entity."""

    def __init__(
        self, entry: ConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry, RECOMMENDED_AI_TASK_MODEL)

        # Supported features
        self._attr_supported_features = (
            ai_task.AITaskEntityFeature.GENERATE_DATA
            | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
            | ai_task.AITaskEntityFeature.GENERATE_IMAGE
        )

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        # Process chat log with optional structure
        await self._async_handle_chat_log(chat_log, task.structure)

        # Get the last assistant message
        text = chat_log.content[-1].content or ""

        # If structure is requested, parse as JSON
        if task.structure:
            try:
                data = json_loads(text)
            except JSONDecodeError as err:
                _LOGGER.error("Failed to parse JSON response: %s", err)
                raise HomeAssistantError(ERROR_GETTING_RESPONSE) from err

            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=data,
            )

        # Otherwise return as text
        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=text,
        )

    async def _async_generate_image(
        self,
        task: ai_task.GenImageTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenImageTaskResult:
        """Handle a generate image task."""
        options = self.subentry.data
        image_model = options.get(CONF_IMAGE_MODEL, RECOMMENDED_IMAGE_MODEL)

        # Get user prompt from chat log
        user_message = chat_log.content[-1]
        prompt = user_message.content

        # Build request parameters (use default size)
        request_params = {
            "model": image_model,
            "prompt": prompt,
            "size": "1024x1024",  # Default size, ZhipuAI supports various sizes
        }

        try:
            # Call ZhipuAI image generation API via HTTP
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            from .const import ZHIPUAI_IMAGE_GEN_URL

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    ZHIPUAI_IMAGE_GEN_URL,
                    json=request_params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise HomeAssistantError(f"Image generation failed: {error_text}")

                    result = await response.json()

                    if not result.get("data") or not result["data"][0].get("url"):
                        raise HomeAssistantError("No image generated")

                    # Download image from URL
                    async with session.get(result["data"][0]["url"]) as img_response:
                        if img_response.status != 200:
                            raise HomeAssistantError(
                                f"Failed to download image: {img_response.status}"
                            )

                        image_bytes = await img_response.read()

            # Convert to PNG
            image = Image.open(io.BytesIO(image_bytes))
            png_buffer = io.BytesIO()
            image.save(png_buffer, format="PNG")
            png_bytes = png_buffer.getvalue()

            return ai_task.GenImageTaskResult(
                conversation_id=chat_log.conversation_id,
                image_data=png_bytes,  # Return bytes, not base64 string
                mime_type="image/png",
                model=image_model,
            )

        except aiohttp.ClientError as err:
            _LOGGER.error("Network error generating image: %s", err)
            raise HomeAssistantError(f"Network error: {err}") from err
        except Exception as err:
            _LOGGER.error("Error generating image: %s", err)
            raise HomeAssistantError(f"Error generating image: {err}") from err

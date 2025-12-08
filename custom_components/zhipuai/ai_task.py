"""AI Task support for 智谱清言."""

from __future__ import annotations

import base64
import io
import logging
from json import JSONDecodeError, loads as json_loads

import aiohttp

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_CHAT_MODEL,
    CONF_IMAGE_MODEL,
    ERROR_GETTING_RESPONSE,
    IMAGE_SIZES,
    RECOMMENDED_AI_TASK_MODEL,
    RECOMMENDED_IMAGE_MODEL,
    RECOMMENDED_IMAGE_ANALYSIS_MODEL,
    VISION_MODELS,
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
        # Use vision model as default for better image support
        default_model = subentry.data.get(CONF_CHAT_MODEL, RECOMMENDED_IMAGE_ANALYSIS_MODEL)
        super().__init__(entry, subentry, default_model)

        # Start with basic features
        self._attr_supported_features = (
            ai_task.AITaskEntityFeature.GENERATE_DATA
            | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
        )

        # Add image generation support if configured
        # Enable for: 1) Vision models, 2) Image generation models, 3) Recommended mode
        model = subentry.data.get(CONF_CHAT_MODEL, default_model)
        is_recommended = subentry.data.get("recommended", False)

        from .const import ZHIPUAI_IMAGE_MODELS

        if (
            model in VISION_MODELS
            or model in ZHIPUAI_IMAGE_MODELS
            or "-image" in model.lower()
            or "cogview" in model.lower()
            or is_recommended  # Always enable in recommended mode
        ):
            self._attr_supported_features |= ai_task.AITaskEntityFeature.GENERATE_IMAGE

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        # Get the actual model being used (may be auto-switched)
        from .const import RECOMMENDED_IMAGE_ANALYSIS_MODEL
        options = self.subentry.data
        configured_model = options.get(CONF_CHAT_MODEL, self.default_model)

        # Check if we need to auto-switch for attachments
        has_attachments = any(
            hasattr(content, 'attachments') and content.attachments
            for content in chat_log.content
        )

        final_model = configured_model
        if has_attachments:
            vision_models = ["glm-4.6v-flash", "glm-4v-flash", "glm-4v", "glm-4v-plus"]  # Use same priority as entity.py (free model first)
            # Check if attachments contain media files
            has_media_attachments = False
            for content in chat_log.content:
                if hasattr(content, 'attachments') and content.attachments:
                    for attachment in content.attachments:
                        mime_type = getattr(attachment, 'mime_type', '')
                        if mime_type.startswith(('image/', 'video/')):
                            has_media_attachments = True
                            break
                if has_media_attachments:
                    break

            if has_media_attachments and configured_model not in vision_models:
                final_model = RECOMMENDED_IMAGE_ANALYSIS_MODEL  # GLM-4V-Flash
                _LOGGER.info("Auto-switched AI Task from %s to vision model %s for media attachments", configured_model, final_model)

        _LOGGER.info("AI Task using final model: %s (configured: %s)", final_model, configured_model)

        # Process chat log with optional structure
        await self._async_handle_chat_log(chat_log, task.structure)

        # Ensure the last message is from assistant
        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            _LOGGER.error(
                "Last content in chat log is not an AssistantContent: %s. This could be due to the model not returning a valid response",
                chat_log.content[-1],
            )
            raise HomeAssistantError(ERROR_GETTING_RESPONSE)

        text = chat_log.content[-1].content or ""

        # If structure is requested, parse as JSON
        if task.structure:
            try:
                # Clean up the response to extract pure JSON
                cleaned_text = text.strip()

                # Remove common prefixes that might appear before JSON
                if cleaned_text.startswith('json'):
                    cleaned_text = cleaned_text[4:].strip()
                elif cleaned_text.startswith('JSON'):
                    cleaned_text = cleaned_text[4:].strip()

                # Remove markdown code blocks if present
                if cleaned_text.startswith('```'):
                    lines = cleaned_text.split('\n')
                    if len(lines) > 1:
                        # Remove first line (```json) and last line (```)
                        cleaned_text = '\n'.join(lines[1:-1]).strip()

                data = json_loads(cleaned_text)
            except JSONDecodeError as err:
                _LOGGER.error(
                    "Failed to parse JSON response: %s. Response: %s",
                    err,
                    text,
                )
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
        assert isinstance(user_message, conversation.UserContent)

        # Build request parameters
        request_params = {
            "model": image_model,
            "prompt": user_message.content,
            "size": "1024x1024",  # Default size, ZhipuAI supports various sizes
        }

        _LOGGER.info("Generating image with model: %s, prompt: %s", image_model, user_message.content[:100])

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
                        _LOGGER.error("Image generation API failed: %s", error_text)
                        raise HomeAssistantError(f"Image generation failed: {error_text}")

                    result = await response.json()
                    _LOGGER.debug("Image generation response: %s", result)

                    if not result.get("data") or len(result["data"]) == 0:
                        raise HomeAssistantError("No image data in response")

                    image_data = result["data"][0]
                    image_url = image_data.get("url")

                    if image_url:
                        # Download image from URL
                        _LOGGER.info("Downloading image from URL: %s", image_url)
                        async with session.get(image_url) as img_response:
                            if img_response.status != 200:
                                raise HomeAssistantError(
                                    f"Failed to download image: {img_response.status}"
                                )

                            image_bytes = await img_response.read()
                            _LOGGER.info("Successfully downloaded image, size: %d bytes", len(image_bytes))
                    else:
                        # Try to get base64 data if URL is not available
                        b64_json = image_data.get("b64_json")
                        if b64_json:
                            _LOGGER.info("Using base64 image data")
                            import base64
                            image_bytes = base64.b64decode(b64_json)
                        else:
                            raise HomeAssistantError("No image URL or base64 data in response")

            # Convert to PNG for better compatibility
            try:
                from PIL import Image
                image = Image.open(io.BytesIO(image_bytes))

                # Convert to RGB if needed
                if image.mode in ("RGBA", "LA", "P"):
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    if image.mode == "P":
                        image = image.convert("RGBA")
                    if image.mode == "RGBA":
                        background.paste(image, mask=image.split()[-1])
                    else:
                        background.paste(image)
                    image = background

                png_buffer = io.BytesIO()
                image.save(png_buffer, format="PNG", optimize=True)
                png_bytes = png_buffer.getvalue()
                _LOGGER.info("Successfully converted image to PNG, size: %d bytes", len(png_bytes))

            except Exception as img_err:
                _LOGGER.warning("Failed to convert image to PNG, using original: %s", img_err)
                png_bytes = image_bytes

            # Add assistant content to chat log
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(
                    agent_id=self.entity_id,
                    content=f"Generated image using {image_model}",
                )
            )

            return ai_task.GenImageTaskResult(
                conversation_id=chat_log.conversation_id,
                image_data=png_bytes,
                mime_type="image/png",
                model=image_model,
            )

        except aiohttp.ClientError as err:
            _LOGGER.error("Network error generating image: %s", err)
            raise HomeAssistantError(f"Network error: {err}") from err
        except Exception as err:
            _LOGGER.error("Error generating image: %s", err)
            raise HomeAssistantError(f"Error generating image: {err}") from err

"""Base entity for 智谱清言 integration."""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator, Callable
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any

import aiohttp
from voluptuous_openapi import convert

from homeassistant.components import conversation, media_source
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import config_entry_flow, device_registry as dr, llm
from homeassistant.helpers.entity import Entity
from homeassistant.util import ulid

from .const import (
    CONF_CHAT_MODEL,
    CONF_LLM_HASS_API,
    CONF_MAX_HISTORY_MESSAGES,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    CONF_WEB_SEARCH,
    DOMAIN,
    ERROR_GETTING_RESPONSE,
    RECOMMENDED_IMAGE_ANALYSIS_MODEL,
    RECOMMENDED_MAX_HISTORY_MESSAGES,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_K,
    RECOMMENDED_TOP_P,
    WEB_SEARCH_TOOL,
    ZHIPUAI_CHAT_URL,
)
from .markdown_filter import filter_markdown_content, filter_markdown_streaming

_LOGGER = logging.getLogger(__name__)


class ZhipuAIBaseLLMEntity(Entity):
    """Base entity for 智谱清言 LLM."""

    _attr_has_entity_name = False
    _attr_should_poll = False

    def __init__(
        self,
        entry: config_entry_flow.ConfigEntry,
        subentry: config_entry_flow.ConfigSubentry,
        default_model: str,
    ) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self.default_model = default_model
        self._attr_unique_id = subentry.subentry_id
        self._attr_name = subentry.title

        # Get API key from runtime data
        self._api_key = entry.runtime_data

        # Device info
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="智谱AI",
            model=subentry.data.get(CONF_CHAT_MODEL, default_model),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    def _get_model_config(self, chat_log: conversation.ChatLog | None = None) -> dict[str, Any]:
        """Get model configuration from options."""
        options = self.subentry.data
        configured_model = options.get(CONF_CHAT_MODEL, self.default_model)

        # Check if we need to switch to vision model
        final_model = configured_model
        if chat_log:
            # Detect if any content has attachments
            has_attachments = any(
                hasattr(content, 'attachments') and content.attachments
                for content in chat_log.content
            )

            # Check if attachments contain images/videos
            has_media_attachments = False
            if has_attachments:
                for content in chat_log.content:
                    if hasattr(content, 'attachments') and content.attachments:
                        for attachment in content.attachments:
                            mime_type = getattr(attachment, 'mime_type', '')
                            if mime_type.startswith(('image/', 'video/')):
                                has_media_attachments = True
                                break
                    if has_media_attachments:
                        break

            # Auto-switch to vision model if needed (prefer free model!)
            if has_media_attachments:
                vision_models = ["glm-4v-flash", "glm-4v", "glm-4v-plus"]
                if configured_model not in vision_models:
                    final_model = RECOMMENDED_IMAGE_ANALYSIS_MODEL  # GLM-4V-Flash
                    _LOGGER.info("Auto-switching to vision model %s for media attachments (original: %s)", final_model, configured_model)

        # Only use parameters that the working service uses (top_p causes API error!)
        return {
            "model": final_model,
            "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
        }

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure: dict[str, Any] | None = None,
    ) -> None:
        """Generate an answer for the chat log."""
        options = self.subentry.data
        model_config = self._get_model_config(chat_log)

        # Build messages from chat log (attachment processing will be done during conversion)
        messages = await self._async_convert_chat_log_to_messages(chat_log)

        # Add JSON format instruction to system message if structure is requested
        if structure and messages:
            for i, message in enumerate(messages):
                if message.get("role") == "system":
                    # Add JSON format requirement to system message
                    original_content = message.get("content", "")
                    if "JSON" not in original_content:
                        message["content"] = original_content + "\n\nWhen providing structured data like automation names/descriptions, respond ONLY with valid JSON. Use the exact JSON structure requested in the prompt. Do not include any markdown formatting, explanations, or additional text."
                    break

        # Add tools if available
        tools = []
        if chat_log.llm_api:
            tools.extend([
                self._format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ])

        # Add web search tool if enabled
        if options.get(CONF_WEB_SEARCH, False):
            tools.append(WEB_SEARCH_TOOL)

        # Build minimal request parameters using only essential parameters
        request_params = {
            "model": model_config.get("model"),
            "messages": messages,
            "stream": True,
        }

        if tools:
            request_params["tools"] = tools

        try:
            # Debug: Log the request parameters and validate model
            model_name = model_config.get("model", "unknown")
            _LOGGER.info("Sending request to ZhipuAI with model: %s", model_name)

            # Note: Model vision capability check is not needed here since
            # we auto-switch to vision models when media attachments are detected
            # This warning is removed to avoid confusion during model switching

            _LOGGER.info("Number of messages: %d", len(messages))
            for i, msg in enumerate(messages):
                _LOGGER.info("Message %d: role=%s, content_type=%s", i, msg.get("role"), type(msg.get("content")))
                if isinstance(msg.get("content"), list):
                    for j, part in enumerate(msg["content"]):
                        _LOGGER.info("  Part %d: type=%s", j, part.get("type"))
                        if part.get("type") == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            _LOGGER.info("    Image URL length: %d", len(url))

            # Debug: Log the full request
            _LOGGER.info("Full API request parameters: %s", json.dumps(request_params, indent=2, ensure_ascii=False))

            # Additional debugging: Check message structure specifically
            if messages:
                first_msg = messages[0]
                if isinstance(first_msg.get("content"), list):
                    _LOGGER.info("Message content structure:")
                    for i, part in enumerate(first_msg["content"]):
                        _LOGGER.info("  Part %d: %s", i, json.dumps(part, indent=2, ensure_ascii=False))
                        if part.get("type") == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            if url.startswith("data:image/"):
                                _LOGGER.info("    Image URL format looks correct (data URI)")
                            else:
                                _LOGGER.error("    Image URL format incorrect: %s", url[:100])
                else:
                    _LOGGER.info("Message content is simple string: %s", first_msg.get("content", "")[:100])

            # Call ZhipuAI API with streaming via HTTP
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    ZHIPUAI_CHAT_URL,
                    json=request_params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        _LOGGER.error("API request failed: %s", error_text)
                        raise HomeAssistantError(f"{ERROR_GETTING_RESPONSE}: {error_text}")

                    # Process streaming response using the new API
                    [
                        content
                        async for content in chat_log.async_add_delta_content_stream(
                            self.entity_id, self._transform_stream(response)
                        )
                    ]

        except aiohttp.ClientError as err:
            _LOGGER.error("Network error calling ZhipuAI API: %s", err)
            raise HomeAssistantError(f"{ERROR_GETTING_RESPONSE}: Network error") from err
        except Exception as err:
            _LOGGER.error("Error calling ZhipuAI API: %s", err)
            raise HomeAssistantError(ERROR_GETTING_RESPONSE) from err

    async def _async_convert_chat_log_to_messages(
        self, chat_log: conversation.ChatLog
    ) -> list[dict[str, Any]]:
        """Convert chat log to ZhipuAI message format."""
        options = self.subentry.data
        max_history = options.get(CONF_MAX_HISTORY_MESSAGES, RECOMMENDED_MAX_HISTORY_MESSAGES)

        messages = []

        if not chat_log.content:
            return []

        # For debugging: Check if we have attachments and simplify format
        last_content = chat_log.content[-1]
        if last_content.role == "user" and last_content.attachments:
            _LOGGER.info("Simplifying to single message format with attachments (like working service)")
            # Only send the last user message with attachments, like the working service
            return [await self._convert_user_message(last_content)]

        # Standard conversation handling for messages without attachments
        # First message is system message (index 0)
        # History is content[1:-1] (excluding first system and last user input)
        # Last message is current user input (index -1)

        # Add system messages
        for content in chat_log.content:
            if content.role == "system":
                messages.append({"role": "system", "content": content.content})

        # Process history messages (excluding system and last user input)
        history_content = chat_log.content[1:-1] if len(chat_log.content) > 1 else []

        # Build history messages
        history_messages = []
        for content in history_content:
            if content.role == "user":
                history_messages.append(await self._convert_user_message(content))
            elif content.role == "assistant":
                history_messages.append(self._convert_assistant_message(content))
            elif content.role == "tool_result":
                history_messages.append(self._convert_tool_message(content))

        # Limit history: keep only the most recent conversation turns
        # Count user messages to determine conversation turns
        if max_history > 0:
            user_message_count = sum(1 for msg in history_messages if msg.get("role") == "user")
            if user_message_count > max_history:
                # Find the index to start keeping messages
                # We want to keep the last max_history user turns and their associated messages
                user_count = 0
                start_index = len(history_messages)
                for i in range(len(history_messages) - 1, -1, -1):
                    if history_messages[i].get("role") == "user":
                        user_count += 1
                        if user_count >= max_history:
                            start_index = i
                            break
                history_messages = history_messages[start_index:]

        # Add history to messages
        messages.extend(history_messages)

        # Add current user input
        if last_content.role == "user":
            messages.append(await self._convert_user_message(last_content))
        elif last_content.role == "assistant":
            messages.append(self._convert_assistant_message(last_content))
        elif last_content.role == "tool_result":
            messages.append(self._convert_tool_message(last_content))

        return messages

    async def _convert_user_message(
        self, content: conversation.Content
    ) -> dict[str, Any]:
        """Convert user message to ZhipuAI format."""
        message: dict[str, Any] = {"role": "user"}

        # Handle text and attachments
        if content.attachments:
            parts = []
            successful_images = []
            _LOGGER.info("Processing %d attachments for user message", len(content.attachments))

            # First, process all attachments and collect successful ones
            for i, attachment in enumerate(content.attachments):
                _LOGGER.info("Processing attachment %d: %s", i, attachment)

                if attachment.mime_type and attachment.mime_type.startswith("image/"):
                    try:
                        image_data = None
                        mime_type = attachment.mime_type

                        # Handle media_content_id - try direct file path first (more reliable)
                        if hasattr(attachment, 'media_content_id'):
                            _LOGGER.info("Attachment has media_content_id: %s", attachment.media_content_id)
                            _LOGGER.info("Attachment attributes: %s", dir(attachment))
                            # First check if we have a direct file path (most reliable)
                            if hasattr(attachment, 'path') and attachment.path:
                                try:
                                    _LOGGER.info("Reading file directly from path: %s", attachment.path)
                                    import asyncio
                                    image_bytes = await asyncio.to_thread(self._read_file_bytes, str(attachment.path))
                                    image_data = base64.b64encode(image_bytes).decode()
                                    _LOGGER.info("Successfully read file directly, base64 length: %d", len(image_data))
                                except Exception as err:
                                    _LOGGER.error("Failed to read file %s: %s", attachment.path, err, exc_info=True)
                            else:
                                # Try media source resolution as fallback
                                _LOGGER.info("No file path, trying media source resolution for %s", attachment.media_content_id)
                                try:
                                    image_bytes = await self._async_get_media_content(
                                        attachment.media_content_id, mime_type
                                    )
                                    if image_bytes:
                                        image_data = base64.b64encode(image_bytes).decode()
                                        _LOGGER.info("Successfully resolved media content, base64 length: %d", len(image_data))
                                    else:
                                        _LOGGER.warning("Failed to resolve media content for: %s", attachment.media_content_id)
                                except Exception as err:
                                    _LOGGER.error("Error resolving media content %s: %s", attachment.media_content_id, err, exc_info=True)
                        # Check if attachment has resolved content
                        elif hasattr(attachment, 'content'):
                            _LOGGER.info("Attachment has resolved content")
                            # Direct content (bytes)
                            if isinstance(attachment.content, bytes):
                                image_data = base64.b64encode(attachment.content).decode()
                                _LOGGER.info("Converted bytes to base64, length: %d", len(image_data))
                            elif isinstance(attachment.content, str):
                                # Already base64 encoded
                                image_data = attachment.content
                                _LOGGER.info("Using existing base64 content, length: %d", len(image_data))
                        elif hasattr(attachment, 'path') and attachment.path:
                            # Traditional file path (handle asynchronously)
                            _LOGGER.info("Reading image from path: %s", attachment.path)
                            try:
                                import asyncio
                                image_bytes = await asyncio.to_thread(self._read_file_bytes, str(attachment.path))
                                image_data = base64.b64encode(image_bytes).decode()
                            except Exception as err:
                                _LOGGER.error("Failed to read file %s: %s", attachment.path, err, exc_info=True)
                        else:
                            _LOGGER.warning("Attachment format not supported: %s", attachment)
                            continue

                        if image_data:
                            # Use image/jpeg like official example (even for PNG images)
                            successful_images.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            })
                            _LOGGER.info("Successfully added image to message parts using official example format (image/jpeg)")
                        else:
                            _LOGGER.warning("Could not get image data from attachment: %s", attachment)

                    except Exception as err:
                        _LOGGER.error("Failed to process image attachment %s: %s", attachment, err, exc_info=True)
                else:
                    _LOGGER.info("Skipping non-image attachment: %s (mime: %s)", attachment, getattr(attachment, 'mime_type', 'unknown'))

            # Build content EXACTLY like our working services.py
            if successful_images:
                # Add images first (like working service)
                parts.extend(successful_images)
                # Add text part (like working service)
                parts.append({
                    "type": "text",
                    "text": content.content
                })

                message["content"] = parts
                _LOGGER.info("Final message content has %d parts (%d images + text) - EXACTLY like working services.py", len(parts), len(successful_images))
            else:
                # No images processed successfully, fall back to text only
                _LOGGER.warning("No images were processed successfully, falling back to text only")
                message["content"] = content.content
        else:
            message["content"] = content.content

        return message

    def _convert_assistant_message(
        self, content: conversation.Content
    ) -> dict[str, Any]:
        """Convert assistant message to ZhipuAI format."""
        message: dict[str, Any] = {"role": "assistant"}

        if content.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.tool_name,
                        "arguments": json.dumps(tool_call.tool_args, ensure_ascii=False),
                    },
                }
                for tool_call in content.tool_calls
            ]
            message["content"] = content.content or ""
        else:
            message["content"] = content.content

        return message

    def _convert_tool_message(
        self, content: conversation.Content
    ) -> dict[str, Any]:
        """Convert tool result to ZhipuAI format."""
        return {
            "role": "tool",
            "tool_call_id": content.tool_call_id,
            "content": json.dumps(content.tool_result, ensure_ascii=False),
        }

    def _format_tool(
        self, tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
    ) -> dict[str, Any]:
        """Format tool for ZhipuAI API."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": self._convert_schema(tool.parameters, custom_serializer),
            },
        }

    def _convert_schema(
        self, schema: dict[str, Any], custom_serializer: Callable[[Any], Any] | None
    ) -> dict[str, Any]:
        """Convert schema to ZhipuAI format."""
        # ZhipuAI uses standard JSON Schema
        # Use voluptuous_openapi to convert the schema properly
        try:
            return convert(
                schema,
                custom_serializer=custom_serializer if custom_serializer else llm.selector_serializer,
            )
        except Exception as err:
            _LOGGER.warning("Failed to convert schema with custom_serializer: %s", err)
            # Fall back to basic conversion without custom_serializer
            try:
                return convert(schema, custom_serializer=llm.selector_serializer)
            except Exception:
                # If all else fails, return as-is
                return schema

    async def _transform_stream(
        self,
        response: aiohttp.ClientResponse,
    ) -> AsyncGenerator[
        conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
    ]:
        """Transform ZhipuAI SSE stream into HA format."""
        buffer = ""
        tool_call_buffer: dict[int, dict[str, Any]] = {}
        has_started = False

        async for chunk in response.content:
            if not chunk:
                continue

            chunk_text = chunk.decode("utf-8", errors="ignore")
            buffer += chunk_text

            # Process complete lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                # Skip empty lines and end markers
                if not line or line == "data: [DONE]":
                    continue

                # Process SSE data lines
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    if not data_str.strip():
                        continue

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        _LOGGER.debug("SSE data parse failed: %s", data_str)
                        continue

                    if not data.get("choices"):
                        continue

                    delta = data["choices"][0].get("delta", {})

                    # Start assistant message if not started
                    if not has_started:
                        yield {"role": "assistant"}
                        has_started = True

                    # Handle content delta
                    if "content" in delta and delta["content"]:
                        # Filter markdown from content using streaming filter to preserve spaces
                        filtered_content = filter_markdown_streaming(delta["content"])
                        yield {"content": filtered_content}

                    # Handle tool calls
                    if "tool_calls" in delta:
                        for tc_delta in delta["tool_calls"]:
                            index = tc_delta.get("index", 0)

                            # Initialize tool call buffer if needed
                            if index not in tool_call_buffer:
                                tool_call_buffer[index] = {
                                    "id": tc_delta.get("id", ulid.ulid_now()),
                                    "type": "function",
                                    "function": {
                                        "name": "",
                                        "arguments": "",
                                    },
                                }

                            # Update tool call data
                            if "id" in tc_delta:
                                tool_call_buffer[index]["id"] = tc_delta["id"]
                            if "function" in tc_delta:
                                func = tc_delta["function"]
                                if "name" in func:
                                    tool_call_buffer[index]["function"]["name"] = func["name"]
                                if "arguments" in func:
                                    tool_call_buffer[index]["function"]["arguments"] += func["arguments"]

        # Yield final tool calls if any
        if tool_call_buffer:
            tool_calls = []
            for tc in tool_call_buffer.values():
                try:
                    args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                    tool_calls.append(
                        llm.ToolInput(
                            id=tc["id"],
                            tool_name=tc["function"]["name"],
                            tool_args=args,
                        )
                    )
                except json.JSONDecodeError as err:
                    _LOGGER.warning("Failed to parse tool call arguments: %s", err)

            if tool_calls:
                yield {"tool_calls": tool_calls}

    async def _async_iterate_response(self, response: Any):
        """Iterate over streaming response asynchronously."""
        # This method is no longer needed with HTTP streaming
        for chunk in response:
            yield chunk

    async def _async_download_image_from_url(self, url: str) -> bytes | None:
        """Download image from URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        _LOGGER.warning("Failed to download image from URL: %s, status: %s", url, response.status)
                        return None
        except Exception as err:
            _LOGGER.warning("Error downloading image from URL %s: %s", url, err)
            return None

    async def _async_get_media_content(self, media_content_id: str, mime_type: str) -> bytes | None:
        """Get media content from Home Assistant media source."""
        _LOGGER.info("Getting media content for ID: %s, mime_type: %s", media_content_id, mime_type)

        try:
            # Handle media-source:// URLs
            if media_content_id.startswith("media-source://"):
                _LOGGER.info("Processing media-source URL: %s", media_content_id)

                if not media_source.is_media_source_id(media_content_id):
                    _LOGGER.warning("Invalid media source ID: %s", media_content_id)
                    return None

                # Resolve media source
                try:
                    media_item = await media_source.async_resolve_media(
                        self.hass, media_content_id, self.entity_id
                    )
                    _LOGGER.info("Resolved media item: %s", media_item)
                except Exception as err:
                    _LOGGER.error("Error resolving media source %s: %s", media_content_id, err, exc_info=True)
                    return None

                if media_item and hasattr(media_item, 'url') and media_item.url:
                    # Build full URL if it's a relative path
                    media_url = media_item.url
                    if media_url.startswith('/'):
                        # Get Home Assistant base URL from configuration
                        try:
                            if hasattr(self.hass.config, 'external_url') and self.hass.config.external_url:
                                base_url = self.hass.config.external_url.rstrip('/')
                                media_url = f"{base_url}{media_url}"
                            elif hasattr(self.hass.config, 'internal_url') and self.hass.config.internal_url:
                                base_url = self.hass.config.internal_url.rstrip('/')
                                media_url = f"{base_url}{media_url}"
                            else:
                                # Default to localhost
                                media_url = f"http://localhost:8123{media_url}"
                        except Exception as err:
                            # Fallback to localhost
                            _LOGGER.warning("Could not get Home Assistant URL, using localhost: %s", err)
                            media_url = f"http://localhost:8123{media_url}"

                    _LOGGER.info("Media item URL: %s", media_url)

                    # Download the media content using Home Assistant's session
                    try:
                        from homeassistant.helpers.aiohttp_client import async_get_clientsession
                        session = async_get_clientsession(self.hass)
                        async with session.get(
                            media_url,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            _LOGGER.info("Response status: %s", response.status)
                            if response.status == 200:
                                content = await response.read()
                                _LOGGER.info("Successfully downloaded media content, size: %d bytes", len(content))
                                return content
                            else:
                                error_text = await response.text()
                                _LOGGER.warning(
                                    "Failed to download media content: %s, status: %s, error: %s",
                                    media_item.url,
                                    response.status,
                                    error_text
                                )
                                return None
                    except Exception as err:
                        _LOGGER.error("Error downloading media content from %s: %s", media_item.url, err, exc_info=True)
                        return None
                else:
                    _LOGGER.warning("Could not resolve media source or no URL: %s", media_content_id)
                    _LOGGER.warning("Media item details: %s", media_item)
                    return None

            # Handle /api/image/serve URLs (Home Assistant image serving)
            elif media_content_id.startswith("/api/image/serve/") or media_content_id.startswith("api/image/serve/"):
                # Construct full URL if needed
                if not media_content_id.startswith("/"):
                    url = f"/{media_content_id}"
                else:
                    url = media_content_id

                _LOGGER.info("Processing serve URL: %s", url)

                # Use Home Assistant's internal API client
                try:
                    from homeassistant.helpers.aiohttp_client import async_get_clientsession
                    session = async_get_clientsession(self.hass)
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            content = await response.read()
                            _LOGGER.info("Successfully got served image, size: %d bytes", len(content))
                            return content
                        else:
                            _LOGGER.warning("Failed to get served image: %s, status: %s", url, response.status)
                            return None
                except Exception as err:
                    _LOGGER.error("Error getting served image %s: %s", url, err, exc_info=True)
                    return None

            # Handle direct URLs
            elif media_content_id.startswith(("http://", "https://")):
                _LOGGER.info("Processing direct URL: %s", media_content_id)
                return await self._async_download_image_from_url(media_content_id)

            else:
                _LOGGER.warning("Unsupported media content ID format: %s", media_content_id)
                return None

        except Exception as err:
            _LOGGER.error("Unexpected error getting media content %s: %s", media_content_id, err, exc_info=True)
            return None

    def _read_file_bytes(self, file_path: str) -> bytes:
        """Read file bytes synchronously (to be used with asyncio.to_thread)."""
        try:
            with open(file_path, "rb") as f:
                return f.read()
        except Exception as err:
            raise Exception(f"Failed to read file {file_path}: {err}")


class ZhipuAIEntityBase(Entity):
    """Base entity for 智谱清言 integration."""

    _attr_has_entity_name = False
    _attr_should_poll = False

    def __init__(
        self,
        entry: config_entry_flow.ConfigEntry,
        subentry: config_entry_flow.ConfigSubentry,
        default_model: str,
    ) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self.default_model = default_model
        self._attr_unique_id = subentry.subentry_id
        self._attr_name = subentry.title

        # Get API key from runtime data
        self._api_key = entry.runtime_data

        # Device info
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="智谱AI",
            model=subentry.data.get(CONF_CHAT_MODEL, default_model),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    
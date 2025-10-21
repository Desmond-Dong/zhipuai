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

from homeassistant.components import conversation
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
    DOMAIN,
    ERROR_GETTING_RESPONSE,
    RECOMMENDED_MAX_HISTORY_MESSAGES,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_K,
    RECOMMENDED_TOP_P,
    ZHIPUAI_CHAT_URL,
)

_LOGGER = logging.getLogger(__name__)


class ZhipuAIBaseLLMEntity(Entity):
    """Base entity for 智谱清言 LLM."""

    _attr_has_entity_name = True
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

    def _get_model_config(self) -> dict[str, Any]:
        """Get model configuration from options."""
        options = self.subentry.data
        return {
            "model": options.get(CONF_CHAT_MODEL, self.default_model),
            "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
        }

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure: dict[str, Any] | None = None,
    ) -> None:
        """Generate an answer for the chat log."""
        options = self.subentry.data
        model_config = self._get_model_config()

        # Build messages from chat log
        messages = self._convert_chat_log_to_messages(chat_log)

        # Add tools if available
        tools = None
        if chat_log.llm_api:
            tools = [
                self._format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        # Build request parameters
        request_params = {
            **model_config,
            "messages": messages,
            "stream": True,
        }

        if tools:
            request_params["tools"] = tools

        try:
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

    def _convert_chat_log_to_messages(
        self, chat_log: conversation.ChatLog
    ) -> list[dict[str, Any]]:
        """Convert chat log to ZhipuAI message format."""
        options = self.subentry.data
        max_history = options.get(CONF_MAX_HISTORY_MESSAGES, RECOMMENDED_MAX_HISTORY_MESSAGES)

        messages = []

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
                history_messages.append(self._convert_user_message(content))
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
        if chat_log.content:
            last_content = chat_log.content[-1]
            if last_content.role == "user":
                messages.append(self._convert_user_message(last_content))
            elif last_content.role == "assistant":
                messages.append(self._convert_assistant_message(last_content))
            elif last_content.role == "tool_result":
                messages.append(self._convert_tool_message(last_content))

        return messages

    def _convert_user_message(
        self, content: conversation.Content
    ) -> dict[str, Any]:
        """Convert user message to ZhipuAI format."""
        message: dict[str, Any] = {"role": "user"}

        # Handle text and attachments
        if content.attachments:
            parts = [{"type": "text", "text": content.content}]
            for attachment in content.attachments:
                if attachment.mime_type and attachment.mime_type.startswith("image/"):
                    # Add image as base64
                    try:
                        with open(attachment.path, "rb") as img_file:
                            img_data = base64.b64encode(img_file.read()).decode()
                            parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{attachment.mime_type};base64,{img_data}"
                                }
                            })
                    except Exception as err:
                        _LOGGER.warning("Failed to load image %s: %s", attachment.path, err)
            message["content"] = parts
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
                        yield {"content": delta["content"]}

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

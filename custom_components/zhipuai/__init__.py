"""The 智谱清言 integration."""

from __future__ import annotations

import logging

import aiohttp

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.config_entries import ConfigSubentry
from homeassistant.helpers import config_entry_flow
from types import MappingProxyType

from .const import (
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_TTS_NAME,
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CONVERSATION_OPTIONS,
    RECOMMENDED_TTS_OPTIONS,
    ZHIPUAI_CHAT_URL,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.CONVERSATION, Platform.AI_TASK, Platform.TTS]

type ZhipuAIConfigEntry = ConfigEntry[str]  # Store API key


async def async_setup_entry(hass: HomeAssistant, entry: ZhipuAIConfigEntry) -> bool:
    """Set up 智谱清言 from a config entry."""

    # Validate API key by testing API connection
    api_key = entry.data[CONF_API_KEY]

    try:
        # Test the connection with a simple API call
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "GLM-4-Flash",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                ZHIPUAI_CHAT_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 401:
                    raise ConfigEntryAuthFailed("Invalid API key")
                if response.status != 200:
                    error_text = await response.text()
                    raise ConfigEntryNotReady(f"API test failed: {error_text}")
    except aiohttp.ClientError as err:
        raise ConfigEntryNotReady(f"Failed to connect: {err}") from err
    except ConfigEntryAuthFailed:
        raise
    except Exception as err:
        raise ConfigEntryNotReady(f"Unexpected error: {err}") from err

    # Store API key in runtime data
    entry.runtime_data = api_key

    # Add TTS subentry if it doesn't exist
    if not any(subentry.subentry_type == "tts" for subentry in entry.subentries.values()):
        hass.config_entries.async_add_subentry(
            entry,
            ConfigSubentry(
                data=MappingProxyType(RECOMMENDED_TTS_OPTIONS),
                subentry_type="tts",
                title=DEFAULT_TTS_NAME,
                unique_id=None,
            ),
        )

    # Forward setup to platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Set up intent handlers
    from .intents import async_setup_intents
    await async_setup_intents(hass)

    # Set up AI automation services
    from .ai_automation import async_setup_ai_automation
    await async_setup_ai_automation(hass)

    # Set up services
    from .services import async_setup_services
    await async_setup_services(hass, entry)

    # Listen for options updates
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ZhipuAIConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    _LOGGER.debug("Migrating configuration from version %s.%s", entry.version, entry.minor_version)

    if entry.version == 1:
        # Migrate from version 1 to version 2
        # Version 2 uses subentries for conversation and ai_task
        new_data = {**entry.data}
        new_options = {**entry.options}

        # Create default subentries
        from .const import (
            CONF_CHAT_MODEL,
            CONF_LLM_HASS_API,
            CONF_MAX_TOKENS,
            CONF_PROMPT,
            CONF_RECOMMENDED,
            CONF_TEMPERATURE,
            CONF_TOP_P,
            DEFAULT_AI_TASK_NAME,
            DEFAULT_CONVERSATION_NAME,
            RECOMMENDED_AI_TASK_MAX_TOKENS,
            RECOMMENDED_AI_TASK_MODEL,
            RECOMMENDED_AI_TASK_TEMPERATURE,
            RECOMMENDED_AI_TASK_TOP_P,
            RECOMMENDED_CHAT_MODEL,
            RECOMMENDED_MAX_TOKENS,
            RECOMMENDED_TEMPERATURE,
            RECOMMENDED_TOP_P,
        )
        from homeassistant.helpers import llm

        # Create conversation subentry from old options
        conversation_data = {
            CONF_RECOMMENDED: new_options.get(CONF_RECOMMENDED, True),
            CONF_CHAT_MODEL: new_options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            CONF_TEMPERATURE: new_options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            CONF_TOP_P: new_options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            CONF_MAX_TOKENS: new_options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
            CONF_PROMPT: new_options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT),
            CONF_LLM_HASS_API: new_options.get(CONF_LLM_HASS_API, [llm.LLM_API_ASSIST]),
        }

        # Create AI task subentry with defaults
        ai_task_data = {
            CONF_RECOMMENDED: True,
            CONF_CHAT_MODEL: RECOMMENDED_AI_TASK_MODEL,
            CONF_TEMPERATURE: RECOMMENDED_AI_TASK_TEMPERATURE,
            CONF_TOP_P: RECOMMENDED_AI_TASK_TOP_P,
            CONF_MAX_TOKENS: RECOMMENDED_AI_TASK_MAX_TOKENS,
        }

        hass.config_entries.async_update_entry(
            entry,
            data=new_data,
            options={},
            version=2,
            minor_version=2,
        )

        # Add subentries
        conversation_subentry = config_entry_flow.ConfigSubentry(
            data=conversation_data,
            subentry_type="conversation",
            title=DEFAULT_CONVERSATION_NAME,
            unique_id=None,
        )
        hass.config_entries.async_add_subentry(entry, conversation_subentry)

        ai_task_subentry = config_entry_flow.ConfigSubentry(
            data=ai_task_data,
            subentry_type="ai_task_data",
            title=DEFAULT_AI_TASK_NAME,
            unique_id=None,
        )
        hass.config_entries.async_add_subentry(entry, ai_task_subentry)

        _LOGGER.debug("Migration to version %s.%s successful", entry.version, entry.minor_version)

    if entry.version == 2 and entry.minor_version == 1:
        # Migrate from version 2.1 to 2.2
        # Update subentry titles to include "智谱"
        from .const import DEFAULT_AI_TASK_NAME, DEFAULT_CONVERSATION_NAME

        for subentry in entry.subentries.values():
            # Update old titles to new format
            if subentry.subentry_type == "conversation":
                if subentry.title in ("对话助手", "Conversation Agent"):
                    hass.config_entries.async_update_subentry(
                        entry, subentry.subentry_id, title=DEFAULT_CONVERSATION_NAME
                    )
            elif subentry.subentry_type == "ai_task_data":
                if subentry.title in ("AI任务", "AI Task"):
                    hass.config_entries.async_update_subentry(
                        entry, subentry.subentry_id, title=DEFAULT_AI_TASK_NAME
                    )

        hass.config_entries.async_update_entry(
            entry,
            minor_version=2,
        )

        _LOGGER.debug("Migration to version %s.%s successful", entry.version, entry.minor_version)

    return True

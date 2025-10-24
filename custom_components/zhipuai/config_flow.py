"""Config flow for 智谱清言 integration."""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    OptionsFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm, selector
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)

from .const import (
    CONF_CHAT_MODEL,
    CONF_IMAGE_MODEL,
    CONF_LLM_HASS_API,
    CONF_MAX_HISTORY_MESSAGES,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    CONF_WEB_SEARCH,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_TITLE,
    DOMAIN,
    RECOMMENDED_AI_TASK_MAX_TOKENS,
    RECOMMENDED_AI_TASK_MODEL,
    RECOMMENDED_AI_TASK_TEMPERATURE,
    RECOMMENDED_AI_TASK_TOP_P,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_IMAGE_MODEL,
    RECOMMENDED_MAX_HISTORY_MESSAGES,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_K,
    RECOMMENDED_TOP_P,
    ZHIPUAI_CHAT_MODELS,
    ZHIPUAI_CHAT_URL,
    ZHIPUAI_IMAGE_MODELS,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema({vol.Required(CONF_API_KEY): str})

# Recommended options for conversation
RECOMMENDED_CONVERSATION_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
    CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
    CONF_TOP_P: RECOMMENDED_TOP_P,
    CONF_TOP_K: RECOMMENDED_TOP_K,
    CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
    CONF_MAX_HISTORY_MESSAGES: RECOMMENDED_MAX_HISTORY_MESSAGES,
    CONF_WEB_SEARCH: False,
}

# Recommended options for AI task
RECOMMENDED_AI_TASK_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_CHAT_MODEL: RECOMMENDED_AI_TASK_MODEL,
    CONF_TEMPERATURE: RECOMMENDED_AI_TASK_TEMPERATURE,
    CONF_TOP_P: RECOMMENDED_AI_TASK_TOP_P,
    CONF_MAX_TOKENS: RECOMMENDED_AI_TASK_MAX_TOKENS,
    CONF_IMAGE_MODEL: RECOMMENDED_IMAGE_MODEL,
}


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect."""
    headers = {
        "Authorization": f"Bearer {data[CONF_API_KEY]}",
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
                raise ValueError("Invalid API key")
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API test failed: {error_text}")


class ZhipuAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for 智谱清言."""

    VERSION = 2
    MINOR_VERSION = 2

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=STEP_USER_DATA_SCHEMA,
                description_placeholders={
                    "api_key_url": "https://open.bigmodel.cn/usercenter/apikeys"
                },
            )

        errors = {}

        try:
            await validate_input(self.hass, user_input)
        except ValueError:
            _LOGGER.exception("Invalid API key")
            errors["base"] = "invalid_auth"
        except aiohttp.ClientError:
            _LOGGER.exception("Cannot connect")
            errors["base"] = "cannot_connect"
        except Exception:
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            # Create entry with two subentries
            return self.async_create_entry(
                title=DEFAULT_TITLE,
                data=user_input,
                subentries=[
                    {
                        "subentry_type": "conversation",
                        "data": RECOMMENDED_CONVERSATION_OPTIONS,
                        "title": DEFAULT_CONVERSATION_NAME,
                        "unique_id": None,
                    },
                    {
                        "subentry_type": "ai_task_data",
                        "data": RECOMMENDED_AI_TASK_OPTIONS,
                        "title": DEFAULT_AI_TASK_NAME,
                        "unique_id": None,
                    },
                ],
            )

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
            description_placeholders={
                "api_key_url": "https://open.bigmodel.cn/usercenter/apikeys"
            },
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return ZhipuAIOptionsFlow()

    @classmethod
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            "conversation": ZhipuAISubentryFlowHandler,
            "ai_task_data": ZhipuAISubentryFlowHandler,
        }


class ZhipuAIOptionsFlow(OptionsFlow):
    """智谱清言 config flow options handler."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        # For parent entry, options are managed through subentry configuration
        # Show information form instead of aborting
        if user_input is not None:
            return self.async_abort(reason="configure_via_subentries")

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({}),
            description_placeholders={
                "info": "请在集成详情页面中找到对应的子条目（对话助手或AI任务），点击子条目的配置按钮来修改设置。"
            },
        )


class ZhipuAISubentryFlowHandler(ConfigSubentryFlow):
    """Handle subentry flow for conversation and AI task."""

    options: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle subentry setup."""
        if self._subentry_type == "ai_task_data":
            self.options = RECOMMENDED_AI_TASK_OPTIONS.copy()
        else:
            self.options = RECOMMENDED_CONVERSATION_OPTIONS.copy()
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle subentry reconfiguration."""
        self.options = self._get_reconfigure_subentry().data.copy()
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle options for subentry."""
        errors: dict[str, str] = {}

        # Get options
        options = self.options
        # Get default name
        if self._is_new:
            if self._subentry_type == "ai_task_data":
                default_name = DEFAULT_AI_TASK_NAME
            else:
                default_name = DEFAULT_CONVERSATION_NAME
        else:
            default_name = self._get_reconfigure_subentry().title
        # If user_input exists, detect interactive changes (e.g., toggling recommended)        if user_input is not None:            # If user only toggled the recommended field, update options and re-render form dynamically            # This allows the frontend to immediately show/hide advanced options without completing the flow            if set(user_input.keys()) == {CONF_RECOMMENDED}:                options[CONF_RECOMMENDED] = user_input[CONF_RECOMMENDED]                # rebuild schema with updated recommended flag and show form                schema = await self._build_schema(options, default_name)                return self.async_show_form(                    step_id="init",                    data_schema=vol.Schema(schema),                    errors=errors,                )            # For conversation, toggling prompt alone should just re-render as well            if self._subentry_type == "conversation" and set(user_input.keys()) == {CONF_PROMPT, CONF_RECOMMENDED}:                options[CONF_RECOMMENDED] = user_input.get(CONF_RECOMMENDED, options.get(CONF_RECOMMENDED))                options[CONF_PROMPT] = user_input.get(CONF_PROMPT, options.get(CONF_PROMPT))                schema = await self._build_schema(options, default_name)                return self.async_show_form(                    step_id="init",                    data_schema=vol.Schema(schema),                    errors=errors,                )            # Otherwise, this is a real submit: set LLM_HASS_API for conversation and create/update entry            if self._subentry_type == "conversation":                user_input[CONF_LLM_HASS_API] = llm.LLM_API_ASSIST            # Update or create subentry            if self._is_new:                return self.async_create_entry(                    title=user_input.pop(CONF_NAME),                    data=user_input,                )            return self.async_update_and_abort(                self._get_entry(),                self._get_reconfigure_subentry(),                data=user_input,            )        # Build schema based on subentry type and recommended mode        schema = await self._build_schema(options, default_name)        return self.async_show_form(            step_id="init",            data_schema=vol.Schema(schema),            errors=errors,        )

    async def _build_schema(        self, options: dict[str, Any], default_name: str    ) -> dict:        """Build configuration schema."""        schema: dict[vol.Required | vol.Optional, Any] = {}        # Add name field for new entries        if self._is_new:            schema[vol.Required(CONF_NAME, default=default_name)] = str        # Add recommended mode toggle        schema[            vol.Required(CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, True))        ] = bool        recommended = options.get(CONF_RECOMMENDED, True)        # Conversation-specific options        if self._subentry_type == "conversation":            schema.update({                vol.Optional(                    CONF_PROMPT,                    description={"suggested_value": options.get(CONF_PROMPT)},                ): TemplateSelector(),            })        # Show advanced options only in non-recommended mode        if not recommended:            if self._subentry_type == "conversation":                schema.update(self._get_conversation_advanced_schema(options))            else:                schema.update(self._get_ai_task_advanced_schema(options))        return schema    def _get_conversation_advanced_schema(        self, options: dict[str, Any]    ) -> dict:        """Get advanced schema for conversation."""        return {            vol.Optional(                CONF_CHAT_MODEL,                default=options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),            ): SelectSelector(                SelectSelectorConfig(                    options=ZHIPUAI_CHAT_MODELS,                    mode=SelectSelectorMode.DROPDOWN,                )            ),            vol.Optional(                CONF_TEMPERATURE,                default=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),            ): NumberSelector(                NumberSelectorConfig(                    min=0, max=2, step=0.01, mode=NumberSelectorMode.SLIDER                )            ),            vol.Optional(                CONF_TOP_P,                default=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),            ): NumberSelector(                NumberSelectorConfig(                    min=0, max=1, step=0.01, mode=NumberSelectorMode.SLIDER                )            ),            vol.Optional(                CONF_TOP_K,                default=options.get(CONF_TOP_K, RECOMMENDED_TOP_K),            ): int,            vol.Optional(                CONF_MAX_TOKENS,                default=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),            ): int,            vol.Optional(                CONF_MAX_HISTORY_MESSAGES,                default=options.get(CONF_MAX_HISTORY_MESSAGES, RECOMMENDED_MAX_HISTORY_MESSAGES),            ): int,            vol.Optional(                CONF_WEB_SEARCH,                default=options.get(CONF_WEB_SEARCH, True),            ): bool,        }    def _get_ai_task_advanced_schema(        self, options: dict[str, Any]    ) -> dict:        """Get advanced schema for AI task."""        return {            vol.Optional(                CONF_CHAT_MODEL,                default=options.get(CONF_CHAT_MODEL, RECOMMENDED_AI_TASK_MODEL),            ): SelectSelector(                SelectSelectorConfig(                    options=ZHIPUAI_CHAT_MODELS,                    mode=SelectSelectorMode.DROPDOWN,                )            ),            vol.Optional(                CONF_IMAGE_MODEL,                default=options.get(CONF_IMAGE_MODEL, RECOMMENDED_IMAGE_MODEL),            ): SelectSelector(                SelectSelectorConfig(                    options=ZHIPUAI_IMAGE_MODELS,                    mode=SelectSelectorMode.DROPDOWN,                )            ),            vol.Optional(                CONF_TEMPERATURE,                default=options.get(CONF_TEMPERATURE, RECOMMENDED_AI_TASK_TEMPERATURE),            ): NumberSelector(                NumberSelectorConfig(                    min=0, max=2, step=0.01, mode=NumberSelectorMode.SLIDER                )            ),            vol.Optional(                CONF_TOP_P,                default=options.get(CONF_TOP_P, RECOMMENDED_AI_TASK_TOP_P),            ): NumberSelector(                NumberSelectorConfig(                    min=0, max=1, step=0.01, mode=NumberSelectorMode.SLIDER                )            ),            vol.Optional(                CONF_MAX_TOKENS,                default=options.get(CONF_MAX_TOKENS, RECOMMENDED_AI_TASK_MAX_TOKENS),            ): int,        }
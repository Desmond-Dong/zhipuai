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
    CONF_TTS_VOICE,
    CONF_TTS_SPEED,
    CONF_TTS_VOLUME,
    CONF_TTS_RESPONSE_FORMAT,
    CONF_TTS_ENCODE_FORMAT,
    CONF_TTS_STREAM,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_TITLE,
    DEFAULT_TTS_NAME,
    DOMAIN,
    RECOMMENDED_AI_TASK_MAX_TOKENS,
    RECOMMENDED_AI_TASK_MODEL,
    RECOMMENDED_AI_TASK_TEMPERATURE,
    RECOMMENDED_AI_TASK_TOP_P,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_IMAGE_ANALYSIS_MODEL,
    RECOMMENDED_IMAGE_MODEL,
    RECOMMENDED_MAX_HISTORY_MESSAGES,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_K,
    RECOMMENDED_TOP_P,
    TTS_DEFAULT_ENCODE_FORMAT,
    TTS_DEFAULT_RESPONSE_FORMAT,
    TTS_DEFAULT_SPEED,
    TTS_DEFAULT_STREAM,
    TTS_DEFAULT_VOICE,
    TTS_DEFAULT_VOLUME,
    TTS_SPEED_MAX,
    TTS_SPEED_MIN,
    TTS_SPEED_STEP,
    TTS_VOLUME_MAX,
    TTS_VOLUME_MIN,
    TTS_VOLUME_STEP,
    ZHIPUAI_CHAT_MODELS,
    ZHIPUAI_CHAT_URL,
    ZHIPUAI_IMAGE_MODELS,
    ZHIPUAI_TTS_ENCODE_FORMATS,
    ZHIPUAI_TTS_RESPONSE_FORMATS,
    ZHIPUAI_TTS_VOICES,
    RECOMMENDED_CONVERSATION_OPTIONS,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_STT_OPTIONS,
    RECOMMENDED_TTS_OPTIONS,
    RECOMMENDED_TTS_MODEL,
    RECOMMENDED_STT_MODEL,
    DEFAULT_STT_NAME,
    CONF_STT_MODEL,
    CONF_STT_TEMPERATURE,
    CONF_STT_LANGUAGE,
    CONF_STT_STREAM,
    STT_DEFAULT_TEMPERATURE,
    STT_DEFAULT_STREAM,
    STT_TEMPERATURE_MAX,
    STT_TEMPERATURE_MIN,
    STT_TEMPERATURE_STEP,
    ZHIPUAI_STT_MODELS,
    )

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema({vol.Required(CONF_API_KEY): str})

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
                    {
                        "subentry_type": "tts",
                        "data": RECOMMENDED_TTS_OPTIONS,
                        "title": DEFAULT_TTS_NAME,
                        "unique_id": None,
                    },
                    {
                        "subentry_type": "stt",
                        "data": RECOMMENDED_STT_OPTIONS,
                        "title": DEFAULT_STT_NAME,
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


    @classmethod
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            "conversation": ZhipuAISubentryFlowHandler,
            "ai_task_data": ZhipuAISubentryFlowHandler,
            "tts": ZhipuAISubentryFlowHandler,
            "stt": ZhipuAISubentryFlowHandler,
        }




class ZhipuAISubentryFlowHandler(ConfigSubentryFlow):
    """Handle subentry flow for conversation and AI task."""

    options: dict[str, Any]
    last_rendered_recommended: bool = False

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == "user"

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle options for subentry."""
        errors: dict[str, str] = {}

        if user_input is None:
            # First render: get current options
            if self._is_new:
                if self._subentry_type == "ai_task_data":
                    self.options = RECOMMENDED_AI_TASK_OPTIONS.copy()
                elif self._subentry_type == "tts":
                    self.options = RECOMMENDED_TTS_OPTIONS.copy()
                elif self._subentry_type == "stt":
                    self.options = RECOMMENDED_STT_OPTIONS.copy()
                else:
                    self.options = RECOMMENDED_CONVERSATION_OPTIONS.copy()
            else:
                # If reconfiguration, copy existing options to show current values
                self.options = self._get_reconfigure_subentry().data.copy()

            self.last_rendered_recommended = self.options.get(CONF_RECOMMENDED, True)

        else:
            # Check if recommended mode has changed
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                # Recommended mode unchanged, save the configuration
                # Set LLM_HASS_API for conversation
                if self._subentry_type == "conversation":
                    user_input[CONF_LLM_HASS_API] = llm.LLM_API_ASSIST

                # Update or create subentry
                if self._is_new:
                    return self.async_create_entry(
                        title=user_input.pop(CONF_NAME),
                        data=user_input,
                    )

                return self.async_update_and_abort(
                    self._get_entry(),
                    self._get_reconfigure_subentry(),
                    data=user_input,
                )

            # Recommended mode changed, re-render form with new options shown/hidden
            self.last_rendered_recommended = user_input[CONF_RECOMMENDED]
            self.options.update(user_input)  # Update current options with user input

        # Build schema based on current options
        schema = await zhipuai_config_option_schema(
            self._is_new, self._subentry_type, self.options
        )
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
            errors=errors,
        )

    async_step_reconfigure = async_step_init
    async_step_user = async_step_init


async def zhipuai_config_option_schema(
    is_new: bool,
    subentry_type: str,
    options: Mapping[str, Any],
) -> dict:
    """Return a schema for ZhipuAI completion options."""
    # 确保所有需要的常量都在作用域内
    global RECOMMENDED_TTS_MODEL, RECOMMENDED_TTS_OPTIONS, RECOMMENDED_STT_MODEL, RECOMMENDED_STT_OPTIONS, TTS_DEFAULT_VOICE, STT_DEFAULT_TEMPERATURE

    schema = {}

    # Add name field for new entries
    if is_new:
        if CONF_NAME in options:
            default_name = options[CONF_NAME]
        elif subentry_type == "ai_task_data":
            default_name = DEFAULT_AI_TASK_NAME
        elif subentry_type == "tts":
            default_name = DEFAULT_TTS_NAME
        elif subentry_type == "stt":
            default_name = DEFAULT_STT_NAME
        else:
            default_name = DEFAULT_CONVERSATION_NAME
        schema[vol.Required(CONF_NAME, default=default_name)] = str

    # Add recommended mode toggle
    schema[
        vol.Required(CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, True))
    ] = bool

    # If recommended mode is enabled, only show basic fields
    if options.get(CONF_RECOMMENDED):
        # In recommended mode, only show prompt for conversation
        if subentry_type == "conversation":
            schema.update({
                vol.Optional(
                    CONF_PROMPT,
                    default=options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT),
                    description={"suggested_value": options.get(CONF_PROMPT)},
                ): TemplateSelector(),
            })
        elif subentry_type == "tts":
            # In recommended mode, no configuration options needed
            pass
        elif subentry_type == "stt":
            # In recommended mode, no configuration options needed
            pass
        return schema

    # Show advanced options only when not in recommended mode
    if subentry_type == "conversation":
        schema.update({
            vol.Optional(
                CONF_PROMPT,
                default=options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT),
                description={"suggested_value": options.get(CONF_PROMPT)},
            ): TemplateSelector(),
            vol.Optional(
                CONF_CHAT_MODEL,
                default=options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                description={"suggested_value": options.get(CONF_CHAT_MODEL)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=ZHIPUAI_CHAT_MODELS,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_TEMPERATURE,
                default=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=2, step=0.01, mode=NumberSelectorMode.SLIDER
                )
            ),
            vol.Optional(
                CONF_TOP_P,
                default=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                description={"suggested_value": options.get(CONF_TOP_P)},
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=1, step=0.01, mode=NumberSelectorMode.SLIDER
                )
            ),
            vol.Optional(
                CONF_TOP_K,
                default=options.get(CONF_TOP_K, RECOMMENDED_TOP_K),
                description={"suggested_value": options.get(CONF_TOP_K)},
            ): int,
            vol.Optional(
                CONF_MAX_TOKENS,
                default=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
            ): int,
            vol.Optional(
                CONF_MAX_HISTORY_MESSAGES,
                default=options.get(CONF_MAX_HISTORY_MESSAGES, RECOMMENDED_MAX_HISTORY_MESSAGES),
                description={"suggested_value": options.get(CONF_MAX_HISTORY_MESSAGES)},
            ): int,
            vol.Optional(
                CONF_WEB_SEARCH,
                default=options.get(CONF_WEB_SEARCH, False),
                description={"suggested_value": options.get(CONF_WEB_SEARCH)},
            ): bool,
        })

    elif subentry_type == "ai_task_data":
        schema.update({
            vol.Optional(
                CONF_CHAT_MODEL,
                default=options.get(CONF_CHAT_MODEL, RECOMMENDED_AI_TASK_MODEL),
                description={"suggested_value": options.get(CONF_CHAT_MODEL)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=ZHIPUAI_CHAT_MODELS,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_IMAGE_MODEL,
                default=options.get(CONF_IMAGE_MODEL, RECOMMENDED_IMAGE_MODEL),
                description={"suggested_value": options.get(CONF_IMAGE_MODEL)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=ZHIPUAI_IMAGE_MODELS,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_TEMPERATURE,
                default=options.get(CONF_TEMPERATURE, RECOMMENDED_AI_TASK_TEMPERATURE),
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=2, step=0.01, mode=NumberSelectorMode.SLIDER
                )
            ),
            vol.Optional(
                CONF_TOP_P,
                default=options.get(CONF_TOP_P, RECOMMENDED_AI_TASK_TOP_P),
                description={"suggested_value": options.get(CONF_TOP_P)},
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=1, step=0.01, mode=NumberSelectorMode.SLIDER
                )
            ),
            vol.Optional(
                CONF_MAX_TOKENS,
                default=options.get(CONF_MAX_TOKENS, RECOMMENDED_AI_TASK_MAX_TOKENS),
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
            ): int,
        })

    elif subentry_type == "tts":
        schema.update({
            vol.Optional(
                CONF_CHAT_MODEL,
                default=options.get(CONF_CHAT_MODEL, RECOMMENDED_TTS_MODEL),
                description={"suggested_value": options.get(CONF_CHAT_MODEL)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[RECOMMENDED_TTS_MODEL],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_TTS_VOICE,
                default=options.get(CONF_TTS_VOICE, TTS_DEFAULT_VOICE),
                description={"suggested_value": options.get(CONF_TTS_VOICE)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=ZHIPUAI_TTS_VOICES,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_TTS_SPEED,
                default=options.get(CONF_TTS_SPEED, TTS_DEFAULT_SPEED),
                description={"suggested_value": options.get(CONF_TTS_SPEED)},
            ): NumberSelector(
                NumberSelectorConfig(
                    min=TTS_SPEED_MIN, max=TTS_SPEED_MAX,
                    step=TTS_SPEED_STEP, mode=NumberSelectorMode.SLIDER
                )
            ),
            vol.Optional(
                CONF_TTS_VOLUME,
                default=options.get(CONF_TTS_VOLUME, TTS_DEFAULT_VOLUME),
                description={"suggested_value": options.get(CONF_TTS_VOLUME)},
            ): NumberSelector(
                NumberSelectorConfig(
                    min=TTS_VOLUME_MIN, max=TTS_VOLUME_MAX,
                    step=TTS_VOLUME_STEP, mode=NumberSelectorMode.SLIDER
                )
            ),
            vol.Optional(
                CONF_TTS_RESPONSE_FORMAT,
                default=options.get(CONF_TTS_RESPONSE_FORMAT, TTS_DEFAULT_RESPONSE_FORMAT),
                description={"suggested_value": options.get(CONF_TTS_RESPONSE_FORMAT)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=ZHIPUAI_TTS_RESPONSE_FORMATS,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_TTS_ENCODE_FORMAT,
                default=options.get(CONF_TTS_ENCODE_FORMAT, TTS_DEFAULT_ENCODE_FORMAT),
                description={"suggested_value": options.get(CONF_TTS_ENCODE_FORMAT)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=ZHIPUAI_TTS_ENCODE_FORMATS,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_TTS_STREAM,
                default=options.get(CONF_TTS_STREAM, TTS_DEFAULT_STREAM),
                description={"suggested_value": options.get(CONF_TTS_STREAM)},
            ): bool,
        })

    elif subentry_type == "stt":
        schema.update({
            vol.Optional(
                CONF_CHAT_MODEL,
                default=options.get(CONF_CHAT_MODEL, RECOMMENDED_STT_MODEL),
                description={"suggested_value": options.get(CONF_CHAT_MODEL)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[RECOMMENDED_STT_MODEL],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_STT_TEMPERATURE,
                default=options.get(CONF_STT_TEMPERATURE, STT_DEFAULT_TEMPERATURE),
                description={"suggested_value": options.get(CONF_STT_TEMPERATURE)},
            ): NumberSelector(
                NumberSelectorConfig(
                    min=STT_TEMPERATURE_MIN, max=STT_TEMPERATURE_MAX,
                    step=STT_TEMPERATURE_STEP, mode=NumberSelectorMode.SLIDER
                )
            ),
            vol.Optional(
                CONF_STT_LANGUAGE,
                default=options.get(CONF_STT_LANGUAGE, "zh"),
                description={"suggested_value": options.get(CONF_STT_LANGUAGE)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        {"value": "zh", "label": "中文"},
                        {"value": "en", "label": "English"},
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
        })

    return schema
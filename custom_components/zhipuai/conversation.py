"""Conversation support for 智谱清言."""

from __future__ import annotations

import logging
from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_LLM_HASS_API, CONF_PROMPT, DOMAIN
from .entity import ZhipuAIBaseLLMEntity

_LOGGER = logging.getLogger(__name__)

MATCH_ALL = "*"


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue

        async_add_entities(
            [ZhipuAIConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class ZhipuAIConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    ZhipuAIBaseLLMEntity,
):
    """智谱清言 conversation agent."""

    _attr_supports_streaming = True

    def __init__(
        self, entry: ConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the agent."""
        from .const import RECOMMENDED_CHAT_MODEL

        super().__init__(entry, subentry, RECOMMENDED_CHAT_MODEL)

        # Enable control feature if LLM Hass API is configured
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process a sentence and return a response."""
        options = self.subentry.data

        try:
            # Provide LLM data (tools, home info, etc.)
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        # Process the chat log with ZhipuAI
        # Loop to handle tool calls: model may call tools, then we need to call again with results
        while True:
            await self._async_handle_chat_log(chat_log)

            # If there are unresponded tool results, continue the loop
            if not chat_log.unresponded_tool_results:
                break

        # Return result from chat log
        return conversation.async_get_result_from_chat_log(user_input, chat_log)

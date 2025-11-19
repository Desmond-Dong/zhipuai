"""Conversation support for 智谱清言."""

from __future__ import annotations

import logging
from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers import intent

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

        # 检查是否是自动化创建请求
        try:
            automation_result = await self._handle_automation_request(user_input)
            if automation_result:
                return automation_result
        except Exception as e:
            _LOGGER.debug("Automation handling failed: %s", e)

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

  
    async def _handle_automation_request(
        self,
        user_input: conversation.ConversationInput
    ) -> Optional[conversation.ConversationResult]:
        """Handle automation creation requests."""
        user_text = user_input.text.lower()

        # 检查是否包含自动化相关的关键词
        automation_keywords = [
            "创建自动化", "创建一个自动化", "设置自动化", "自动执行",
            "automation", "create automation", "自动", "定时执行",
            "当...时候", "如果...就", "当...就", "帮我创建"
        ]

        if not any(keyword in user_text for keyword in automation_keywords):
            return None

        try:
            from .ai_automation import get_automation_manager
            manager = get_automation_manager(self.hass)

            # 提取自动化描述
            description = self._extract_automation_description(user_input.text)
            if not description:
                return None

            # 创建自动化
            result = await manager.create_automation_from_description(description)

            # 创建响应
            intent_response = intent.IntentResponse(language=user_input.language)

            if result.get("success", False):
                config = result.get("config", {})
                automation_name = config.get("alias", "新自动化")
                message = f"我已经为您创建了自动化: {automation_name}"

                # 可以添加更多配置详情
                if config.get("trigger"):
                    triggers = [t.get("platform", "unknown") for t in config["trigger"]]
                    message += f"\\n触发条件: {', '.join(triggers)}"

                intent_response.async_set_speech(message)
            else:
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"创建自动化失败: {result.get('error', '未知错误')}"
                )

            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id
            )

        except Exception as e:
            _LOGGER.error("Error handling automation request: %s", e)
            return None

    def _extract_automation_description(self, user_text: str) -> Optional[str]:
        """Extract automation description from user input."""
        # 移除常见的自动化创建前缀
        prefixes = [
            "创建自动化", "创建一个自动化", "设置自动化", "帮我创建",
            "create automation", "automation", "自动", "帮我设置"
        ]

        description = user_text
        for prefix in prefixes:
            if prefix in description:
                description = description.replace(prefix, "").strip()

        # 如果描述太短或为空，返回None
        if len(description) < 5:
            return None

        return description

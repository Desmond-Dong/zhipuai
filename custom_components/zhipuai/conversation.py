from __future__ import annotations
import json
import asyncio
import time
import re
from datetime import datetime, timedelta
from typing import Any, TypedDict, Dict, List, Optional, AsyncGenerator
from voluptuous_openapi import convert 
from homeassistant.components import assist_pipeline, conversation
from homeassistant.components.conversation import trace
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import device_registry as dr, intent, llm, template, entity_registry, chat_session
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.history import get_significant_states
from homeassistant.util import ulid
from home_assistant_intents import get_languages
from .ai_request import send_ai_request, send_api_request
from .intents import IntentHandler, extract_intent_info
from .markdown_filter import filter_markdown_content
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_MAX_HISTORY_MESSAGES, 
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_MAX_HISTORY_MESSAGES,  
    RECOMMENDED_TOP_P,
    CONF_MAX_TOOL_ITERATIONS,
    DEFAULT_MAX_TOOL_ITERATIONS,
    LOGGER,
    CONF_HISTORY_ANALYSIS,
    CONF_HISTORY_ENTITIES,
    CONF_HISTORY_DAYS,
    DEFAULT_HISTORY_DAYS,
    CONF_WEB_SEARCH,
    DEFAULT_WEB_SEARCH,
    CONF_HISTORY_INTERVAL,
    DEFAULT_HISTORY_INTERVAL,
    CONF_PRESENCE_PENALTY,
    DEFAULT_PRESENCE_PENALTY,
    CONF_FREQUENCY_PENALTY,
    DEFAULT_FREQUENCY_PENALTY,
    CONF_STOP_SEQUENCES,
    DEFAULT_STOP_SEQUENCES,
    CONF_TOOL_CHOICE,
    DEFAULT_TOOL_CHOICE,
    CONF_LOGIT_BIAS,
    DEFAULT_LOGIT_BIAS,
    CONF_REQUEST_TIMEOUT,
    DEFAULT_REQUEST_TIMEOUT,
    CONF_FILTER_MARKDOWN,
    DEFAULT_FILTER_MARKDOWN,
)
from .prompts import (
    get_prompts_for_text, 
    get_prompts_for_tools, 
    get_basic_tools_guide,
    get_tools_for_text,
    get_web_search_tool,
    is_web_search_request,
    detect_keywords_in_text

)
from .optimizations import (
    ParallelToolExecutor,
    ContextCompressor,
    SmartRetryHandler
)


class ChatCompletionMessageParam(TypedDict, total=False):
    role: str
    content: str | None
    name: str | None
    tool_calls: list["ChatCompletionMessageToolCallParam"] | None

class Function(TypedDict, total=False):
    name: str
    arguments: str

class ChatCompletionMessageToolCallParam(TypedDict):
    id: str
    type: str
    function: Function

class ChatCompletionToolParam(TypedDict):
    type: str
    function: dict[str, Any]


def _format_tool(tool: llm.Tool, custom_serializer: Any | None) -> ChatCompletionToolParam:
    tool_spec = {
        "name": tool.name,
        "description": tool.description or f"用于{tool.name}的工具",
        "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
    }
    return ChatCompletionToolParam(type="function", function=tool_spec)

def is_service_call(user_input: str) -> bool:
    patterns = {
        "control": ["让", "请", "帮我", "麻烦", "把", "将", "计时", "要", "想", "希望", "需要", "能否", "能不能", "可不可以", "可以", "帮忙", "给我", "替我", "为我", "我要", "我想", "我希望"],
        "action": {
            "turn_on": ["打开", "开启", "启动", "激活", "运行", "执行"],
            "turn_off": ["关闭", "关掉", "停止"],
            "toggle": ["切换"],
            "press": ["按", "按下", "点击"],
            "select": ["选择", "下一个", "上一个", "第一个", "最后一个", "上1个", "下1个"],
            "trigger": ["触发", "调用"],
            "number": ["数字", "数值"],
            "media": ["暂停", "继续播放", "停止", "下一首", "下一曲", "下一个", "切歌", "换歌", "上一首", "上一曲", "上一个", "返回上一首", "上1首", "上1曲", "上1个", "下1首", "下1曲", "下1个", "音量"]
        }
    }
    
    return bool(user_input and (
        any(k in user_input for k in patterns["control"]) or 
        any(k in user_input for action in patterns["action"].values() for k in (action if isinstance(action, list) else []))
    ))

def extract_service_info(user_input: str, hass: HomeAssistant) -> Optional[Dict[str, Any]]:
    def find_entity(domain: str, text: str) -> Optional[str]:
        text = text.lower()
        entity_id = next((entity_id for entity_id in hass.states.async_entity_ids(domain) 
                    if text in entity_id.split(".")[1].lower() or 
                    text in hass.states.get(entity_id).attributes.get("friendly_name", "").lower() or
                    entity_id.split(".")[1].lower() in text or
                    hass.states.get(entity_id).attributes.get("friendly_name", "").lower() in text), None)
        
        if not entity_id:
            ent_reg = entity_registry.async_get(hass)
            for reg_entity_id in hass.states.async_entity_ids(domain):
                reg_entity = ent_reg.async_get(reg_entity_id)
                if reg_entity and hasattr(reg_entity, "aliases") and reg_entity.aliases:
                    for alias in reg_entity.aliases:
                        if text in alias.lower() or alias.lower() in text:
                            return reg_entity_id
        
        return entity_id

    def clean_text(text: str, patterns: List[str]) -> str:
        control_words = ["让", "请", "帮我", "麻烦", "把", "将"]
        return "".join(char for char in text if not any(word in char for word in patterns + control_words)).strip()

    if not is_service_call(user_input):
        return None

    if not hasattr(hass, '_last_media_player'):
        hass._last_media_player = {}
    
    last_media_player = None
    if hasattr(hass, '_last_media_player'):
        last_media_player = hass._last_media_player.get('entity_id')
    
    media_patterns = {"暂停": "media_pause", "继续播放": "media_play", "停止": "media_stop",
                     "下一首": "media_next_track", "下一曲": "media_next_track", "下一个": "media_next_track",
                     "切歌": "media_next_track", "换歌": "media_next_track", "上一首": "media_previous_track",
                     "上一曲": "media_previous_track", "上一个": "media_previous_track",
                     "返回上一首": "media_previous_track", "音量": "volume_set", 
                     "上1首": "media_previous_track", "下1首": "media_next_track",
                     "上1曲": "media_previous_track", "下1曲": "media_next_track",
                     "上1个": "media_previous_track", "下1个": "media_next_track"}
    
    text_lower = user_input.lower()
    if "播放器" in text_lower:
        entity_id = find_entity("media_player", text_lower)
    else:
        entity_id = find_entity("media_player", text_lower)
        if not entity_id and last_media_player and any(p in text_lower for p in media_patterns.keys()):
            entity_id = last_media_player
    
    if entity_id:
        hass._last_media_player = {'entity_id': entity_id}
        
        for pattern, service in media_patterns.items():
            if pattern in user_input.lower():
                if service == "volume_set":
                    volume_match = re.search(r'(\d+)', user_input)
                    if volume_match:
                        return {"domain": "media_player", "service": service, "data": {"entity_id": entity_id, "volume_level": int(volume_match.group(1)) / 100}}
                    else:
                        return {"domain": "media_player", "service": "media_play", "data": {"entity_id": entity_id}}
                else:
                    return {"domain": "media_player", "service": service, "data": {"entity_id": entity_id}}

    if any(p in user_input for p in ["按", "按下", "点击"]):
        return {"domain": "button", "service": "press", "data": {"entity_id": (re.search(r'(button\.\w+)', user_input).group(1) if re.search(r'(button\.\w+)', user_input) else 
                find_entity("button", clean_text(user_input, ["按", "按下", "点击"])))}} if (re.search(r'(button\.\w+)', user_input) or 
                find_entity("button", clean_text(user_input, ["按", "按下", "点击"]))) else None

    select_patterns = {"下一个": ("select_next", True), "上一个": ("select_previous", True),
                      "第一个": ("select_first", False), "最后一个": ("select_last", False),
                      "选择": ("select_option", False),
                      "上1个": ("select_previous", True), "下1个": ("select_next", True),
                      "上1个": ("select_previous", True), "下1个": ("select_next", True)}
                      
    if entity_id := find_entity("select", user_input):
        return {"domain": "select", "service": select_patterns.get(next((k for k in select_patterns.keys() if k in user_input), "选择"))[0],
                "data": {"entity_id": entity_id, "cycle": select_patterns.get(next((k for k in select_patterns.keys() if k in user_input), "选择"))[1]}} if any(p in user_input for p in select_patterns.keys()) else None

    if any(p in user_input for p in ["触发", "调用", "执行", "运行", "启动"]):
        name = clean_text(user_input, ["触发", "调用", "执行", "运行", "启动", "脚本", "自动化", "场景"])
        return next(({"domain": domain, "service": service, "data": {"entity_id": entity_id}}
                    for domain, service in [("script", "turn_on"), ("automation", "trigger"), ("scene", "turn_on")]
                    if (entity_id := find_entity(domain, name))), None)

    if any(p in user_input for p in ["数字", "数值"]) and (number_match := re.search(r'\d+(?:\.\d+)?', user_input)) and (entity_id := find_entity("number", clean_text(user_input, ["数字", "数值"]))):
        return {"domain": "number", "service": "set_value", "data": {"entity_id": entity_id, "value": number_match.group(0)}}

    return None

class AIResponseStrategy:
    @staticmethod
    async def direct_stream(api_key, payload, options, chat_log, entity_id, transform_stream_func, llm_api, on_tool_call):
        timeout = options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
        stream_generator = send_ai_request(api_key, payload, options, timeout=timeout)
        final_content = ""
        tool_calls_detected = []
        
        try:
            content_stream = transform_stream_func(stream_generator, llm_api)
            
            async for content in chat_log.async_add_delta_content_stream(entity_id, content_stream):
                if isinstance(content, conversation.AssistantContent):
                    if content.content:
                        final_content += content.content
                    if hasattr(content, 'tool_calls') and content.tool_calls:
                        tool_calls_detected.extend(content.tool_calls)
                elif isinstance(content, dict):
                    if content.get('content'):
                        final_content += content.get('content')
                    if content.get('tool_calls'):
                        tool_calls_detected.extend(content.get('tool_calls'))
                    if content.get('collected_tool_calls'):
                        tool_calls_detected = content.get('collected_tool_calls')
                        break
            
            if tool_calls_detected and on_tool_call:
                await on_tool_call(tool_calls_detected, final_content)
                return None
            
            return final_content
        except asyncio.CancelledError:
            
            LOGGER.info("流处理被取消")
            raise
        except Exception as e:
            
            LOGGER.info("流处理过程中发生错误: %s", str(e))
            return final_content
    
    @staticmethod
    async def collect_stream(api_key, payload, options):
        timeout = options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
        tool_calls_from_ai = []
        ai_content = ""
        
        
        if "tools" in payload and "tool_choice" in payload:
            tool_names = [t.get("function", {}).get("name", "unknown") for t in payload.get("tools", [])]

        
        async for chunk in send_ai_request(api_key, payload, options, timeout=timeout):
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            
            if delta.get("content"):
                ai_content += delta["content"]
            
            if "tool_calls" in delta and delta["tool_calls"]:
                for tool_call in delta["tool_calls"]:
                    if (tool_call.get("type") == "function" and "function" in tool_call and
                        "name" in tool_call["function"] and "arguments" in tool_call["function"]):
                        tool_calls_from_ai.append(tool_call)
        
        return ai_content, tool_calls_from_ai

    @staticmethod
    async def non_stream_request(hass, api_key, payload, options):
        
        return await send_api_request(api_key, payload, options)

class ToolCallProcessor:
    def __init__(self, entity, api_key, options):
        self.entity = entity
        self.api_key = api_key
        self.options = options
        self.id_tracker = IdTracker()
        self.message_factory = MessageFactory()
        self.result_handler = ResultHandler(entity)
        self.window_id = f"tool_call_{int(time.time())}"
    
    async def process(self, tool_calls, ai_content, messages, user_input, base_payload):
        request_id = f"req_{int(time.time())}"
        current_content = ai_content or ""
        max_iterations = self.entity.max_tool_iterations
        pending_calls = tool_calls.copy() if tool_calls else []
        
        LOGGER.info("开始处理工具调用，最大迭代次数: %s，工具调用数: %s", 
                   max_iterations, len(pending_calls) if pending_calls else 0)
        
        
        for iteration in range(max_iterations):
            if not pending_calls:
                break
            
            
            new_call = None
            for call in pending_calls:
                call_id = call.get("id")
                if not call_id or call_id not in self.id_tracker.processed_ids:
                    new_call = call
                    if call_id:
                        self.id_tracker.processed_ids.add(call_id)
                    break
            
            if not new_call:
                pending_calls = []
                continue
            
            
            messages.append(self.message_factory.create_assistant_message(
                current_content, [new_call]))
            
            
            results = await self._execute_tool_calls([new_call], user_input.text)
            
            
            for result in results:
                messages.append(self.message_factory.create_tool_result_message(result))
            
            
            session = chat_session.ChatSession(user_input.conversation_id or ulid.ulid_now())
            with conversation.async_get_chat_log(self.entity.hass, session, user_input) as chat_log:
                payload = dict(base_payload)
                payload["messages"] = messages.copy()
                payload["stream"] = True
                
                try:
                    response_data = await self._get_ai_response(
                        payload, chat_log, self.entity.entity_id)
                    
                    current_content = response_data.get("content", current_content)
                    new_pending_calls = response_data.get("tool_calls", [])
                    
                    if new_pending_calls:
                        pending_calls = [
                            call for call in new_pending_calls 
                            if not call.get("id") or call.get("id") not in self.id_tracker.processed_ids
                        ]
                    else:
                        pending_calls = [
                            call for call in pending_calls 
                            if call != new_call
                        ]
                
                except Exception as e:
                    LOGGER.exception("工具调用迭代 %s/%s 处理响应时出错: %s", 
                                    iteration + 1, max_iterations, str(e))
                    pending_calls = [call for call in pending_calls if call != new_call]
        
        
        final_content = current_content or ai_content or getattr(self.entity, '_last_error_message', "")
        
        filtered_content = self.entity._filter_response_content(final_content)
        await self.entity._update_response(filtered_content)
        
        
        final_message = self.message_factory.create_final_message(final_content)
        messages.append(final_message)
        
        
        if hasattr(self.entity, 'session_histories') and user_input.conversation_id:
            self._save_session_history(messages, filtered_content, user_input.conversation_id)
        
        return filtered_content
    
    async def _execute_tool_calls(self, tool_calls, user_text):
        """执行工具调用 - 使用并行执行器优化"""
        return await self.entity.parallel_executor.execute_parallel(tool_calls, user_text)

    async def _execute_single_call(self, tool_call, user_text):
        """执行单个工具调用 - 使用智能重试处理器"""
        tool_name = tool_call["function"].get("name", "")
        tool_call_id = f"{self.window_id}_{int(time.time())}"
        
        async def execute_call(tool_call, user_text):
            tool_args_json = tool_call["function"].get("arguments", "{}")
            tool_call_id = tool_call.get("id", tool_call_id)
            
            LOGGER.debug("执行工具调用: %s, 参数: %s", tool_name, tool_args_json)
            
            tool_args = json.loads(tool_args_json) if tool_args_json else {"text": tool_args_json}
            tool_args["_window_id"] = self.window_id
            
            tool_input = llm.ToolInput(id=tool_call_id, tool_name=tool_name, tool_args=tool_args)
            result = await self.entity._handle_tool_call(tool_input, user_text)
            
            self.entity._last_tool_result = result
            
            result_content = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
            
            if not result.get("success", True):
                self.entity._last_error_message = result.get("message", result_content)
                LOGGER.error("工具调用失败: %s, 错误: %s", tool_name, self.entity._last_error_message)
            
            return {
                "success": result.get("success", True), 
                "tool_call_id": tool_call_id, 
                "tool_name": tool_name, 
                "content": result_content
            }
        
        return await self.entity.retry_handler.execute_with_retry(
            tool_call, 
            execute_call, 
            user_text
        )

    def _save_session_history(self, messages, filtered_content, conversation_id):
        """保存会话历史 - 使用上下文压缩器优化"""
        filtered_history = []
        
        for msg in messages:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            
            if role == "user":
                if isinstance(msg, dict):
                    filtered_history.append(msg)
                else:
                    filtered_history.append({"role": "user", "content": getattr(msg, "content", "")})
            
            elif role == "assistant":
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
                
                if content and not tool_calls:
                    filtered_history.append({"role": "assistant", "content": content})
        
        if filtered_content:
            if filtered_history and filtered_history[-1].get("role") == "assistant":
                filtered_history[-1] = {"role": "assistant", "content": filtered_content}
            else:
                filtered_history.append({"role": "assistant", "content": filtered_content})
        
        # 使用上下文压缩器压缩历史
        compressed_history = self.entity.context_compressor.compress_messages(filtered_history)
        self.entity.session_histories[conversation_id] = compressed_history

    async def _get_ai_response(self, payload, chat_log, entity_id):
        payload["stream"] = True
        
        try:
            stream = send_ai_request(self.api_key, payload, self.options, 
                                  timeout=self.options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT))
            content_stream = self.entity._transform_stream(stream, self.entity.llm_api)
            result = {"content": "", "tool_calls": []}
            seen_tool_ids = set()
            
            content_handlers = {
                conversation.AssistantContent: lambda content: {
                    "content": content.content if content.content else "",
                    "tool_calls": [call for call in (content.tool_calls if hasattr(content, 'tool_calls') and content.tool_calls else [])
                                 if call.get("id") and call.get("id") not in seen_tool_ids]
                },
                dict: lambda content: {
                    "content": content.get('content', ""),
                    "tool_calls": [call for call in content.get('tool_calls', [])
                                 if call.get("id") and call.get("id") not in seen_tool_ids],
                    "collected_tool_calls": content.get('collected_tool_calls')
                }
            }
            
            async for content in chat_log.async_add_delta_content_stream(entity_id, content_stream):
                handler = content_handlers.get(type(content))
                if handler:
                    processed = handler(content)
                    result["content"] += processed["content"]
                    
                    for call in processed.get("tool_calls", []):
                        call_id = call.get("id")
                        if call_id:
                            seen_tool_ids.add(call_id)
                            result["tool_calls"].append(call)
                    
                    if processed.get("collected_tool_calls"):
                        new_calls = [call for call in processed["collected_tool_calls"] 
                                   if call.get("id") and call.get("id") not in seen_tool_ids]
                        for call in new_calls:
                            seen_tool_ids.add(call.get("id"))
                        result["tool_calls"] = new_calls if new_calls else result["tool_calls"]
            
            
            result["tool_calls"] = [call for call in result["tool_calls"] 
                                  if call.get("function") and call["function"].get("name") and call["function"].get("arguments")]
            
            return result
        except Exception as e:
            error_text = str(e)
            return {"content": f"处理请求时出错: {error_text}", "tool_calls": []}

class IdTracker:
    def __init__(self):
        self.processed_ids = set()
    
    def filter_calls(self, tool_calls):
        new_calls = []
        for call in tool_calls:
            call_id = call.get("id")
            if call_id and call_id in self.processed_ids:
                continue
            if call_id:
                self.processed_ids.add(call_id)
            new_calls.append(call)
        return new_calls
    
    def all_processed(self, tool_calls):
        return all(call.get("id") in self.processed_ids for call in tool_calls if call.get("id"))


class MessageFactory:
    def create_assistant_message(self, content, tool_calls):
        return ChatCompletionMessageParam(
            role="assistant",
            content=content,
            tool_calls=tool_calls
        )
    
    def create_tool_result_message(self, result):
        return {
            "role": "tool",
            "content": result["content"],
            "name": result["tool_name"],
            "tool_call_id": result["tool_call_id"]
        }
    
    def create_final_message(self, content):
        return ChatCompletionMessageParam(
            role="assistant",
            content=content
        )


class ResultHandler:
    def __init__(self, entity):
        self.entity = entity
    
    async def update_response(self, content):
        filtered = self.entity._filter_response_content(content)
        await self.entity._update_response(filtered)
        return filtered


class ZhipuAIConversationEntity(conversation.ConversationEntity, conversation.AbstractConversationAgent):
    _attr_has_entity_name = True
    _attr_name = None
    _attr_response = ""

    def __init__(self, entry: ConfigEntry, subentry, hass: HomeAssistant) -> None:
        self.entry = entry
        self.subentry = subentry
        self.hass = hass
        self.session_histories: dict[str, list[ChatCompletionMessageParam]] = {}
        self._current_session_id = None
        
        # 初始化优化器
        self.parallel_executor = ParallelToolExecutor(self, max_parallel=5)
        self.context_compressor = ContextCompressor(max_tokens=4000)
        self.retry_handler = SmartRetryHandler(max_retries=3)
        
        # unique_id和identifiers归属于subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_entity_id = f"conversation.{DOMAIN}_{subentry.subentry_id}"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="北京智谱华章科技",
            model="ChatGLM AI",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self._attr_supported_features = conversation.ConversationEntityFeature.CONTROL
        self.max_tool_iterations = min(entry.options.get(CONF_MAX_TOOL_ITERATIONS, DEFAULT_MAX_TOOL_ITERATIONS), 60)
        self.llm_api = None
        self.intent_handler = IntentHandler(hass)
        self.entity_registry = er.async_get(hass)
        self.device_registry = dr.async_get(hass)
        self.service_call_attempts = 0
        self._attr_native_value = "就绪"
        self._attr_extra_state_attributes = {"response": ""}
        # availability / agent registration tracking
        self._attr_available = True
        self._agent_registered = False

    @property
    def supported_languages(self) -> list[str]:
        return list(dict.fromkeys(languages + ["zh-cn", "zh-tw", "zh-hk", "en"])) if (languages := get_languages()) and "zh" in languages else languages

    @property
    def available(self) -> bool:
        """Return True if the conversation agent is available.

        Keep entity available when the integration is set up even if optional
        assist pipeline registration failed — we want the entity to be usable.
        """
        # If we have successfully registered our agent, mark available
        if getattr(self, "_agent_registered", False):
            return True

        # If the parent config entry is present in hass.data, consider it available
        entry = getattr(self, "entry", None)
        if entry and self.hass and self.hass.data.get(DOMAIN) and entry.entry_id in self.hass.data.get(DOMAIN):
            return True

        return bool(getattr(self, "_attr_available", True))

    @property
    def state_attributes(self):
        attributes = super().state_attributes or {}
        attributes["entity"] = "ZHIPU.AI"
        if self._attr_response:
            attributes["response"] = self._attr_response
        return attributes

    def _filter_response_content(self, content: str) -> str:
        filter_markdown = self.entry.options.get(CONF_FILTER_MARKDOWN, DEFAULT_FILTER_MARKDOWN)
        return filter_markdown_content(content, filter_markdown == "on")

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()

        # Guard calls that may not exist on all HA versions or may fail
        try:
            migrate = getattr(assist_pipeline, "async_migrate_engine", None)
            if migrate:
                result = migrate(self.hass, "conversation", self.entry.entry_id, self.entity_id)
                if asyncio.iscoroutine(result):
                    await result
        except Exception as err:  # defensive: don't let migration failure make entity unavailable
            LOGGER.debug("assist_pipeline.async_migrate_engine failed: %s", err)

        try:
            set_agent = conversation.async_set_agent
            result = set_agent(self.hass, self.entry, self)
            if asyncio.iscoroutine(result):
                await result
            self._agent_registered = True
        except Exception as err:
            LOGGER.exception("Failed to register conversation agent: %s", err)
            self._agent_registered = False

        # Reflect agent registration in availability so UI doesn't show 'unavailable'
        self._attr_available = bool(self._agent_registered)
        try:
            self.async_write_ha_state()
        except Exception:
            pass

        try:
            self.entry.async_on_unload(self.entry.add_update_listener(self._async_entry_update_listener))
        except Exception:
            pass

    async def async_will_remove_from_hass(self) -> None:
        self.session_histories.clear()
        try:
            conversation.async_unset_agent(self.hass, self.entry)
        except Exception:
            pass
        await super().async_will_remove_from_hass()

    async def async_process(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        try:
            if not hasattr(self, '_conversation_history'):
                self._conversation_history = {}
            
            if user_input.conversation_id != getattr(self, '_last_conversation_id', None):
                self._conversation_history[user_input.conversation_id] = []
                self._last_conversation_id = user_input.conversation_id
            
            if len(user_input.text) <= 300:  
                self._conversation_history[user_input.conversation_id].append({
                    "role": "user",
                    "content": user_input.text,
                    "timestamp": time.time()
                })

            is_internal_call = False
            if user_input.text and '[INTERNAL_CALL]' in user_input.text:
                is_internal_call = True
                user_input.text = user_input.text.replace('[INTERNAL_CALL]', '')
            
            if is_internal_call and getattr(self, '_last_tool_call_time', 0) > time.time() - 2:
                intent_response = intent.IntentResponse(language=user_input.language)
                
                if hasattr(self, '_last_error_message') and self._last_error_message:
                    error_message = self._last_error_message
                elif hasattr(self, '_last_tool_result') and self._last_tool_result:
                    if isinstance(self._last_tool_result, dict):
                        error_message = json.dumps(self._last_tool_result, ensure_ascii=False)
                    else:
                        error_message = str(self._last_tool_result)
                else:
                    error_message = "MatchFailedError: MatchFailedReason.AREA: 2"  
                
                intent_response.async_set_speech(error_message)
                return conversation.ConversationResult(response=intent_response, conversation_id=user_input.conversation_id or ulid.ulid_now())
            
            self._last_tool_call_time = time.time()
            if (user_input.context and user_input.context.id and user_input.context.id.startswith(f"{DOMAIN}_service_call")) or getattr(user_input, "prefer_local_intents", False):
                return conversation.ConversationResult(None)

            self._last_user_input = user_input.text
            conversation_id = user_input.conversation_id or ulid.ulid_now()
            session = chat_session.ChatSession(conversation_id)
            
            with conversation.async_get_chat_log(self.hass, session, user_input) as chat_log:
                if service_info := (is_service_call(user_input.text) and extract_service_info(user_input.text, self.hass)):
                    result = await self.intent_handler.call_service(service_info["domain"], service_info["service"], service_info["data"])
                    intent_response = intent.IntentResponse(language=user_input.language)
                    if result["success"]: intent_response.async_set_speech(result["message"])
                    else: intent_response.async_set_error(intent.IntentResponseErrorCode.NO_VALID_TARGETS, result["message"])
                    return conversation.ConversationResult(response=intent_response, conversation_id=conversation_id)
                
                if intent_info := extract_intent_info(user_input.text, self.hass):
                    result = await self.intent_handler.handle_intent(intent_info)
                    intent_response = intent.IntentResponse(language=user_input.language)
                    result["success"] and intent_response.async_set_speech(result["message"]) or intent_response.async_set_error(
                        intent.IntentResponseErrorCode.NO_VALID_TARGETS, result["message"])
                    return conversation.ConversationResult(response=intent_response, conversation_id=conversation_id)
                
                options = self.entry.options
                tools = None
                user_name = None
                llm_context = llm.LLMContext(platform=DOMAIN, context=user_input.context,
                    language=user_input.language, assistant=conversation.DOMAIN, device_id=user_input.device_id)

                
                current_model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
                is_z1_model = any(z1_name in current_model.upper() for z1_name in ["Z1-AIR", "Z1-AIRX", "Z1-FLASH", "Z1-FLASHX"])

                if is_z1_model:
                    LOGGER.warning("检测到Z1系列模型 %s，自动禁用工具调用", current_model)
                    
                    options_copy = dict(options)
                    options_copy[CONF_TOOL_CHOICE] = "none"
                    
                    options = options_copy
                    
                    
                    z1_warning = "\n注意: 您正在使用Z1系列模型，此模型只能用于查询设备状态，不支持设备控制和工具调用。请直接回答用户问题，不要尝试使用工具。如果用户需要控制设备，请礼貌告知需要使用其他系列模型才能执行此操作。\n"

                try:
                    if not options.get(CONF_LLM_HASS_API) or options[CONF_LLM_HASS_API] == "none":
                        api_key = self.entry.data[CONF_API_KEY]
                        base_payload = {
                            "model": options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                            "max_tokens": min(options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS), 8096),
                            "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                            "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                            "request_id": conversation_id,
                            "presence_penalty": options.get(CONF_PRESENCE_PENALTY, DEFAULT_PRESENCE_PENALTY),
                            "frequency_penalty": options.get(CONF_FREQUENCY_PENALTY, DEFAULT_FREQUENCY_PENALTY),
                            "stop": options.get(CONF_STOP_SEQUENCES, DEFAULT_STOP_SEQUENCES),
                            "logit_bias": options.get(CONF_LOGIT_BIAS, DEFAULT_LOGIT_BIAS),
                            "response_format": {"type": "text"}
                        }
                        
                        tool_choice_setting = options.get(CONF_TOOL_CHOICE, DEFAULT_TOOL_CHOICE)
                        
                        base_payload["tool_choice"] = "required"
                        base_payload["temperature"] = 0.1  
                        base_payload["top_p"] = 0.1  
                        base_payload["do_sample"] = False  


                        if tools:
                            base_payload["tool_choice"] = "required"
                            base_payload["temperature"] = 0.1
                            base_payload["top_p"] = 0.1
                            base_payload["do_sample"] = False
                            base_payload["tools"] = tools

                        if is_z1_model:
                            base_payload["tool_choice"] = "none"
                            
                            if "tools" in base_payload:
                                del base_payload["tools"]
                            
                            if "messages" not in base_payload:
                                base_payload["messages"] = [ChatCompletionMessageParam(role="user", content=user_input.text)]

                        try:
                            final_content = await AIResponseStrategy.direct_stream(
                                api_key, base_payload, options, chat_log, 
                                self.entity_id, self._transform_stream, None, None
                            )
                            
                            if final_content:
                                filtered_content = self._filter_response_content(final_content)
                                await self._update_response(filtered_content)
                                intent_response = intent.IntentResponse(language=user_input.language)
                                intent_response.async_set_speech(filtered_content)
                                return conversation.ConversationResult(
                                    response=intent_response,
                                    conversation_id=conversation_id,
                                    continue_conversation=chat_log.continue_conversation,
                                )
                        except Exception as e:
                            intent_response = intent.IntentResponse(language=user_input.language)
                            intent_response.async_set_speech(f"{str(e)}")
                            return conversation.ConversationResult(
                                response=intent_response,
                                conversation_id=conversation_id
                            )

                    try:
                        self.llm_api = await llm.async_get_api(self.hass, options[CONF_LLM_HASS_API], llm_context)
                        chat_log.llm_api = self.llm_api
                        tools_description = ""
                        
                        if self.llm_api and hasattr(self.llm_api, "tools"):
                            tools = [_format_tool(tool, self.llm_api.custom_serializer) for tool in self.llm_api.tools]
                            
                            
                            additional_tools = get_tools_for_text(user_input.text, tools)
                            if additional_tools:
                                tools.extend(additional_tools)
                            
                            if tools:
                                tool_choice_setting = options.get(CONF_TOOL_CHOICE, DEFAULT_TOOL_CHOICE)
                                
                                
                                tools_desc_parts = get_basic_tools_guide(tool_choice_setting)
                                
                                
                                tools_desc_parts.append("\n")
                                for tool in tools:
                                    tool_name = tool["function"]["name"]
                                    tool_desc = tool["function"].get("description", "")
                                    params = tool["function"].get("parameters", {})
                                    required = params.get("required", [])
                                    
                                    tools_desc_parts.append(f"- {tool_name}: {tool_desc}")
                                    if required:
                                        tools_desc_parts.append(f"  Required parameters: {', '.join(required)}")
                                
                                
                                keywords = detect_keywords_in_text(user_input.text)
                                dynamic_prompts = get_prompts_for_text(user_input.text)

                                tools_desc_parts.append(f"\n\n{dynamic_prompts}")

                                tools_description = "\n".join(tools_desc_parts)
                        else:
                            tools_description = ""
                    except HomeAssistantError as err:
                        api_key = self.entry.data[CONF_API_KEY]
                        base_payload = {
                            "model": options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                            "max_tokens": min(options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS), 4096),
                            "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                            "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                            "request_id": conversation_id,
                            "logit_bias": options.get(CONF_LOGIT_BIAS, DEFAULT_LOGIT_BIAS),
                            "response_format": {"type": "text"}
                        }
                        final_content = await AIResponseStrategy.direct_stream(
                            api_key, base_payload, options, chat_log, 
                            self.entity_id, self._transform_stream, None, None
                        )
                        if final_content:
                            filtered_content = self._filter_response_content(final_content)
                            await self._update_response(filtered_content)
                            intent_response = intent.IntentResponse(language=user_input.language)
                            intent_response.async_set_speech(filtered_content)
                            return conversation.ConversationResult(
                                response=intent_response,
                                conversation_id=conversation_id,
                                continue_conversation=chat_log.continue_conversation,
                            )

                    web_search_keywords = ["联网", "搜索", "查询", "互联网", "上网", "百度", "谷歌", "必应"]
                    has_web_search = any(keyword in user_input.text for keyword in web_search_keywords)
                    if has_web_search and options.get(CONF_WEB_SEARCH, DEFAULT_WEB_SEARCH):
                        
                        additional_tools = get_tools_for_text(user_input.text, tools)
                        if additional_tools:
                            tools.extend(additional_tools)
                except HomeAssistantError:
                    pass

                
                if user_input.context and user_input.context.user_id:
                    user = await self.hass.auth.async_get_user(user_input.context.user_id)
                    if user:
                        user_name = user.name

                
                try:
                    er = entity_registry.async_get(self.hass)
                    entities_dict = {entity_id: er.async_get(entity_id) for entity_id in self.hass.states.async_entity_ids()}
                    exposed_entities = [entity for entity in entities_dict.values() if entity and not entity.hidden]
                    
                    max_history = options.get(CONF_MAX_HISTORY_MESSAGES, RECOMMENDED_MAX_HISTORY_MESSAGES)
                    
                    history_prompt = ""
                    if user_input.conversation_id in self._conversation_history:
                        history = self._conversation_history[user_input.conversation_id]
                        history_messages = list(history)[-max_history:] if len(history) > max_history else history
                        history_prompt = "\n用户历史消息\n" + "\n".join([
                            f"- {msg['role']}: {msg['content']}" 
                            for msg in history_messages
                        ])
                    
                    
                    media_player_prompt = ""
                    if hasattr(self.hass, '_last_media_player') and self.hass._last_media_player.get('entity_id'):
                        entity_id = self.hass._last_media_player.get('entity_id')
                        state = self.hass.states.get(entity_id)
                        if state:
                            friendly_name = state.attributes.get('friendly_name', entity_id)
                            media_player_prompt = f"\n当前活动的媒体播放器: {friendly_name} ({entity_id})"
                    
                    combined_prompt = "\n".join([
                        llm.BASE_PROMPT,
                        options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT),
                        history_prompt,
                        media_player_prompt,  
                    ])

                    prompt_parts = [template.Template(combined_prompt, self.hass).async_render({
                        "ha_name": self.hass.config.location_name,
                        "user_name": user_name,
                        "llm_context": llm_context,
                        "exposed_entities": exposed_entities if self.entry.options.get(CONF_LLM_HASS_API) and self.entry.options.get(CONF_LLM_HASS_API) != "none" else [],
                    }, parse_result=False)]


                    if self.entry.options.get(CONF_LLM_HASS_API) and self.entry.options.get(CONF_LLM_HASS_API) != "none" and self.entry.options.get(CONF_HISTORY_ANALYSIS):
                        if entities := self.entry.options.get(CONF_HISTORY_ENTITIES):
                            await self._add_history_analysis(prompt_parts, entities)

                    base_prompt = prompt_parts[0]
                    base_instructions = [line.strip() if line.startswith(" ") else line for line in base_prompt.split('\n') if line.strip()]
                    prompt_parts[0] = {"type": "system_instructions", "content": base_instructions}

                    all_lines = []
                    for part in prompt_parts:
                        if isinstance(part["content"], list): 
                            all_lines.extend([line.strip() if line.startswith(" ") else line for line in part["content"]])
                        else: 
                            all_lines.extend([line.strip() if line.startswith(" ") else line for line in part["content"].split('\n') if line.strip()])

                    
                    if self.llm_api and hasattr(self.llm_api, "api_prompt") and self.llm_api.api_prompt:
                        all_lines.append(self.llm_api.api_prompt)

                    
                    system_content = "\n".join(all_lines)
                    if dynamic_prompts:
                        system_content = system_content + "\n\n" + dynamic_prompts

                    if is_z1_model:
                        z1_warning = "\n注意: 您正在使用Z1系列模型，此模型只能用于查询设备状态，不支持设备控制和工具调用。请直接回答用户问题，不要尝试使用工具。如果用户需要控制设备，请礼貌告知需要使用其他系列模型才能执行此操作。\n"
                        system_content = system_content + z1_warning
                        
                        tools_description = ""
                    
                    messages_for_ai = [ChatCompletionMessageParam(role="system", content=system_content + "\n" + tools_description)]
                    recent_messages = list(chat_log.content)[-max_history:] if len(chat_log.content) > max_history else chat_log.content

                    
                    filtered_recent_messages = []
                    for msg in recent_messages:
                        
                        if msg.role == "user":
                            filtered_recent_messages.append(msg)
                        
                        elif msg.role == "assistant" and msg.content and not getattr(msg, "tool_calls", None):
                            filtered_recent_messages.append(msg)
                        

                    
                    for msg in filtered_recent_messages:
                        if msg.role == "user":
                            messages_for_ai.append(ChatCompletionMessageParam(role="user", content=msg.content or ""))
                        elif msg.role == "assistant":
                            messages_for_ai.append(ChatCompletionMessageParam(role="assistant", content=msg.content))

                    messages_for_ai.append(ChatCompletionMessageParam(role="user", content=user_input.text))
                    api_key = self.entry.data[CONF_API_KEY]
                    base_payload = {
                        "model": options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                        "max_tokens": min(options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS), 4096),
                        "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                        "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                        "request_id": conversation_id,
                        "presence_penalty": options.get(CONF_PRESENCE_PENALTY, DEFAULT_PRESENCE_PENALTY),
                        "frequency_penalty": options.get(CONF_FREQUENCY_PENALTY, DEFAULT_FREQUENCY_PENALTY),
                        "stop": options.get(CONF_STOP_SEQUENCES, DEFAULT_STOP_SEQUENCES),
                        "logit_bias": options.get(CONF_LOGIT_BIAS, DEFAULT_LOGIT_BIAS),
                        "response_format": {"type": "text"}
                    }
                    if tools: base_payload["tools"] = tools

                    if user_input.context and user_input.context.user_id:
                        user = await self.hass.auth.async_get_user(user_input.context.user_id)
                        if user and user.id:
                            safe_user_id = f"ha_user_{user.id}"
                            if 6 <= len(safe_user_id) <= 128:
                                base_payload["user_id"] = safe_user_id

                    
                    if is_z1_model:
                        base_payload["tool_choice"] = "none"
                        
                        if "tools" in base_payload:
                            del base_payload["tools"]
                        
                        if "messages" not in base_payload and len(messages_for_ai) > 0:
                            base_payload["messages"] = messages_for_ai

                    try:
                        current_payload = dict(base_payload)
                        current_payload["messages"] = messages_for_ai
                        current_payload["stream"] = True
                        
                        if tools:
                            current_payload["tools"] = tools
                            current_payload["tool_choice"] = "required"
                            current_payload["temperature"] = 0.1  
                            current_payload["top_p"] = 0.1  
                            current_payload["do_sample"] = False  

                        final_content = await AIResponseStrategy.direct_stream(api_key, current_payload, options, chat_log, self.entity_id, self._transform_stream, self.llm_api, self._handle_tool_call)
                        
                        if final_content:
                            filtered_content = self._filter_response_content(final_content)
                            await self._update_response(filtered_content)
                            intent_response = intent.IntentResponse(language=user_input.language)
                            intent_response.async_set_speech(filtered_content)
                            
                            return conversation.ConversationResult(
                                response=intent_response,
                                conversation_id=conversation_id,
                                continue_conversation=chat_log.continue_conversation,
                            )
                        
                        
                    except Exception:

                        pass

                    current_payload = dict(base_payload)
                    current_payload["messages"] = messages_for_ai
                    current_payload["stream"] = True

                    try:
                        
                        ai_content, tool_calls_from_ai = await AIResponseStrategy.collect_stream(api_key, current_payload, options)
                        if ai_content:
                            filtered_content = self._filter_response_content(ai_content)
                            await self._update_response(filtered_content)
                        
                        ai_history_message = ChatCompletionMessageParam(role="assistant", content=ai_content)
                        if tool_calls_from_ai: ai_history_message["tool_calls"] = tool_calls_from_ai
                        messages_for_ai.append(ai_history_message)
                        
                        if tool_calls_from_ai:
                            final_response = await self._handle_tool_calls_parallel(tool_calls_from_ai, ai_content, messages_for_ai, user_input, base_payload, api_key, options)
                            intent_response = intent.IntentResponse(language=user_input.language)
                            intent_response.async_set_speech(final_response)
                            if hasattr(self, 'session_histories') and conversation_id:
                                self.session_histories[conversation_id] = messages_for_ai
                            return conversation.ConversationResult(response=intent_response, conversation_id=conversation_id, continue_conversation=chat_log.continue_conversation)
                        else:
                            filtered_content = self._filter_response_content(ai_content)
                            intent_response = intent.IntentResponse(language=user_input.language)
                            intent_response.async_set_speech(filtered_content)
                            return conversation.ConversationResult(response=intent_response, conversation_id=conversation_id, continue_conversation=chat_log.continue_conversation)

                    except Exception as e:
                        error_text = str(e)
                        intent_response = intent.IntentResponse(language=user_input.language)
                        intent_response.async_set_speech(f"{error_text}")
                        await self._update_response(error_text)
                        return conversation.ConversationResult(response=intent_response, conversation_id=conversation_id)

                except template.TemplateError as err:
                    content_message = f"抱歉，Jinja2 模板解析出错，请检查配置模板: {err}"
                    filtered_content = self._filter_response_content(content_message)
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, filtered_content)
                    await self._update_response(filtered_content)
                    return conversation.ConversationResult(response=intent_response, conversation_id=conversation_id)
        except Exception as err:
            
            error_result = await self._handle_exception(err, user_input.text, user_input.conversation_id)
            error_text = error_result.get("message", f"处理错误: {str(err)}")
            
            
            await self._update_response(error_text)
            
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(error_text)
            
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id or ulid.ulid_now()
            )

    async def _add_history_analysis(self, prompt_parts: list, entities: list) -> None:
        try:
            if (self.entry.options.get(CONF_LLM_HASS_API) and self.entry.options.get(CONF_LLM_HASS_API) != "none" and 
                self.entry.options.get(CONF_HISTORY_ANALYSIS)):
                if entities:
                    now = datetime.now()
                    days = self.entry.options.get(CONF_HISTORY_DAYS, DEFAULT_HISTORY_DAYS)
                    interval_seconds = self.entry.options.get(CONF_HISTORY_INTERVAL, DEFAULT_HISTORY_INTERVAL) * 60
                    history_text = [f"以下是询问者所关注的实体的历史数据分析（{days}天内）："]
                    history_data = await get_instance(self.hass).async_add_executor_job(
                        get_significant_states, self.hass, now - timedelta(days=days), now,
                        entities, None, True, True)

                def process_states(states, current_state):
                    return ([f"- {current_state.state if current_state else 'unknown'} ({current_state.last_updated.astimezone().strftime('%m-%d %H:%M:%S') if current_state else 'unknown'})"] if not states else
                            [f"- {state} ({time.strftime('%m-%d %H:%M:%S')})" for state, time, _ in sorted(
                                ((s.state, s.last_updated.astimezone(), i) for i, s in enumerate(states) if s.state != "unavailable"),
                                key=lambda x: x[1]) if not _ or time.timestamp() - states[_-1].last_updated.timestamp() >= interval_seconds])

                for entity_id in entities:
                    current_state = self.hass.states.get(entity_id)
                    states = history_data.get(entity_id, [])
                    history_text.append(f"{entity_id} ({('历史状态变化' if states else '当前状态')}):")
                    history_text.extend(process_states(states, current_state))
                
                if len(history_text) > 1:
                    prompt_parts.append({"type": "history_analysis", "content": history_text})
        except Exception: pass

    async def _handle_tool_call(self, tool_input: llm.ToolInput, user_text):
        try:
            
            LOGGER.debug("处理工具调用: %s, 参数: %s", 
                       tool_input.tool_name, 
                       json.dumps(tool_input.tool_args, ensure_ascii=False))
            
            if not self.llm_api or not hasattr(self.llm_api, "async_call_tool"):
                error_msg = "LLM API未初始化"
                self._last_error_message = error_msg
                return {"success": False, "message": error_msg}
            
            
            if isinstance(tool_input.tool_args, str):
                try: 
                    tool_args_dict = json.loads(tool_input.tool_args)
                except json.JSONDecodeError as json_err: 
                    error_msg = str(json_err)
                    self._last_error_message = error_msg
                    tool_args_dict = {"text": tool_input.tool_args}
                tool_input = llm.ToolInput(id=tool_input.id, tool_name=tool_input.tool_name, tool_args=tool_args_dict)
            
            try:
                
                result = await self.llm_api.async_call_tool(tool_input)
                return result if isinstance(result, dict) else {"success": True, "result": str(result)}
            except Exception as e:
                
                error_msg = str(e)
                self._last_error_message = error_msg
                LOGGER.error("工具调用直接失败: %s", error_msg)
                return {"success": False, "message": error_msg}
        except Exception as e:
            
            error_msg = str(e)
            self._last_error_message = error_msg
            return {"success": False, "message": error_msg}

    async def _fallback_to_hass_llm(self, user_input: conversation.ConversationInput, conversation_id: str) -> conversation.ConversationResult:
        try:
            agent = await conversation.async_get_agent(self.hass)
            if agent:
                try:
                    if hasattr(agent, 'async_process') and asyncio.iscoroutinefunction(agent.async_process):
                        result = await agent.async_process(user_input)
                        return result
                    else:
                        intent_response = intent.IntentResponse(language=user_input.language)
                        intent_response.async_set_speech(f"处理请求时发生错误: {str(err)}")
                        return conversation.ConversationResult(
                            response=intent_response,
                            conversation_id=conversation_id
                        )
                except Exception as err:
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_speech(f"处理请求时发生错误: {str(err)}")
                    return conversation.ConversationResult(
                        response=intent_response,
                        conversation_id=conversation_id
                    )
            else:
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    "没有可用的对话代理"
                )
                return conversation.ConversationResult(
                    response=intent_response,
                    conversation_id=conversation_id
                )
        except Exception as err:
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"{str(err)}"
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=conversation_id
            )

    @staticmethod
    async def _async_entry_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
        entity = hass.data[DOMAIN].get(entry.entry_id)
        if entity:
            entity.entry = entry

    async def _transform_stream(
        self, stream: AsyncGenerator[dict, None], llm_api: llm.AbstractLLMApi
    ) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]:
        is_first = True
        collected_content = ""
        collected_tool_calls = []
        tool_call_fragments = {}  
        
        try:
            async for chunk in stream:
                chunk = chunk.get("data", chunk) if "data" in chunk else chunk
                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                
                if is_first and "role" in delta:
                    yield {"role": delta["role"]}
                    is_first = False
                
                if "content" in delta and delta["content"] is not None:
                    collected_content += delta["content"]
                    yield {"content": delta["content"]}
                
                if "tool_calls" in delta and delta["tool_calls"]:
                    for tool_call in delta["tool_calls"]:
                        call_id = tool_call.get("id")
                        if not call_id:
                            continue
                        
                        if call_id not in tool_call_fragments:
                            tool_call_fragments[call_id] = {
                                "id": call_id,
                                "type": tool_call.get("type", "function"),
                                "function": {
                                    "name": tool_call.get("function", {}).get("name", ""),
                                    "arguments": tool_call.get("function", {}).get("arguments", "")
                                }
                            }
                        else:
                            fragment = tool_call_fragments[call_id]
                            if tool_call.get("function", {}).get("name"):
                                fragment["function"]["name"] = tool_call["function"]["name"]
                            if tool_call.get("function", {}).get("arguments"):
                                fragment["function"]["arguments"] += tool_call["function"]["arguments"]
                        
                        try:
                            fragment = tool_call_fragments[call_id]
                            if fragment["function"]["name"] and fragment["function"]["arguments"]:
                                args_str = fragment["function"]["arguments"].strip()
                                if args_str.endswith("}"): 
                                    try:
                                        json.loads(args_str)  
                                        
                                        if not any(c.get("id") == call_id for c in collected_tool_calls):
                                            collected_tool_calls.append(fragment.copy())
                                    except json.JSONDecodeError:
                                        pass  
                        except Exception as e:
                            LOGGER.debug("解析工具调用时出错: %s", str(e))
                
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    if collected_tool_calls:
                        for call_id, fragment in tool_call_fragments.items():
                            if not any(c.get("id") == call_id for c in collected_tool_calls):
                                try:
                                    args_str = fragment["function"]["arguments"].strip()
                                    json.loads(args_str)  
                                    collected_tool_calls.append(fragment.copy())
                                except (json.JSONDecodeError, Exception):
                                    pass  
                                
                        yield {"collected_tool_calls": collected_tool_calls, "content": collected_content}
                        return
                    elif finish_reason in ["stop", "length"]:
                        break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if collected_tool_calls:
                yield {"collected_tool_calls": collected_tool_calls, "content": collected_content, "error": str(e)}
            else:
                yield {"content": collected_content, "error": str(e)}

    async def _update_response(self, content: str) -> None:
        if content and isinstance(content, str):
            self._attr_response = content
            self._attr_extra_state_attributes["response"] = content
            
            self.async_write_ha_state()

    
    async def _handle_exception(self, exception: Exception, user_input: str, conversation_id: str = None, tool_name: str = None) -> Dict[str, Any]:
        if self.llm_api and hasattr(self.llm_api, "async_generate_text"):
            try:
                ai_response = await self.llm_api.async_generate_text(f"解释这个错误: {str(exception)}")
                return {"success": False, "message": ai_response or "操作执行失败"}
            except Exception:
                pass
        return {"success": False, "message": str(exception)}

    async def _handle_tool_calls_parallel(self, tool_calls, ai_content, messages_for_ai, user_input, base_payload, api_key, options):
        processor = ToolCallProcessor(self, api_key, options)
        return await processor.process(
            tool_calls, 
            ai_content, 
            messages_for_ai,
            user_input, 
            base_payload
        )

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    # 遍历subentries，注册conversation类型实体
    for subentry in getattr(config_entry, "subentries", {}).values():
        if getattr(subentry, "subentry_type", None) == "conversation":
            async_add_entities([ZhipuAIConversationEntity(config_entry, subentry, hass)], config_subentry_id=subentry.subentry_id)

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, ["conversation"]):
        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unload_ok

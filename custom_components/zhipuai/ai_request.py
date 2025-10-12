from __future__ import annotations
import aiohttp
import json
import asyncio
import time
from typing import AsyncGenerator, Dict, Any, Protocol, Callable
from homeassistant.exceptions import HomeAssistantError
from homeassistant.core import HomeAssistant
from .const import (
    LOGGER, ZHIPUAI_URL, CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT,
    ERROR_INVALID_AUTH, ERROR_TOO_MANY_REQUESTS, ERROR_SERVER_ERROR, ERROR_TIMEOUT, ERROR_UNKNOWN
)

_SESSION = None

async def get_session() -> aiohttp.ClientSession:
    global _SESSION
    if _SESSION is None:
        connector = aiohttp.TCPConnector(
            limit=10,             
            limit_per_host=5,      
            keepalive_timeout=20.0
        )
        _SESSION = aiohttp.ClientSession(connector=connector)
    return _SESSION

class AIRequestHandler(Protocol):
    async def send_request(self, api_key: str, payload: Dict[str, Any], options: Dict[str, Any]) -> Any: pass
    async def handle_tool_call(self, api_key: str, tool_call: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]: pass

def _handle_error_status(status: int, error_text: str) -> None:
    if status == 401:
        raise HomeAssistantError(ERROR_INVALID_AUTH)
    elif status == 429:
        raise HomeAssistantError(ERROR_TOO_MANY_REQUESTS)
    elif status in [500, 502, 503, 504]:
        raise HomeAssistantError(ERROR_SERVER_ERROR)
    else:
        raise HomeAssistantError(f"{ERROR_UNKNOWN}: {error_text}")

class StreamingRequestHandler:
    async def send_request(self, api_key: str, payload: Dict[str, Any], options: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """简化的流式请求处理器 - 学习Go代码的SSE流处理方式"""
        payload["stream"] = True
        if "request_id" not in payload:
            payload["request_id"] = f"req_{int(time.time() * 1000)}"
        
        LOGGER.info("发送给AI的消息: %s", json.dumps(payload, ensure_ascii=False))

        try:
            session = await get_session()
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            timeout = aiohttp.ClientTimeout(total=options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT))
            api_url = options.get("base_url", ZHIPUAI_URL)
            
            async with session.post(api_url, json=payload, headers=headers, timeout=timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    _handle_error_status(response.status, error_text)

                # 学习Go代码的SSE流处理方式
                buffer = ""
                async for chunk in response.content:
                    if not chunk:
                        continue
                        
                    chunk_text = chunk.decode('utf-8', errors='ignore')
                    buffer += chunk_text
                    
                    # 处理完整的行
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        # 跳过空行和结束标记
                        if not line or line == "data: [DONE]":
                            continue
                        
                        # 处理SSE数据行
                        if line.startswith("data: "):
                            data_str = line[6:]  # 移除 "data: " 前缀
                            if data_str.strip():
                                try:
                                    data = json.loads(data_str)
                                    yield data
                                except json.JSONDecodeError:
                                    LOGGER.debug("SSE数据解析失败: %s", data_str)
                                    continue
                
                # 处理剩余的buffer
                if buffer.strip() and buffer.startswith("data: "):
                    data_str = buffer[6:].strip()
                    if data_str and data_str != "[DONE]":
                        try:
                            data = json.loads(data_str)
                            yield data
                        except json.JSONDecodeError:
                            LOGGER.debug("剩余buffer解析失败: %s", data_str)
                    
        except (aiohttp.ClientConnectorError, aiohttp.ServerTimeoutError, aiohttp.ClientOSError, asyncio.TimeoutError) as e:
            LOGGER.error("网络请求错误: %s", str(e))
            raise HomeAssistantError(f"{ERROR_UNKNOWN}: {str(e)}")
        except Exception as e:
            LOGGER.error("流式请求处理错误: %s", str(e))
            raise HomeAssistantError(f"{ERROR_UNKNOWN}: {str(e)}")

    async def handle_tool_call(self, api_key: str, tool_call: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        for attempt in range(3):
            try:
                session = await get_session()
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                timeout = aiohttp.ClientTimeout(total=options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT))
                api_url = options.get("base_url", ZHIPUAI_URL)
                tool_url = f"{api_url}/tool_calls"
                
                payload = {
                    "tool_call_id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"],
                    "stream": False
                }
                
                async with session.post(tool_url, json=payload, headers=headers, timeout=timeout) as response:
                    if response.status != 200:
                        _handle_error_status(response.status, await response.text())
                    return await response.json()
                    
            except (aiohttp.ClientConnectorError, aiohttp.ServerTimeoutError, aiohttp.ClientOSError, asyncio.TimeoutError):
                if attempt < 2: 
                    await asyncio.sleep(1.0)
            except Exception as e:
                raise HomeAssistantError(f"{ERROR_UNKNOWN}: {str(e)}")
                
        raise HomeAssistantError(f"{ERROR_UNKNOWN}")

class DirectRequestHandler:
    async def send_request(self, api_key: str, payload: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        for attempt in range(3):
            try:
                session = await get_session()
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                timeout = aiohttp.ClientTimeout(total=options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT))
                payload_copy = dict(payload)
                payload_copy["stream"] = False
                
                api_url = options.get("base_url", ZHIPUAI_URL)
                async with session.post(api_url, json=payload_copy, headers=headers, timeout=timeout) as response:
                    response_json = await response.json()
                    if response.status != 200:
                        _handle_error_status(response.status, str(response_json))
                    return response_json
                    
            except (aiohttp.ClientConnectorError, aiohttp.ServerTimeoutError, aiohttp.ClientOSError, asyncio.TimeoutError):
                if attempt < 2: 
                    await asyncio.sleep(1.0)
            except Exception as e:
                raise HomeAssistantError(f"{ERROR_UNKNOWN}: {str(e)}")
                
        raise HomeAssistantError(f"{ERROR_UNKNOWN}")

    async def handle_tool_call(self, api_key: str, tool_call: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        return await StreamingRequestHandler().handle_tool_call(api_key, tool_call, options)

def create_request_handler(streaming: bool = True) -> AIRequestHandler:
    return StreamingRequestHandler() if streaming else DirectRequestHandler()

async def send_ai_request(api_key: str, payload: Dict[str, Any], options: Dict[str, Any] = None, timeout=30) -> AsyncGenerator[Dict[str, Any], None]:
    options = options or {}
    handler = create_request_handler(True)
    async for chunk in handler.send_request(api_key, payload, options):
        yield chunk

async def send_api_request(api_key: str, payload: Dict[str, Any], options: Dict[str, Any] = None, timeout=30) -> Dict[str, Any]:
    options = options or {}
    handler = create_request_handler(False)
    return await handler.send_request(api_key, payload, options)

async def handle_tool_call(api_key: str, tool_call: Dict[str, Any], options: Dict[str, Any] = None) -> Dict[str, Any]:
    options = options or {}
    handler = create_request_handler(False)
    return await handler.handle_tool_call(api_key, tool_call, options)
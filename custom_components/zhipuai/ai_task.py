from __future__ import annotations
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List
from collections import defaultdict
from enum import IntFlag

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.entity import Entity
from homeassistant.const import CONF_API_KEY
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
import homeassistant.helpers.config_validation as cv
import voluptuous as vol

from .const import (
    DOMAIN, 
    LOGGER, 
)
from .optimizations import ParallelToolExecutor, ContextCompressor, SmartRetryHandler

DEFAULT_AI_TASK_NAME = "default_ai_task"
CONF_CHAT_MODEL = "chat_model"
CONF_TEMPERATURE = "temperature"
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_CHAT_MODEL = "gpt-4"
RECOMMENDED_TEMPERATURE = 0.7
RECOMMENDED_MAX_TOKENS = 1500
from .ai_request import send_ai_request, send_api_request


class TaskCache:
    """任务结果缓存管理器"""
    
    def __init__(self, max_size: int = 100, ttl: int = 300):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._ttl = ttl  # 缓存过期时间(秒)
        self._access_count: Dict[str, int] = defaultdict(int)
        self._hit_count = 0
        self._miss_count = 0
    
    def _generate_key(self, task_type: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        param_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
        return f"{task_type}:{param_str}"
    
    def get(self, task_type: str, params: Dict[str, Any]) -> Optional[Any]:
        """获取缓存结果"""
        key = self._generate_key(task_type, params)
        
        if key in self._cache:
            cache_entry = self._cache[key]
            # 检查是否过期
            if time.time() - cache_entry["timestamp"] < self._ttl:
                self._access_count[key] += 1
                self._hit_count += 1
                LOGGER.debug(f"缓存命中: {key}")
                return cache_entry["result"]
            else:
                # 过期删除
                del self._cache[key]
                del self._access_count[key]
        
        self._miss_count += 1
        return None
    
    def set(self, task_type: str, params: Dict[str, Any], result: Any) -> None:
        """设置缓存"""
        key = self._generate_key(task_type, params)
        
        # 如果缓存已满，删除最少访问的项
        if len(self._cache) >= self._max_size:
            least_used_key = min(self._access_count, key=self._access_count.get)
            del self._cache[least_used_key]
            del self._access_count[least_used_key]
        
        self._cache[key] = {
            "result": result,
            "timestamp": time.time()
        }
        self._access_count[key] = 0
        LOGGER.debug(f"缓存设置: {key}")
    
    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._access_count.clear()
        self._hit_count = 0
        self._miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total * 100) if total > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": f"{hit_rate:.2f}%",
            "ttl": self._ttl
        }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._max_records = 100
    
    def record(self, operation: str, duration: float) -> None:
        """记录操作耗时"""
        if len(self._metrics[operation]) >= self._max_records:
            self._metrics[operation].pop(0)
        self._metrics[operation].append(duration)
        
        LOGGER.debug(f"性能记录: {operation} 耗时 {duration:.3f}秒")
    
    def get_stats(self, operation: str = None) -> Dict[str, Any]:
        """获取性能统计"""
        if operation:
            if operation not in self._metrics:
                return {}
            
            durations = self._metrics[operation]
            return {
                "operation": operation,
                "count": len(durations),
                "avg": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "total": sum(durations)
            }
        
        # 返回所有操作的统计
        stats = {}
        for op, durations in self._metrics.items():
            stats[op] = {
                "count": len(durations),
                "avg": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "total": sum(durations)
            }
        return stats
    
    def clear(self) -> None:
        """清空统计"""
        self._metrics.clear()


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    """Set up ai_task platform from a config entry."""
    try:
        LOGGER.debug("Setting up ai_task platform for entry %s", entry.entry_id)
        
        # 检查entry是否包含subentries
        if not hasattr(entry, "subentries"):
            LOGGER.error("Entry does not have subentries")
            return
            
        # 遍历subentries，注册ai_task_data类型实体
        for subentry in entry.subentries.values():
            if not hasattr(subentry, "subentry_type"):
                LOGGER.error("Subentry missing subentry_type")
                continue
                
            if subentry.subentry_type != "ai_task_data":
                LOGGER.debug("Skipping subentry with type: %s", subentry.subentry_type)
                continue

            LOGGER.debug("Creating AITaskEntity for subentry %s", subentry.subentry_id)
            entity = AITaskEntity(entry, subentry, hass)
            
            async_add_entities(
                [entity],
                config_subentry_id=subentry.subentry_id,
            )
            
            # 注册服务
            await async_setup_services(hass, entity)
            
            LOGGER.debug("Entity added successfully for subentry %s", subentry.subentry_id)
            
    except Exception as err:
        LOGGER.exception("Error setting up ai_task platform: %s", err)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    try:
        # 清理子条目
        if hasattr(entry, "subentries"):
            for subentry_id in list(entry.subentries.keys()):
                if getattr(entry.subentries[subentry_id], "subentry_type", None) == "ai_task_data":
                    del entry.subentries[subentry_id]
                    LOGGER.debug("已移除AI任务子条目: %s", subentry_id)
    except Exception as err:
        LOGGER.exception("卸载AI任务平台时出错: %s", err)
    
    return True


async def async_setup_services(hass: HomeAssistant, entity: AITaskEntity) -> None:
    """注册AI任务服务"""
    
    async def handle_execute_task(call: ServiceCall) -> Dict[str, Any]:
        """处理执行任务服务"""
        return await entity.async_execute_task(
            task_type=call.data.get("task_type"),
            params=call.data.get("params", {}),
            use_cache=call.data.get("use_cache", True)
        )
    
    async def handle_get_cache_stats(call: ServiceCall) -> Dict[str, Any]:
        """获取缓存统计"""
        return entity.get_cache_stats()
    
    async def handle_get_performance_stats(call: ServiceCall) -> Dict[str, Any]:
        """获取性能统计"""
        operation = call.data.get("operation")
        return entity.get_performance_stats(operation)
    
    async def handle_clear_cache(call: ServiceCall) -> None:
        """清空缓存"""
        entity.clear_cache()
    
    # 注册服务
    hass.services.async_register(
        DOMAIN,
        "execute_task",
        handle_execute_task,
        schema=vol.Schema({
            vol.Required("task_type"): cv.string,
            vol.Optional("params", default={}): dict,
            vol.Optional("use_cache", default=True): cv.boolean,
        })
    )
    
    hass.services.async_register(
        DOMAIN,
        "get_cache_stats",
        handle_get_cache_stats,
        schema=vol.Schema({})
    )
    
    hass.services.async_register(
        DOMAIN,
        "get_performance_stats",
        handle_get_performance_stats,
        schema=vol.Schema({
            vol.Optional("operation"): cv.string,
        })
    )
    
    hass.services.async_register(
        DOMAIN,
        "clear_cache",
        handle_clear_cache,
        schema=vol.Schema({})
    )


class AITaskEntityFeature(IntFlag):
    """AI任务实体支持的功能"""
    GENERATE_DATA = 1
    SUPPORT_ATTACHMENTS = 2
    GENERATE_IMAGE = 4

    def __contains__(self, item):
        """AI任务实体支持的功能标志"""
        return bool(self & item)


class AITaskEntity(Entity):
    """AI任务实体 - 支持生成数据和图片"""

    _attr_should_poll = False

    def __init__(self, entry: ConfigEntry, subentry: ConfigEntry, hass: HomeAssistant) -> None:
        """初始化实体，仅保留最基本的功能。"""
        self._current_task = None
        self._entry = entry
        self._subentry = subentry
        self._hass = hass
        self._attr_name = subentry.title or DEFAULT_AI_TASK_NAME
        
        # unique_id和identifiers归属于subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_entity_id = f"ai_task.{DOMAIN}_{subentry.subentry_id}"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="北京智谱华章科技",
            model="AI Task Manager",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        
        # 支持的功能
        self._attr_supported_features = (
            AITaskEntityFeature.GENERATE_DATA
            | AITaskEntityFeature.SUPPORT_ATTACHMENTS
            | AITaskEntityFeature.GENERATE_IMAGE
        )
        
        # 初始化状态
        self._state = "idle"

    @property
    def name(self) -> str:
        return self._attr_name

    @property
    def unique_id(self) -> str:
        return self._attr_unique_id

    @property
    def state(self) -> str:
        return self._state

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """返回额外状态属性"""
        attrs = {
            "current_task": self._current_task,
            "task_count": len(self._task_history),
            "cache_stats": self._cache.get_stats(),
            "performance_stats": self._performance.get_stats(),
        }
        
        # 添加最近的任务历史
        if self._task_history:
            attrs["recent_tasks"] = self._task_history[-5:]
        
        return attrs

    async def async_execute_task(
        self, 
        task_type: str, 
        params: Dict[str, Any],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """执行AI任务"""
        start_time = time.time()
        task_id = f"task_{int(time.time() * 1000)}"
        
        try:
            # 更新状态
            self._state = "processing"
            self._current_task = {
                "id": task_id,
                "type": task_type,
                "params": params,
                "start_time": datetime.now().isoformat(),
                "status": "running"
            }
            self.async_write_ha_state()
            
            # 检查缓存
            if use_cache:
                cached_result = self._cache.get(task_type, params)
                if cached_result is not None:
                    duration = time.time() - start_time
                    self._performance.record(f"{task_type}_cached", duration)
                    
                    result = {
                        "success": True,
                        "task_id": task_id,
                        "result": cached_result,
                        "cached": True,
                        "duration": duration
                    }
                    
                    self._add_to_history(task_id, task_type, params, result, duration)
                    self._state = "idle"
                    self._current_task = None
                    self.async_write_ha_state()
                    
                    return result
            
            # 执行任务
            result = await self._execute_task_internal(task_type, params)
            
            # 缓存结果
            if use_cache and result.get("success"):
                self._cache.set(task_type, params, result.get("result"))
            
            duration = time.time() - start_time
            self._performance.record(task_type, duration)
            
            # 添加到历史
            self._add_to_history(task_id, task_type, params, result, duration)
            
            # 更新状态
            self._state = "idle"
            self._current_task = None
            self.async_write_ha_state()
            
            return {
                "success": result.get("success", False),
                "task_id": task_id,
                "result": result.get("result"),
                "cached": False,
                "duration": duration
            }
            
        except Exception as e:
            duration = time.time() - start_time
            LOGGER.error(f"任务执行失败: {task_type}, 错误: {str(e)}")
            
            error_result = {
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "duration": duration
            }
            
            self._add_to_history(task_id, task_type, params, error_result, duration)
            self._state = "error"
            self._current_task = None
            self.async_write_ha_state()
            
            return error_result

    async def _execute_task_internal(self, task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """内部任务执行逻辑"""
        
        # 获取API配置
        api_key = self._entry.data.get(CONF_API_KEY)
        if not api_key:
            raise HomeAssistantError("未找到API密钥")
        
        # 根据任务类型执行不同的处理
        if task_type == "generate_data":
            return await self._async_generate_data(params)
        
        elif task_type == "generate_image":
            from .image_gen import async_generate_image
            return await async_generate_image(hass=self._hass, **params)
        
        else:
            raise HomeAssistantError(f"不支持的任务类型: {task_type}")

    async def _async_generate_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理生成数据的任务"""
        try:
            # 调用API生成数据
            response = await send_api_request(
                self._entry.data.get(CONF_API_KEY),
                {"prompt": params.get("prompt")},
                self._subentry.options
            )
            
            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                return {"success": True, "result": content}
            
            return {"success": False, "error": "未获取到有效响应"}
        except Exception as e:
            LOGGER.error(f"生成数据失败: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _async_generate_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理生成图像的任务"""
        import aiohttp
        import os
        
        try:
            # 获取参数
            prompt = params.get("prompt")
            if not prompt:
                return {"success": False, "error": "缺少prompt参数"}
            
            model = params.get("model", "cogview-3-flash")
            size = params.get("size", "1024x1024")
            api_key = self._entry.data.get(CONF_API_KEY)
            
            if not api_key:
                return {"success": False, "error": "未找到API密钥"}
            
            # 准备请求
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "prompt": prompt,
                "size": size
            }
            
            # 调用智谱AI图片生成API
            from .const import ZHIPUAI_IMAGE_GEN_URL, DOMAIN
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    ZHIPUAI_IMAGE_GEN_URL,
                    headers=headers,
                    json=payload,
                    timeout=300
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
            
            if not result.get("data") or not result["data"][0].get("url"):
                return {"success": False, "error": "API未返回有效的图片URL"}
            
            image_url = result["data"][0]["url"]
            
            # 下载图片到本地
            www_dir = self._hass.config.path("www")
            img_dir = os.path.join(www_dir, "ai_task_img")
            if not os.path.exists(img_dir):
                os.makedirs(img_dir, exist_ok=True)
            
            filename = os.path.basename(image_url.split('?')[0]) or 'ai_task_sc.png'
            local_path = os.path.join(img_dir, filename)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        content = await response.content.read()
                        with open(local_path, "wb") as f:
                            f.write(content)
            
            local_url = f"/local/ai_task_img/{filename}"
            
            # 触发事件
            self._hass.bus.async_fire(f"ai_task_response", {
                "type": "image_gen",
                "content": local_url,
                "success": True
            })
            
            return {
                "success": True,
                "result": {
                    "local_url": local_url,
                    "original_url": image_url,
                    "model": model,
                    "size": size,
                    "domain": DOMAIN
                }
            }
            
        except aiohttp.ClientError as e:
            LOGGER.error(f"API请求失败: {str(e)}")
            return {"success": False, "error": f"API请求失败: {str(e)}"}
        except Exception as e:
            LOGGER.error(f"生成图像失败: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _handle_text_generation(self, api_key: str, options: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """处理文本生成任务"""
        prompt = params.get("prompt")
        if not prompt:
            raise HomeAssistantError("缺少prompt参数")
        
        payload = {
            "model": params.get("model", options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.get("temperature", options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)),
            "max_tokens": params.get("max_tokens", options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)),
            "stream": False
        }
        
        response = await send_api_request(api_key, payload, options)
        
        if response and "choices" in response:
            content = response["choices"][0]["message"]["content"]
            return {
                "success": True,
                "result": content,
                "usage": response.get("usage", {})
            }
        
        return {"success": False, "error": "未获取到有效响应"}

    async def _handle_text_analysis(self, api_key: str, options: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """处理文本分析任务"""
        text = params.get("text")
        analysis_type = params.get("analysis_type", "summary")
        
        if not text:
            raise HomeAssistantError("缺少text参数")
        
        prompts = {
            "summary": f"请总结以下内容:\n{text}",
            "sentiment": f"请分析以下内容的情感倾向:\n{text}",
            "keywords": f"请提取以下内容的关键词:\n{text}",
            "classification": f"请对以下内容进行分类:\n{text}"
        }
        
        prompt = prompts.get(analysis_type, f"请分析以下内容:\n{text}")
        
        payload = {
            "model": options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "stream": False
        }
        
        response = await send_api_request(api_key, payload, options)
        
        if response and "choices" in response:
            content = response["choices"][0]["message"]["content"]
            return {
                "success": True,
                "result": content,
                "analysis_type": analysis_type
            }
        
        return {"success": False, "error": "分析失败"}

    async def _handle_entity_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理实体查询任务"""
        domain = params.get("domain")
        entity_id = params.get("entity_id")
        
        if entity_id:
            state = self._hass.states.get(entity_id)
            if state:
                return {
                    "success": True,
                    "result": {
                        "entity_id": state.entity_id,
                        "state": state.state,
                        "attributes": dict(state.attributes)
                    }
                }
            return {"success": False, "error": f"实体不存在: {entity_id}"}
        
        elif domain:
            entities = []
            for entity_id in self._hass.states.async_entity_ids(domain):
                state = self._hass.states.get(entity_id)
                if state:
                    entities.append({
                        "entity_id": state.entity_id,
                        "state": state.state,
                        "name": state.attributes.get("friendly_name", entity_id)
                    })
            
            return {
                "success": True,
                "result": {
                    "domain": domain,
                    "count": len(entities),
                    "entities": entities
                }
            }
        
        return {"success": False, "error": "缺少domain或entity_id参数"}

    async def _handle_batch_query(self, api_key: str, options: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """处理批量查询任务"""
        queries = params.get("queries", [])
        if not queries:
            raise HomeAssistantError("缺少queries参数")
        
        results = []
        for query in queries:
            try:
                result = await self._handle_text_generation(api_key, options, {"prompt": query})
                results.append({
                    "query": query,
                    "success": result.get("success", False),
                    "result": result.get("result")
                })
            except Exception as e:
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "result": {
                "total": len(queries),
                "results": results
            }
        }

    async def _handle_parallel_tasks(self, api_key: str, options: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """处理并行任务 - 优化点：无依赖任务并行执行"""
        tasks = params.get("tasks", [])
        if not tasks:
            raise HomeAssistantError("缺少tasks参数")
        
        # 创建并行任务
        async_tasks = []
        for task in tasks:
            task_type = task.get("type")
            task_params = task.get("params", {})
            
            if task_type == "text_generation":
                async_tasks.append(self._handle_text_generation(api_key, options, task_params))
            elif task_type == "entity_query":
                async_tasks.append(self._handle_entity_query(task_params))
        
        # 并行执行
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # 处理结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "task_index": i,
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append({
                    "task_index": i,
                    "success": result.get("success", False),
                    "result": result.get("result")
                })
        
        return {
            "success": True,
            "result": {
                "total": len(tasks),
                "results": processed_results
            }
        }

    def _add_to_history(self, task_id: str, task_type: str, params: Dict[str, Any], result: Dict[str, Any], duration: float) -> None:
        """添加任务到历史记录"""
        if len(self._task_history) >= self._max_history:
            self._task_history.pop(0)
        
        self._task_history.append({
            "id": task_id,
            "type": task_type,
            "params": params,
            "result": result,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self._cache.get_stats()

    async def async_update(self) -> None:
        """更新实体状态"""
        pass
    
    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """返回额外状态属性"""
        return {"current_task": self._current_task}

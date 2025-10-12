"""
优化模块 - 提供并行工具调用、上下文压缩等性能优化功能
"""
from __future__ import annotations
import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from .const import LOGGER


class ParallelToolExecutor:
    """并行工具执行器 - 优化无依赖工具的并行执行"""
    
    def __init__(self, entity, max_parallel: int = 5):
        self.entity = entity
        self.max_parallel = max_parallel
        self._execution_stats = defaultdict(list)
    
    def analyze_dependencies(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        分析工具调用之间的依赖关系
        返回: {"independent": [...], "dependent": [...]}
        """
        independent = []
        dependent = []
        
        # 简单的依赖分析：检查工具参数中是否引用了其他工具的结果
        tool_names = {call.get("function", {}).get("name") for call in tool_calls}
        
        for call in tool_calls:
            args_str = json.dumps(call.get("function", {}).get("arguments", "{}"))
            
            # 检查是否依赖其他工具
            has_dependency = False
            for tool_name in tool_names:
                if tool_name != call.get("function", {}).get("name"):
                    # 简单检查：参数中是否包含其他工具名称
                    if tool_name in args_str:
                        has_dependency = True
                        break
            
            if has_dependency:
                dependent.append(call)
            else:
                independent.append(call)
        
        return {
            "independent": independent,
            "dependent": dependent
        }
    
    async def execute_parallel(
        self, 
        tool_calls: List[Dict[str, Any]], 
        user_text: str
    ) -> List[Dict[str, Any]]:
        """
        并行执行独立的工具调用
        """
        if not tool_calls:
            return []
        
        # 分析依赖关系
        categorized = self.analyze_dependencies(tool_calls)
        independent_calls = categorized["independent"]
        dependent_calls = categorized["dependent"]
        
        LOGGER.info(
            f"工具调用分析: 独立={len(independent_calls)}, 依赖={len(dependent_calls)}"
        )
        
        results = []
        
        # 并行执行独立的工具调用
        if independent_calls:
            start_time = time.time()
            
            # 分批执行，避免过多并发
            for i in range(0, len(independent_calls), self.max_parallel):
                batch = independent_calls[i:i + self.max_parallel]
                batch_tasks = [
                    self._execute_single_tool(call, user_text) 
                    for call in batch
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # 处理结果
                for result in batch_results:
                    if isinstance(result, Exception):
                        LOGGER.error(f"并行工具执行异常: {str(result)}")
                        results.append({
                            "success": False,
                            "error": str(result),
                            "content": json.dumps({"error": str(result)}, ensure_ascii=False)
                        })
                    else:
                        results.append(result)
            
            duration = time.time() - start_time
            self._execution_stats["parallel_execution"].append({
                "count": len(independent_calls),
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            })
            
            LOGGER.info(
                f"并行执行 {len(independent_calls)} 个工具，耗时 {duration:.3f}秒"
            )
        
        # 串行执行有依赖的工具调用
        if dependent_calls:
            for call in dependent_calls:
                result = await self._execute_single_tool(call, user_text)
                results.append(result)
        
        return results
    
    async def _execute_single_tool(
        self, 
        tool_call: Dict[str, Any], 
        user_text: str
    ) -> Dict[str, Any]:
        """执行单个工具调用"""
        tool_name = ""
        tool_call_id = f"tool_{int(time.time() * 1000)}"
        
        try:
            tool_name = tool_call["function"].get("name", "")
            tool_args_json = tool_call["function"].get("arguments", "{}")
            tool_call_id = tool_call.get("id", tool_call_id)
            
            LOGGER.debug(f"执行工具: {tool_name}, 参数: {tool_args_json}")
            
            tool_args = json.loads(tool_args_json) if tool_args_json else {}
            
            tool_input = llm.ToolInput(
                id=tool_call_id, 
                tool_name=tool_name, 
                tool_args=tool_args
            )
            
            start_time = time.time()
            result = await self.entity._handle_tool_call(tool_input, user_text)
            duration = time.time() - start_time
            
            # 记录执行统计
            self._execution_stats[tool_name].append({
                "duration": duration,
                "success": result.get("success", True),
                "timestamp": datetime.now().isoformat()
            })
            
            result_content = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
            
            return {
                "success": result.get("success", True),
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "content": result_content,
                "duration": duration
            }
            
        except Exception as e:
            LOGGER.error(f"工具执行失败: {tool_name}, 错误: {str(e)}")
            return {
                "success": False,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "content": json.dumps({"error": str(e)}, ensure_ascii=False),
                "error": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        stats = {}
        
        for tool_name, executions in self._execution_stats.items():
            if executions:
                durations = [e["duration"] for e in executions]
                success_count = sum(1 for e in executions if e.get("success", True))
                
                stats[tool_name] = {
                    "total_executions": len(executions),
                    "success_count": success_count,
                    "failure_count": len(executions) - success_count,
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "total_duration": sum(durations)
                }
        
        return stats


class ContextCompressor:
    """上下文压缩器 - 智能压缩长对话历史"""
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self._compression_stats = []
    
    def estimate_tokens(self, text: str) -> int:
        """
        估算文本的token数量
        简单估算：中文约1.5字符/token，英文约4字符/token
        """
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        
        return int(chinese_chars / 1.5 + other_chars / 4)
    
    def compress_messages(
        self, 
        messages: List[Dict[str, Any]], 
        keep_recent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        压缩消息历史
        策略：
        1. 保留系统消息
        2. 保留最近N条消息
        3. 压缩中间的消息（提取关键信息）
        """
        if not messages:
            return []
        
        start_time = time.time()
        original_count = len(messages)
        
        # 分离不同类型的消息
        system_messages = []
        user_assistant_messages = []
        tool_messages = []
        
        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                system_messages.append(msg)
            elif role in ["user", "assistant"]:
                user_assistant_messages.append(msg)
            elif role == "tool":
                tool_messages.append(msg)
        
        # 保留最近的消息
        recent_messages = user_assistant_messages[-keep_recent:] if len(user_assistant_messages) > keep_recent else user_assistant_messages
        
        # 压缩中间的消息
        middle_messages = user_assistant_messages[:-keep_recent] if len(user_assistant_messages) > keep_recent else []
        
        compressed_middle = []
        if middle_messages:
            # 提取关键对话
            compressed_middle = self._extract_key_messages(middle_messages)
        
        # 重新组合
        compressed_messages = system_messages + compressed_middle + recent_messages
        
        # 记录统计
        duration = time.time() - start_time
        self._compression_stats.append({
            "original_count": original_count,
            "compressed_count": len(compressed_messages),
            "reduction_rate": (original_count - len(compressed_messages)) / original_count if original_count > 0 else 0,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
        
        LOGGER.info(
            f"上下文压缩: {original_count} -> {len(compressed_messages)} 条消息, "
            f"耗时 {duration:.3f}秒"
        )
        
        return compressed_messages
    
    def _extract_key_messages(
        self, 
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        从消息列表中提取关键消息
        策略：
        1. 保留包含重要关键词的消息
        2. 保留较长的消息（可能包含更多信息）
        3. 保留问答对
        """
        key_messages = []
        important_keywords = [
            "错误", "失败", "成功", "完成", "设置", "配置", 
            "控制", "查询", "状态", "问题", "解决"
        ]
        
        i = 0
        while i < len(messages):
            msg = messages[i]
            content = msg.get("content", "")
            
            # 检查是否包含重要关键词
            has_keyword = any(keyword in content for keyword in important_keywords)
            
            # 检查消息长度
            is_long = len(content) > 50
            
            # 检查是否是问答对
            is_qa_pair = False
            if i + 1 < len(messages):
                next_msg = messages[i + 1]
                if msg.get("role") == "user" and next_msg.get("role") == "assistant":
                    is_qa_pair = True
            
            if has_keyword or is_long or is_qa_pair:
                key_messages.append(msg)
                if is_qa_pair and i + 1 < len(messages):
                    key_messages.append(messages[i + 1])
                    i += 2
                    continue
            
            i += 1
        
        # 如果压缩后仍然太多，进一步压缩
        if len(key_messages) > 10:
            # 只保留最重要的消息
            key_messages = key_messages[:5] + key_messages[-5:]
        
        return key_messages
    
    def smart_compress(
        self, 
        messages: List[Dict[str, Any]], 
        target_tokens: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        智能压缩到目标token数
        """
        if target_tokens is None:
            target_tokens = self.max_tokens
        
        # 估算当前token数
        current_tokens = sum(
            self.estimate_tokens(msg.get("content", "")) 
            for msg in messages
        )
        
        if current_tokens <= target_tokens:
            return messages
        
        # 逐步压缩
        keep_recent = 10
        while current_tokens > target_tokens and keep_recent > 2:
            compressed = self.compress_messages(messages, keep_recent)
            current_tokens = sum(
                self.estimate_tokens(msg.get("content", "")) 
                for msg in compressed
            )
            keep_recent -= 1
        
        return compressed if current_tokens <= target_tokens else messages[-5:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取压缩统计"""
        if not self._compression_stats:
            return {}
        
        total_compressions = len(self._compression_stats)
        avg_reduction = sum(s["reduction_rate"] for s in self._compression_stats) / total_compressions
        avg_duration = sum(s["duration"] for s in self._compression_stats) / total_compressions
        
        return {
            "total_compressions": total_compressions,
            "avg_reduction_rate": f"{avg_reduction * 100:.2f}%",
            "avg_duration": f"{avg_duration:.3f}s",
            "recent_compressions": self._compression_stats[-5:]
        }


class SmartRetryHandler:
    """智能重试处理器 - 工具失败后的智能重试策略"""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._retry_stats = defaultdict(list)
    
    async def execute_with_retry(
        self,
        tool_call: Dict[str, Any],
        executor_func,
        user_text: str
    ) -> Dict[str, Any]:
        """
        带重试的工具执行
        """
        tool_name = tool_call.get("function", {}).get("name", "unknown")
        
        for attempt in range(self.max_retries):
            try:
                result = await executor_func(tool_call, user_text)
                
                if result.get("success", True):
                    # 记录成功
                    self._retry_stats[tool_name].append({
                        "attempts": attempt + 1,
                        "success": True,
                        "timestamp": datetime.now().isoformat()
                    })
                    return result
                
                # 失败但还有重试机会
                if attempt < self.max_retries - 1:
                    LOGGER.warning(
                        f"工具 {tool_name} 执行失败，尝试重试 ({attempt + 1}/{self.max_retries})"
                    )
                    
                    # 调整参数（如果可能）
                    tool_call = self._adjust_parameters(tool_call, result)
                    
                    # 等待一段时间再重试
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    # 最后一次尝试也失败
                    self._retry_stats[tool_name].append({
                        "attempts": attempt + 1,
                        "success": False,
                        "error": result.get("error", "unknown"),
                        "timestamp": datetime.now().isoformat()
                    })
                    return result
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    LOGGER.warning(
                        f"工具 {tool_name} 执行异常，尝试重试: {str(e)}"
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    self._retry_stats[tool_name].append({
                        "attempts": attempt + 1,
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                    return {
                        "success": False,
                        "error": str(e),
                        "content": json.dumps({"error": str(e)}, ensure_ascii=False)
                    }
        
        return {
            "success": False,
            "error": "Max retries exceeded",
            "content": json.dumps({"error": "Max retries exceeded"}, ensure_ascii=False)
        }
    
    def _adjust_parameters(
        self, 
        tool_call: Dict[str, Any], 
        previous_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        根据失败结果调整工具参数
        """
        # 简单策略：如果是参数错误，尝试调整
        error_msg = previous_result.get("error", "").lower()
        
        if "timeout" in error_msg:
            # 超时错误，可能需要调整超时参数
            pass
        elif "not found" in error_msg:
            # 未找到错误，可能需要调整查询参数
            pass
        
        # 默认返回原参数
        return tool_call
    
    def get_stats(self) -> Dict[str, Any]:
        """获取重试统计"""
        stats = {}
        
        for tool_name, retries in self._retry_stats.items():
            if retries:
                total = len(retries)
                success_count = sum(1 for r in retries if r.get("success", False))
                avg_attempts = sum(r["attempts"] for r in retries) / total
                
                stats[tool_name] = {
                    "total_executions": total,
                    "success_count": success_count,
                    "failure_count": total - success_count,
                    "success_rate": f"{success_count / total * 100:.2f}%",
                    "avg_attempts": f"{avg_attempts:.2f}"
                }
        
        return stats

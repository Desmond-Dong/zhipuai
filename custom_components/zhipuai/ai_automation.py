"""Simplified AI Automation support for natural language automation creation."""

from __future__ import annotations

import json
import logging
import os
import yaml
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.service import ServiceCall
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util
import voluptuous as vol

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


class AIAutomationManager:
    """Simplified AI automation manager that directly uses AI-generated YAML."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the AI automation manager."""
        self.hass = hass

    async def create_automation_from_description(
        self,
        description: str,
        name: Optional[str] = None,
        area_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create an automation from natural language description."""
        try:
            # 使用对话引擎生成YAML配置
            yaml_config = await self._generate_automation_yaml(description, name, area_id)

            if not yaml_config:
                return {
                    "success": False,
                    "error": "Failed to generate automation YAML"
                }

            # 直接写入automations.yaml文件
            success = await self._write_yaml_to_config(yaml_config)

            if success:
                # 重新加载自动化
                try:
                    await self.hass.services.async_call("automation", "reload", {}, blocking=True)
                    return {
                        "success": True,
                        "message": f"自动化创建成功！配置已写入automations.yaml并重新加载。你可以在自动化界面中找到它。"
                    }
                except Exception as reload_error:
                    return {
                        "success": True,
                        "message": f"自动化配置已写入，但重新加载失败: {reload_error}。请手动重新加载自动化或重启Home Assistant。"
                    }
            else:
                return {
                    "success": False,
                    "error": "Failed to write automation configuration"
                }

        except Exception as e:
            _LOGGER.error("Error creating automation: %s", e)
            return {
                "success": False,
                "error": str(e)
            }

    async def _generate_automation_yaml(
        self,
        description: str,
        name: Optional[str] = None,
        area_id: Optional[str] = None
    ) -> Optional[str]:
        """Generate automation YAML using the ZhipuAI conversation engine."""
        try:
            # 构建专门的YAML生成请求
            yaml_prompt = f"""请根据以下描述生成一个标准的Home Assistant自动化YAML配置：

描述: {description}"""

            if name:
                yaml_prompt += f"\n自动化名称: {name}"

            if area_id:
                yaml_prompt += f"\n区域ID: {area_id}"

            yaml_prompt += """

要求:
1. 生成完整的YAML格式自动化配置
2. 只返回YAML代码，不要任何其他说明
3. 确保YAML格式正确，可以直接写入automations.yaml文件
4. 使用标准的Home Assistant自动化字段: alias, trigger, condition, action
5. 不要包含id字段

示例格式:
```yaml
alias: "每天晚上8点半提醒接儿子"
trigger:
  - platform: time
    at: "20:30:00"
condition: []
action:
  - service: notify.persistent_notification
    data:
      title: "提醒"
      message: "时间到了，该去接儿子了"
mode: single
```

请直接返回YAML代码，不要包含任何其他文字："""

            # 调用智谱AI对话引擎
            try:
                # 获取对话组件
                from .conversation import ZhipuAIConversationAgent

                # 创建一个特殊的对话输入用于YAML生成
                import homeassistant.components.conversation as conversation

                yaml_input = conversation.ConversationInput(
                    text=yaml_prompt,
                    conversation_id="automation_yaml_generation",
                    language="zh"
                )

                # 查找智谱AI对话实例
                # 由于我们无法直接访问对话实例，使用一个备选方案
                # 备选方案：调用现有的智谱API
                yaml_result = await self._call_zhipuai_api_for_yaml(yaml_prompt)

                if yaml_result:
                    # 清理返回结果，提取纯YAML代码
                    return self._extract_yaml_from_response(yaml_result)

            except ImportError:
                _LOGGER.warning("Could not import conversation agent, using fallback logic")

            # 备选逻辑：如果无法访问对话引擎，使用简化逻辑
            return self._generate_fallback_yaml(description, name, area_id)

        except Exception as e:
            _LOGGER.error("Error generating automation YAML: %s", e)
            return None

    async def _call_zhipuai_api_for_yaml(self, prompt: str) -> Optional[str]:
        """Call ZhipuAI API directly for YAML generation."""
        try:
            import aiohttp
            import json

            # 获取API密钥 - 从HASS配置中获取
            api_key = self._get_api_key()
            if not api_key:
                _LOGGER.error("No ZhipuAI API key available")
                return None

            # 构建请求参数
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            request_params = {
                "model": "glm-4-flash",  # 使用默认模型
                "messages": messages,
                "stream": False,  # 不使用流式响应
                "max_tokens": 2000,
                "temperature": 0.1  # 低温度以确保稳定的YAML输出
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            # 调用智谱AI API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                    json=request_params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        _LOGGER.error("API request failed: %s", error_text)
                        return None

                    response_data = await response.json()

                    # 提取回复内容
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        content = response_data["choices"][0]["message"]["content"]
                        _LOGGER.info("ZhipuAI response received for YAML generation")
                        return content
                    else:
                        _LOGGER.error("Invalid response format from ZhipuAI")
                        return None

        except Exception as e:
            _LOGGER.error("Error calling ZhipuAI API: %s", e)
            return None

    def _get_api_key(self) -> Optional[str]:
        """Get ZhipuAI API key from Home Assistant configuration."""
        try:
            # 查找智谱AI的配置条目
            from homeassistant.config_entries import ConfigEntries

            config_entries: ConfigEntries = self.hass.config_entries
            zhipuai_entries = config_entries.async_entries_for_domain("zhipuai")

            if zhipuai_entries:
                # 获取第一个智谱AI配置条目
                entry = zhipuai_entries[0]
                # API密钥存储在runtime_data中
                api_key = entry.runtime_data
                if api_key:
                    return api_key

            _LOGGER.warning("No ZhipuAI configuration found")
            return None

        except Exception as e:
            _LOGGER.error("Error getting API key: %s", e)
            return None

    def _extract_yaml_from_response(self, response: str) -> Optional[str]:
        """Extract pure YAML code from AI response."""
        try:
            import re

            # 查找 ```yaml 代码块
            yaml_match = re.search(r'```yaml\s*\n(.*?)\n```', response, re.DOTALL)
            if yaml_match:
                return yaml_match.group(1).strip()

            # 查找 ``` 代码块（没有语言标识）
            code_match = re.search(r'```\s*\n(.*?)\n```', response, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()

            # 如果没有代码块，尝试提取看起来像YAML的内容
            lines = response.split('\n')
            yaml_lines = []
            in_yaml = False

            for line in lines:
                line = line.strip()
                if line.startswith('alias:') or line.startswith('trigger:') or line.startswith('action:'):
                    in_yaml = True

                if in_yaml and line:
                    yaml_lines.append(line)

            if yaml_lines:
                return '\n'.join(yaml_lines)

            return None

        except Exception as e:
            _LOGGER.error("Error extracting YAML from response: %s", e)
            return None

    def _generate_fallback_yaml(
        self,
        description: str,
        name: Optional[str] = None,
        area_id: Optional[str] = None
    ) -> Optional[str]:
        """Fallback YAML generation when AI is not available."""
        if "提醒" in description or "通知" in description or "记得" in description or "记住" in description:
            time_str = self._extract_time_from_description(description)
            alias_name = name or self._extract_name_from_description(description)

            return f'''alias: "{alias_name}"
trigger:
  - platform: time
    at: "{time_str}"
condition: []
action:
  - service: notify.persistent_notification
    data:
      title: "⏰ 提醒"
      message: "{description}"
mode: single'''

        elif "当" in description and "就" in description:
            return f'''alias: "{name or self._extract_name_from_description(description)}"
trigger:
  - platform: state
    entity_id: binary_sensor.motion_sensor
    to: "on"
condition: []
action:
  - service: light.turn_on
    target:
      entity_id: light.living_room
mode: single'''

        else:
            return f'''alias: "{name or self._extract_name_from_description(description)}"
trigger:
  - platform: state
    entity_id: binary_sensor.default_sensor
    to: "on"
condition: []
action:
  - service: light.turn_on
    target:
      entity_id: light.default_light
mode: single'''

    def _extract_time_from_description(self, description: str) -> str:
        """Extract time from natural language description."""
        import re

        description_lower = description.lower()

        # 匹配 "X点Y分" 格式
        match = re.search(r'(\d+)\s*点\s*(\d+)\s*分', description_lower)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            return f"{hour:02d}:{minute:02d}:00"

        # 匹配 "X点半" 格式
        match = re.search(r'(\d+)\s*点半', description_lower)
        if match:
            hour = int(match.group(1))
            return f"{hour:02d}:30:00"

        # 匹配 "X点" 格式
        match = re.search(r'(\d+)\s*点', description_lower)
        if match:
            hour = int(match.group(1))
            return f"{hour:02d}:00:00"

        # 默认时间
        return "20:30:00"

    def _extract_name_from_description(self, description: str) -> str:
        """Extract a meaningful name from description."""
        # 简单的名称提取逻辑
        if len(description) > 50:
            return description[:47] + "..."
        return description

    async def _write_yaml_to_config(self, yaml_config: str) -> bool:
        """Write YAML configuration to automations.yaml file."""
        try:
            import os
            import yaml
            import shutil

            config_dir = self.hass.config.config_dir
            automations_file = os.path.join(config_dir, "automations.yaml")

            def _write_file():
                """在执行器中执行文件操作"""
                # 备份现有文件
                if os.path.exists(automations_file):
                    backup_file = f"{automations_file}.backup_{int(dt_util.now().timestamp())}"
                    shutil.copy2(automations_file, backup_file)
                    _LOGGER.info("Backed up existing automations.yaml")

                # 读取现有配置
                existing_automations = []
                if os.path.exists(automations_file):
                    with open(automations_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            existing_automations = yaml.safe_load(content) or []
                        else:
                            # 如果文件为空，添加YAML头部
                            existing_automations = []

                # 解析新的YAML配置
                try:
                    new_automation = yaml.safe_load(yaml_config)
                    if not isinstance(new_automation, list):
                        new_automation = [new_automation]
                except yaml.YAMLError as e:
                    _LOGGER.error("Invalid YAML generated: %s", e)
                    raise ValueError(f"生成的YAML格式无效: {e}")

                # 添加新自动化
                existing_automations.extend(new_automation)

                # 写入配置文件
                with open(automations_file, 'w', encoding='utf-8') as f:
                    f.write("# Home Assistant Automations\n")
                    f.write("# Generated by ZhipuAI Integration\n\n")
                    yaml.dump(existing_automations, f,
                             default_flow_style=False,
                             allow_unicode=True,
                             sort_keys=False,
                             indent=2)

                return True

            # 在执行器中运行文件操作
            await self.hass.async_add_executor_job(_write_file)
            _LOGGER.info("Automation written to config file")
            return True

        except Exception as e:
            _LOGGER.error("Could not write automation to config file: %s", e)
            return False


# 全局管理器实例
_automation_manager: Optional[AIAutomationManager] = None


def get_automation_manager(hass: HomeAssistant) -> AIAutomationManager:
    """Get the AI automation manager instance."""
    global _automation_manager
    if _automation_manager is None:
        _automation_manager = AIAutomationManager(hass)
    return _automation_manager


async def async_setup_ai_automation(hass: HomeAssistant) -> None:
    """Set up AI automation services."""
    manager = get_automation_manager(hass)

    # 注册服务
    async def create_automation_service(call: ServiceCall) -> Dict[str, Any]:
        """Create automation from natural language."""
        description = call.data.get("description", "")
        name = call.data.get("name")
        area_id = call.data.get("area_id")

        if not description:
            return {"success": False, "error": "Description is required"}

        return await manager.create_automation_from_description(description, name, area_id)

    hass.services.async_register(
        DOMAIN, "create_automation", create_automation_service,
        schema=vol.Schema({
            vol.Required("description"): cv.string,
            vol.Optional("name"): cv.string,
            vol.Optional("area_id"): cv.string,
        })
    )

    _LOGGER.info("AI automation services registered")
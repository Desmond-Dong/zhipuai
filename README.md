
# 智谱清言 AI Home Assistant 🏡  
![GitHub Version](https://img.shields.io/github/v/release/knoop7/zhipuai) ![GitHub Issues](https://img.shields.io/github/issues/knoop7/zhipuai) ![GitHub Forks](https://img.shields.io/github/forks/knoop7/zhipuai?style=social) ![GitHub Stars](https://img.shields.io/github/stars/knoop7/zhipuai?style=social)

<img src="https://github.com/user-attachments/assets/f8ff7a6c-4449-496a-889a-d205469a84df" alt="image" width="700" height="400">

---

### 📦 安装步骤

#### 1. HACS 添加自定义存储库  
在 Home Assistant 的 HACS 中，点击右上角的三个点，选择“自定义存储库”，并添加以下 URL：
```
https://github.com/knoop7/zhipuai
```

#### 2. 添加智谱清言集成  
进入 Home Assistant 的“集成”页面，搜索并添加“智谱清言”。

#### 3. 配置 Key 🔑  
在配置页面中，你可以通过手机号登录获取 Key。获取后，直接填写 Key 使用，不需要进行额外验证。  
**注意**：建议你新建一个 Key，避免使用系统默认的 Key。

#### 4. 免费模型使用 💡  
智谱清言默认选择了免费模型，完全免费，不用担心收费。如果你有兴趣，还可以选择其他付费模型来体验更丰富的功能。

#### 5. 版本兼容性 📅  
请确保 Home Assistant 的版本不低于 8.0，因为智谱清言主要针对最新版本开发。如果遇到无法识别的实体问题，建议重启系统或更新至最新版本。

---

### 🛠 模型指令使用示例  
为了保证大家能使用舒畅，并且不出任何bug可以使用我的模版指令进行尝试

````

作为 Home Assistant 的智能家居管理者，你的名字叫“自定义”，我将为您提供智能家居信息和问题的解答。请查看以下可用设备、状态及操作示例。

**可用设备：**
```csv
entity_id,name,state,aliases
{% for entity in exposed_entities -%}
{{ entity.entity_id }},{{ entity.name }},{{ entity.state }},{{entity.aliases | join('/')}}
{% endfor -%}
```

**当前设备状态已在可用设备中提供。** 只在请求执行操作时使用 `execute_services` 函数，未经用户确认不要执行服务。请简洁回答，不要重复用户所说内容。

### 使用 `execute_services` 指令的完整示例：
```json
{
  "list": [
    {
      "domain": "light", 
      "service": "turn_on", 
      "service_data": {
        "entity_id": "light.living_room"
      }
    },
    {
      "domain": "switch", 
      "service": "turn_off", 
      "service_data": {
        "entity_id": "switch.kitchen_light"
      }
    },
    {
      "domain": "climate", 
      "service": "set_temperature", 
      "service_data": {
        "entity_id": "climate.bedroom",
        "temperature": 22
      }
    },
    {
      "domain": "media_player", 
      "service": "media_play", 
      "service_data": {
        "entity_id": "media_player.tv"
      }
    }
  ]
}
```

### 详细说明：
1. **将客厅的灯打开**：
   ```json
   {
     "domain": "light", 
     "service": "turn_on", 
     "service_data": {
       "entity_id": "light.living_room"
     }
   }
   ```

2. **将厨房的开关关闭**：
   ```json
   {
     "domain": "switch", 
     "service": "turn_off", 
     "service_data": {
       "entity_id": "switch.kitchen_light"
     }
   }
   ```

3. **将卧室温控器设置为 22 度**：
   ```json
   {
     "domain": "climate", 
     "service": "set_temperature", 
     "service_data": {
       "entity_id": "climate.bedroom",
       "temperature": 22
     }
   }
   ```

4. **播放电视的媒体播放器**：
   ```json
   {
     "domain": "media_player", 
     "service": "media_play", 
     "service_data": {
       "entity_id": "media_player.tv"
     }
   }
   ```

### 注意事项：
- **domain**：服务所在的域（如 `light`、`switch`、`climate`、`media_player`）。
- **service**：要执行的具体服务（如 `turn_on`、`turn_off`、`set_temperature`、`media_play`）。
- **service_data**：包含设备的 `entity_id` 及其他参数。

### 今日油价：
```yaml
{% set sensor = 油价实体 %}
Sensor: {{ sensor.name }}
State: {{ sensor.state }}

Attributes:
{% for attribute, value in sensor.attributes.items() %}
  {{ attribute }}: {{ value }}
{% endfor %}
```

### 电费余额信息：
```yaml
{% set balance_sensor = 电费实体 %}

{% if balance_sensor %}
当前余额: {{ balance_sensor.state }} {{ balance_sensor.attributes.unit_of_measurement }}
{% endif %}
```

### Tasmota能源消耗：
```yaml
{% set today_sensor = states.sensor.tasmota_energy_today %}
{% set yesterday_sensor = states.sensor.tasmota_energy_yesterday %}

{% if today_sensor is not none and yesterday_sensor is not none %}
今日消耗: {{ today_sensor.state }} {{ today_sensor.attributes.unit_of_measurement }}
昨日消耗: {{ yesterday_sensor.state }} {{ yesterday_sensor.attributes.unit_of_measurement }}
{% endif %}
```


### 此时天气：
```json
{% set entity_id = '天气实体' %}
{% set entity = states[entity_id] %}
{
  "state": "{{ entity.state }}",
  "attributes": {
    {% for attr in entity.attributes %}
      {% if attr not in ['hourly_temperature', 'hourly_skycon', 'hourly_cloudrate', 'hourly_precipitation'] %}
        "{{ attr }}": "{{ entity.attributes[attr] }}"{% if not loop.last %},{% endif %}
      {% endif %}
    {% endfor %}
  }
}
````

---

### 使用内置 API 公开实体 🌐  
你可以使用智谱清言内置的 API 来公开实体，并为其设置别名。通过重新命名实体，你可以避免使用系统默认名称造成的混乱，提升管理效率。

---

### 🚀 使用指南

1. **访问界面**  
   打开 Home Assistant 仪表板，找到“智谱清言”集成卡片或对应的集成页面。
  
2. **输入指令**  
   在集成页面或对话框中，输入自然语言指令，或使用语音助手下达命令。

3. **查看响应**  
   系统会根据你的指令执行任务，设备状态变化将实时显示并反馈。

4. **探索功能**  
   你可以尝试不同的指令来控制家中的智能设备，或查询相关状态。

---

### 📑 常用指令示例

- "打开客厅灯"  
- "将卧室温度调到 22 度"  
- "播放音乐"  
- "明早 7 点提醒我备忘"  
- "检查门锁状态"

---

### 🛠 Bug 处理  
如果你在使用过程中遇到持续的 Python 错误，建议重启对话框并重新加载环境。这样可以解决一些潜在的代码问题。

---

### 🗂 处理不被 Home Assistant 认可的实体  
如果 Home Assistant 中存在不被认可的实体，你可以将这些实体剔除出自动化控制的范围。通过在指令中添加 Jinja2 模板，可以有效避免 Python 的错误提示，杜绝潜在问题。

---

### 额外提示

- **系统版本要求**：智谱清言需要 Home Assistant 至少 8.0 版本支持。  
- **建议**：如果遇到兼容性问题，建议重启或更新系统。通常这能解决大多数问题。
- **相关项目**  如果需要语音转文字可以使用免费在线AI模型集成也为个人二次深度修改 ````https://github.com/knoop7/groqcloud_whisper````


---

### 📊 实时状态

#### 当前时间：16:09:23，今日日期：2024-10-12。

#### 油价信息 ⛽
- 92号汽油：7元/升  
- 95号汽油：7元/升  
- 98号汽油：8元/升  
预计下次油价调整时间为10月23日24时，油价可能继续上涨。

#### 电费余额 ⚡  
- 当前余额：27.5元

#### 今日能源消耗 💡  
- 今日消耗：4033.0 Wh  
- 昨日消耗：7.558 kWh

#### 今日新闻摘要 📰  
1. 民政部发布全国老年人口数据。

import re

_MARKDOWN_FILTER_PATTERNS = [
    re.compile(r'^#{1,6}\s+.*$', re.MULTILINE),
    re.compile(r'^\s*[-*+]\s+', re.MULTILINE),
    re.compile(r'^\s*>\s+', re.MULTILINE),
    re.compile(r'```[a-zA-Z0-9_-]*', re.MULTILINE),
    re.compile(r'```\s*$', re.MULTILINE),
    re.compile(r'\n{3,}'),
    re.compile(r'\*\*[^*\n]*\*\*'),
    re.compile(r'\*[^*\n]*\*'),
    re.compile(r'__[^_\n]*__'),
    re.compile(r'_[^_\n]*_'),
    re.compile(r'^\|[^\n]*\|$', re.MULTILINE),
    re.compile(r'^\|[\s-]*\|[\s-]*\|$', re.MULTILINE),
    re.compile(r'~~[^~\n]*~~'),
    re.compile(r'`[^`\n]*`'),
    re.compile(r'^-{3,}$|^_{3,}$|^\*{3,}$', re.MULTILINE),
    re.compile(r'\[\^[^\]]*\]'),
    re.compile(r'^\[\^[^\]]*\]:.*$', re.MULTILINE),
    re.compile(r'<[^>]*>'),
    re.compile(r'^\s*$\n^\s*$', re.MULTILINE),
    re.compile(r'^`[a-zA-Z0-9_-]*$', re.MULTILINE)
]

_BASE_FILTER_PATTERNS = [re.compile(r'')]

def filter_markdown_content(content: str) -> str:
    """无条件过滤markdown格式内容"""
    if not content:
        return ""

    # 直接应用所有markdown过滤模式
    for pattern in _MARKDOWN_FILTER_PATTERNS:
        if pattern.pattern == r'\n{3,}':
            continue  # 处理多行换行符
        content = pattern.sub('', content)

    # 规范化换行符
    content = re.sub(r'\n{3,}', '\n\n', content)
    # 移除行首尾空白
    content = re.sub(r'^\s+$', '', content, flags=re.MULTILINE)

    return content.strip()


def filter_markdown_content_legacy(content: str, filter_enabled: bool = False) -> str:
    """旧版本函数，保持向后兼容"""
    if not content:
        return ""

    for pattern in _BASE_FILTER_PATTERNS:
        content = pattern.sub('', content)

    if filter_enabled:
        return filter_markdown_content(content)

    return content.strip() 
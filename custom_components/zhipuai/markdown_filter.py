import re

_MARKDOWN_FILTER_PATTERNS = [
    re.compile(r'^#{1,6}\s+.*$', re.MULTILINE),
    re.compile(r'^\s*[-*+]\s+', re.MULTILINE),
    re.compile(r'^\s*>\s+', re.MULTILINE),
    re.compile(r'```[a-zA-Z0-9_-]*', re.MULTILINE),
    re.compile(r'```\s*$', re.MULTILINE),
    re.compile(r'\n{3,}'),
    re.compile(r'\*\*([^*\n]*)\*\*'),      # 修复粗体：保留内容
    re.compile(r'\*([^*\n]*)\*'),          # 修复斜体：保留内容
    re.compile(r'__([^_\n]*)__'),          # 修复粗体：保留内容
    re.compile(r'_([^_\n]*)_'),            # 修复斜体：保留内容
    re.compile(r'^\|[^\n]*\|$', re.MULTILINE),
    re.compile(r'^\|[\s-]*\|[\s-]*\|$', re.MULTILINE),
    re.compile(r'~~([^~\n]*)~~'),          # 修复删除线：保留内容
    re.compile(r'`([^`\n]*)`'),            # 修复行内代码：保留内容
    re.compile(r'^-{3,}$|^_{3,}$|^\*{3,}$', re.MULTILINE),
    re.compile(r'\[\^[^\]]*\]'),
    re.compile(r'^\[\^[^\]]*\]:.*$', re.MULTILINE),
    re.compile(r'<[^>]*>'),
    re.compile(r'^\s*$\n^\s*$', re.MULTILINE),
    re.compile(r'^`[a-zA-Z0-9_-]*$', re.MULTILINE)
]

_BASE_FILTER_PATTERNS = [re.compile(r'')]

def filter_markdown_content(content: str) -> str:
    """无条件过滤markdown格式内容，保留英文单词间的空格"""
    if not content:
        return ""

    # 定义需要保留内容的模式（这些模式有捕获组，用于移除markdown语法但保留内容）
    patterns_with_capture = [
        re.compile(r'\*\*([^*\n]*)\*\*'),      # 粗体：保留内容
        re.compile(r'\*([^*\n]*)\*'),          # 斜体：保留内容
        re.compile(r'__([^_\n]*)__'),          # 粗体：保留内容
        re.compile(r'_([^_\n]*)_'),            # 斜体：保留内容
        re.compile(r'~~([^~\n]*)~~'),          # 删除线：保留内容
        re.compile(r'`([^`\n]*)`'),            # 行内代码：保留内容
    ]

    # 定义需要完全移除的模式（这些模式没有捕获组，用于移除整行markdown语法）
    patterns_remove = [
        re.compile(r'^#{1,6}\s+.*$', re.MULTILINE),
        re.compile(r'^\s*[-*+]\s+', re.MULTILINE),
        re.compile(r'^\s*>\s+', re.MULTILINE),
        re.compile(r'```[a-zA-Z0-9_-]*', re.MULTILINE),
        re.compile(r'```\s*$', re.MULTILINE),
        re.compile(r'^\|[^\n]*\|$', re.MULTILINE),
        re.compile(r'^\|[\s-]*\|[\s-]*\|$', re.MULTILINE),
        re.compile(r'^-{3,}$|^_{3,}$|^\*{3,}$', re.MULTILINE),
        re.compile(r'\[\^[^\]]*\]'),
        re.compile(r'^\[\^[^\]]*\]:.*$', re.MULTILINE),
        re.compile(r'<[^>]*>'),
        re.compile(r'^\s*$\n^\s*$', re.MULTILINE),
        re.compile(r'^`[a-zA-Z0-9_-]*$', re.MULTILINE)
    ]

    # 应用需要保留内容的模式（移除markdown语法但保留内容和空格）
    for pattern in patterns_with_capture:
        content = pattern.sub(r'\1', content)

    # 应用需要完全移除的模式
    for pattern in patterns_remove:
        content = pattern.sub('', content)

    # 规范化换行符
    content = re.sub(r'\n{3,}', '\n\n', content)
    # 移除行首尾空白
    content = re.sub(r'^\s+$', '', content, flags=re.MULTILINE)

    return content.strip()


def filter_markdown_streaming(content: str) -> str:
    """专门用于streaming处理的markdown过滤器，保留所有空格"""
    if not content:
        return ""

    # 定义需要保留内容的模式（这些模式有捕获组，用于移除markdown语法但保留内容）
    patterns_with_capture = [
        re.compile(r'\*\*([^*\n]*)\*\*'),      # 粗体：保留内容
        re.compile(r'\*([^*\n]*)\*'),          # 斜体：保留内容
        re.compile(r'__([^_\n]*)__'),          # 粗体：保留内容
        re.compile(r'_([^_\n]*)_'),            # 斜体：保留内容
        re.compile(r'~~([^~\n]*)~~'),          # 删除线：保留内容
        re.compile(r'`([^`\n]*)`'),            # 行内代码：保留内容
    ]

    # 定义需要完全移除的模式（只移除整行markdown语法，不影响单个chunk内的空格）
    patterns_remove = [
        re.compile(r'^#{1,6}\s+.*$', re.MULTILINE),
        re.compile(r'^\s*[-*+]\s+', re.MULTILINE),
        re.compile(r'^\s*>\s+', re.MULTILINE),
        re.compile(r'```[a-zA-Z0-9_-]*', re.MULTILINE),
        re.compile(r'```\s*$', re.MULTILINE),
        re.compile(r'^\|[^\n]*\|$', re.MULTILINE),
        re.compile(r'^\|[\s-]*\|[\s-]*\|$', re.MULTILINE),
        re.compile(r'^-{3,}$|^_{3,}$|^\*{3,}$', re.MULTILINE),
        re.compile(r'\[\^[^\]]*\]'),
        re.compile(r'^\[\^[^\]]*\]:.*$', re.MULTILINE),
        re.compile(r'<[^>]*>'),
        re.compile(r'^\s*$\n^\s*$', re.MULTILINE),
        re.compile(r'^`[a-zA-Z0-9_-]*$', re.MULTILINE)
    ]

    # 应用需要保留内容的模式（移除markdown语法但保留内容和空格）
    for pattern in patterns_with_capture:
        content = pattern.sub(r'\1', content)

    # 应用需要完全移除的模式
    for pattern in patterns_remove:
        content = pattern.sub('', content)

    # 对于streaming，不使用strip()以保留chunk内的所有空格
    # 只规范化换行符
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content


def filter_markdown_content_legacy(content: str, filter_enabled: bool = False) -> str:
    """旧版本函数，保持向后兼容"""
    if not content:
        return ""

    for pattern in _BASE_FILTER_PATTERNS:
        content = pattern.sub('', content)

    if filter_enabled:
        return filter_markdown_content(content)

    return content.strip() 
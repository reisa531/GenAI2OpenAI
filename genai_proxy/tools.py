"""Message normalization and prompt-based tool-call compatibility."""

import json
import re
import uuid


def convert_messages_to_genai_format(messages):
    """从消息列表中提取 GenAI 所需的 `chatInfo`。

    当前上游实际请求中 `chatInfo` 只使用最后一条用户消息内容，因此这里
    仅做最小提取。

    Args:
        messages (list[dict]): OpenAI 风格的消息列表。

    Returns:
        str: 最后一条用户消息的文本内容；若不存在则返回空字符串。
    """
    # 上游会单独接收一份 chatInfo，这里取最后一条 user 消息与网页行为对齐。
    chat_info = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            chat_info = msg.get("content", "")
            break

    return chat_info


def normalize_content_for_genai(content):
    """将 OpenAI 消息 content 归一化为上游可读文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    text_parts.append(text)
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)
    return str(content)


def normalize_messages_for_genai(messages):
    """把 OpenAI tool messages 降级为普通文本，避免上游无法理解原生工具结构。"""
    normalized_messages = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user")
        content = normalize_content_for_genai(message.get("content"))

        if role == "tool":
            tool_name = message.get("name") or message.get("tool_call_id") or "tool"
            normalized_messages.append({
                "role": "user",
                "content": f"工具 {tool_name} 返回结果：\n{content}",
            })
            continue

        tool_calls = message.get("tool_calls")
        if role == "assistant" and tool_calls and not content:
            normalized_messages.append({
                "role": "assistant",
                "content": "已请求调用工具：\n" + json.dumps(tool_calls, ensure_ascii=False),
            })
            continue

        normalized_messages.append({
            "role": role,
            "content": content,
        })

    return normalized_messages


def should_enable_tools(tools, tool_choice):
    """判断当前请求是否需要启用本地工具调用兼容层。"""
    return bool(tools) and tool_choice != "none"


def get_request_tools(req_data):
    """读取新版 tools 或旧版 functions 入参，统一为 OpenAI tools 结构。"""
    tools = req_data.get("tools")
    if tools:
        return tools

    functions = req_data.get("functions")
    if not functions:
        return []

    return [
        {
            "type": "function",
            "function": function,
        }
        for function in functions
        if isinstance(function, dict)
    ]


def get_request_tool_choice(req_data):
    """读取新版 tool_choice 或旧版 function_call 入参。"""
    if "tool_choice" in req_data:
        return req_data.get("tool_choice")

    function_call = req_data.get("function_call")
    if isinstance(function_call, dict) and function_call.get("name"):
        return {"name": function_call["name"]}
    return function_call


def normalize_tool_choice(tool_choice):
    """将 OpenAI 的 tool_choice 归一化为提示词中便于描述的约束。"""
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("name"):
            return {"name": tool_choice["name"]}
        function_name = tool_choice.get("function", {}).get("name")
        if function_name:
            return {"name": function_name}
    return "auto"


def build_tool_calling_messages(messages, tools, tool_choice):
    """通过提示词工程让无原生工具调用能力的上游返回可解析的工具调用 JSON。"""
    normalized_choice = normalize_tool_choice(tool_choice)
    tool_prompt = [
        "你可以调用调用方提供的工具，但上游 API 没有原生 tool calling 能力。",
        "当你决定调用工具时，优先输出一个 JSON 对象，不要输出 Markdown、解释或额外文本。",
        "JSON 格式必须为：{\"tool_calls\":[{\"name\":\"工具名\",\"arguments\":{}}]}。",
        "兼容格式：也允许输出 <tool_call>{\"name\":\"工具名\",\"arguments\":{}}</tool_call>；若并行调用可连续输出多个 <tool_call>...</tool_call>。",
        "arguments 必须是符合工具 JSON Schema 的对象。",
        "如果不需要调用工具，则正常回答用户，不要输出上述 JSON。",
        f"tool_choice: {json.dumps(normalized_choice, ensure_ascii=False)}",
        "可用工具：",
        json.dumps(tools, ensure_ascii=False),
    ]

    if normalized_choice == "required":
        tool_prompt.append("本次请求必须调用至少一个工具。")
    elif isinstance(normalized_choice, dict):
        tool_prompt.append(f"本次请求必须调用工具 {normalized_choice['name']}。")

    return [
        {"role": "system", "content": "\n".join(tool_prompt)},
        *messages,
    ]


def strip_json_code_fence(text):
    """去掉模型偶尔包裹的 JSON Markdown 代码块。"""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_json_object(text):
    """从文本中提取第一个完整 JSON 对象。"""
    stripped = strip_json_code_fence(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(stripped[start:index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def normalize_tool_call_arguments(arguments):
    """OpenAI 要求 function.arguments 是 JSON 字符串。"""
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return json.dumps(arguments, ensure_ascii=False)


def extract_tool_calls_from_xml(content):
    """从 <tool_call>...</tool_call> 中提取工具调用（XML 兼容，JSON 仍为主）。"""
    if not content:
        return []

    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.DOTALL)
    tool_calls = []
    for block in blocks:
        parsed = extract_json_object(block)
        if not isinstance(parsed, dict):
            continue

        name = parsed.get("name")
        if not name:
            continue

        arguments = parsed.get("arguments", {})
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": normalize_tool_call_arguments(arguments),
            },
        })

    return tool_calls


def parse_tool_calls_from_content(content):
    """解析提示词约定的工具调用 JSON，并转换为 OpenAI tool_calls 结构。"""
    if not content:
        return []

    # 优先 JSON：兼容当前主路径。
    parsed = extract_json_object(content)
    raw_calls = None
    if isinstance(parsed, dict):
        raw_calls = parsed.get("tool_calls")
        if raw_calls is None and parsed.get("name"):
            raw_calls = [parsed]
    elif "<tool_call>" in content:
        # JSON 解析失败时回退 XML。
        return extract_tool_calls_from_xml(content)

    if not isinstance(raw_calls, list):
        if "<tool_call>" in content:
            return extract_tool_calls_from_xml(content)
        return []

    tool_calls = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            continue

        function = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else raw_call
        name = function.get("name")
        if not name:
            continue

        arguments = function.get("arguments", {})
        tool_calls.append({
            "id": raw_call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": normalize_tool_call_arguments(arguments),
            },
        })

    return tool_calls

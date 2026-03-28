"""
ReAct Agent for battery recycling line planning.

Implements Reason-Act-Observe loop with two tools:
  1. search_kb   — semantic search over the ChromaDB knowledge base
  2. plan_line   — deterministic lookup: given battery traits, suggest unit path

Each iteration yields event dicts consumed by the SSE layer in server.py.
"""

import re
import logging
from typing import Generator

import ollama

from rag import KnowledgeBase

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:14b"
MAX_ITERATIONS = 6

SYSTEM_PROMPT = """\
你是 BRD-Agent，一个专业的废旧电池回收产线设计助手。

## 可用工具
你可以通过以下格式调用工具：

Thought: <你的推理过程>
Action: <工具名称>
Action Input: <工具参数>

可用的工具有：
1. search_kb — 在知识库中检索电池技术文件、工艺单元说明或回收案例。参数为检索关键词。
2. plan_line — 根据电池特性规划产线路径。参数为电池类型描述。

工具执行后你会收到 Observation，然后继续推理。

## 结束条件
当你收集到足够信息后，输出最终答案：

Thought: 我已经收集了足够的信息，可以生成回收报告了。
Final Answer:
<在此输出完整的结构化报告>

## 最终报告格式要求
你的 Final Answer 必须严格包含以下HTML格式内容：

<h4>1. 电池名称</h4>
<p><strong>[电池全称]</strong></p>
<p>[简要分析电池的化学体系、封装形式、主要特性、应用场景，50-100字]</p>

<h4>2. 产线搭建</h4>
<p><strong>工艺路线：</strong>[推荐的工艺路线名称]</p>
<p><strong>路径：</strong>[使用→连接工艺单元编号，如 U1→U2→U3→U12]</p>
<p><strong>预计回收率：</strong>[预计的金属回收率]</p>

<h4>3. 回收流程</h4>
<p>[详细描述回收流程，说明每个工艺单元的作用和注意事项，100-200字]</p>
<p><em>[风险等级和处理能力建议]</em></p>

## 重要规则
- 路径中的工艺单元必须使用 U1-U18 的标准编号
- 路径格式必须使用 → 符号连接，如 U1→U2→U3
- 工艺单元选择必须符合电池类型的实际回收需求
- 使用HTML标签(h4, p, strong, em)格式化输出
- 每个回复必须完整包含三个部分（电池名称、产线搭建、回收流程）
- 请用中文回答
"""


VALID_UNITS = {f"U{i}" for i in range(1, 19)}


def _call_llm(messages: list[dict]) -> str:
    """Send messages to Ollama and return the assistant reply."""
    resp = ollama.chat(model=LLM_MODEL, messages=messages, stream=False)
    return resp["message"]["content"]


def _parse_action(text: str) -> tuple[str | None, str | None]:
    """Extract Action and Action Input from LLM output."""
    action_m = re.search(r"Action:\s*(.+)", text)
    input_m = re.search(r"Action Input:\s*(.+)", text)
    if action_m and input_m:
        return action_m.group(1).strip(), input_m.group(1).strip()
    return None, None


def _extract_thought(text: str) -> str:
    """Return the Thought line(s) from the LLM output."""
    thoughts = re.findall(r"Thought:\s*(.+)", text)
    return " | ".join(thoughts) if thoughts else ""


def _tool_search_kb(kb: KnowledgeBase, query: str) -> str:
    hits = kb.search(query, top_k=4)
    if not hits:
        return "未找到相关文档。"
    parts = []
    for h in hits:
        src = h["source"]
        heading = h["heading"]
        snippet = h["text"][:600]
        parts.append(f"[{src} — {heading}]\n{snippet}")
    return "\n---\n".join(parts)


def _tool_plan_line(kb: KnowledgeBase, battery_desc: str) -> str:
    """Search KB for matching recovery cases and extract the unit path."""
    hits = kb.search(f"回收实例 产线配置 工艺单元路径 {battery_desc}", top_k=3)
    paths_found = []
    for h in hits:
        path_match = re.search(r"U\d+(?:\s*→\s*U\d+)+", h["text"])
        if path_match:
            paths_found.append(path_match.group(0))
    if paths_found:
        return f"根据知识库中的回收实例，推荐产线路径：\n" + "\n".join(paths_found)
    return "知识库中未找到完全匹配的回收路径，请根据电池特性和工艺单元说明自行规划。"


TOOLS = {
    "search_kb": _tool_search_kb,
    "plan_line": _tool_plan_line,
}


def run_agent(query: str, kb: KnowledgeBase) -> Generator[dict, None, None]:
    """
    Execute the ReAct loop.

    Yields event dicts with keys:
      - {"type": "thought", "content": str}
      - {"type": "tool_call", "tool": str, "input": str}
      - {"type": "observation", "content": str}
      - {"type": "answer", "content": str}
      - {"type": "error", "content": str}
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        logger.info("Iteration %d / %d", iteration, MAX_ITERATIONS)

        try:
            reply = _call_llm(messages)
        except Exception as e:
            yield {"type": "error", "content": f"LLM call failed: {e}"}
            return

        thought = _extract_thought(reply)
        if thought:
            yield {"type": "thought", "content": thought}

        if "Final Answer:" in reply:
            answer = reply.split("Final Answer:", 1)[1].strip()
            yield {"type": "answer", "content": answer}
            return

        action, action_input = _parse_action(reply)
        if action and action in TOOLS:
            yield {"type": "tool_call", "tool": action, "input": action_input}
            observation = TOOLS[action](kb, action_input)
            yield {"type": "observation", "content": f"[{action}] {observation[:800]}"}

            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        elif action:
            err = f"Unknown tool: {action}"
            yield {"type": "observation", "content": err}
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Observation: {err}. Available tools: search_kb, plan_line"})
        else:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": "请按照规定的格式调用工具(Action/Action Input)或输出最终答案(Final Answer)。"})

    yield {"type": "thought", "content": "达到最大迭代次数，正在生成最终答案..."}
    messages.append({
        "role": "user",
        "content": "你已经用完了所有迭代次数。请立即根据已有信息输出 Final Answer，使用要求的HTML格式。",
    })
    try:
        reply = _call_llm(messages)
        if "Final Answer:" in reply:
            answer = reply.split("Final Answer:", 1)[1].strip()
        else:
            answer = reply
        yield {"type": "answer", "content": answer}
    except Exception as e:
        yield {"type": "error", "content": f"Final answer generation failed: {e}"}

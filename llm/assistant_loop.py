"""LLM tool-calling loop, independent of robot transport details."""

import json
from collections.abc import Awaitable, Callable
from typing import Any


async def run_user_request(
    user_text: str,
    *,
    complete_chat: Callable[..., Awaitable[Any]],
    model: str,
    system_prompt: str,
    tools: list[dict[str, Any]],
    handlers: dict[str, Callable[..., Awaitable[str]]],
    speak: Callable[[str], Awaitable[bool]],
) -> str:
    """Run one LLM request and dispatch any requested robot actions."""
    suppress_final_reply = False
    messages: list[Any] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    response = await complete_chat(model=model, messages=messages, tools=tools, tool_choice="auto")

    while response.choices[0].finish_reason == "tool_calls":
        assistant_message = response.choices[0].message
        messages.append(assistant_message)
        for tool_call in assistant_message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"  [Tool] {name}({args})")
            handler = handlers.get(name)
            result = await handler(**args) if handler else json.dumps({"status": "error", "message": "Unknown tool."})
            print(f"  [Result] {result}")
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                parsed = {}
            if name in {"say_message", "deliver_message_to_person"}:
                suppress_final_reply = True
            elif name == "check_seat_and_report_back" and parsed.get("say_result", {}).get("status") == "ok":
                suppress_final_reply = True
            elif name == "go_to" and (
                parsed.get("say_result", {}).get("status") == "ok"
                or parsed.get("result", {}).get("say_result", {}).get("status") == "ok"
            ):
                suppress_final_reply = True
            messages.append({"role": "tool", "name": name, "content": result, "tool_call_id": tool_call.id})
        response = await complete_chat(model=model, messages=messages, tools=tools)

    if suppress_final_reply:
        print("[Robot] Final reply suppressed because the action already included spoken output")
        return ""
    reply = response.choices[0].message.content.strip()
    print(f"[Robot]: {reply}")
    try:
        await speak(reply)
    except Exception as exc:
        print(f"[TTS Error] {exc}")
    return reply

"""Mistral function-calling schemas used by the Go2 assistant."""

from typing import Any, Sequence


def build_tool_schemas(skills: Sequence[str]) -> list[dict[str, Any]]:
    """Return the tool contract exposed to the language model."""
    skill_names = list(skills)
    skill_list = ", ".join(skill_names)
    return [
        {
            "type": "function",
            "function": {
                "name": "deliver_message_to_person",
                "description": (
                    "Navigate to a person's seat or desk area and speak a short message there. "
                    "Use for requests to go to someone and tell, relay, deliver, greet, or say something. "
                    "Write the message as casual speech directed to the recipient. "
                    "Only set `skill` when the user explicitly requested a greeting or gesture."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "person": {
                            "type": "string",
                            "description": "The person to go to, such as Jini or Chen.",
                        },
                        "message": {
                            "type": "string",
                            "description": "The exact short, natural message to say after arriving.",
                        },
                        "skill": {
                            "type": "string",
                            "description": f"Optional gesture. Available skills: {skill_list}",
                            "enum": skill_names,
                        },
                    },
                    "required": ["person", "message"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "say_message",
                "description": "Speak a short message through the robot or local speaker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "The exact message to say."},
                        "skill": {
                            "type": "string",
                            "description": f"Optional gesture. Available skills: {skill_list}",
                            "enum": skill_names,
                        },
                    },
                    "required": ["message"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "do_skill",
                "description": f"Make Go2 perform a physical skill. Available skills: {skill_list}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string", "enum": skill_names},
                    },
                    "required": ["skill"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "describe_view",
                "description": "Answer a question using the robot's front camera.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The user's actual visual question.",
                        },
                    },
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "go_to",
                "description": "Navigate Go2 to a named destination or numbered point.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "Destination name or point."},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_person",
                "description": "Check whether a named person's seat appears occupied.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description": "The seat owner to check."},
                    },
                    "required": ["description"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_seat_and_report_back",
                "description": "Visit a person's seat, inspect it, return, and report the result aloud.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "person": {"type": "string", "description": "The seat owner to check."},
                    },
                    "required": ["person"],
                },
            },
        },
    ]

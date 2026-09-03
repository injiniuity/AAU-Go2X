import builtins
from typing import Any

from .. import config


def should_emit_log_message(message: str) -> bool:
    if config.VERBOSE_LOGS:
        return True
    normalized = str(message).lstrip()
    return normalized.startswith(config.DEFAULT_LOG_PREFIXES) or normalized.startswith(
        config.DEFAULT_LOG_STARTS
    )


def print(*args: Any, **kwargs: Any) -> None:
    sep = kwargs.get("sep", " ")
    message = sep.join(str(arg) for arg in args)
    if should_emit_log_message(message):
        builtins.print(*args, **kwargs)

from contextvars import ContextVar
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from literalai.step import Step
    from literalai.thread import Thread

active_steps_var = ContextVar[List["Step"]]("active_steps", default=[])
active_thread_var = ContextVar[Optional["Thread"]]("active_thread", default=None)

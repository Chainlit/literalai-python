from contextvars import ContextVar
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from chainlit_client.step import Step

active_steps_var = ContextVar[List["Step"]]("active_steps", default=[])
active_thread_id_var = ContextVar[Optional[str]]("active_thread", default=None)

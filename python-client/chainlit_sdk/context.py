from contextvars import ContextVar

active_steps_var = ContextVar("active_steps", default=[])
active_thread_id_var = ContextVar("active_thread", default=None)

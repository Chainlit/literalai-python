from contextvars import ContextVar

active_steps_var = ContextVar("active_steps", default=[])

import asyncio
import queue
import threading
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from literalai.api import API
    from literalai.step import StepDict


# to_thread is a backport of asyncio.to_thread from Python 3.9
async def to_thread(func, /, *args, **kwargs):
    import contextvars
    import functools

    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)


class EventProcessor:
    event_queue: queue.Queue

    def __init__(self, api: "API", batch_size: int = 1):
        self.batch_size = batch_size
        self.api = api
        self.event_queue = queue.Queue()
        self.processing_thread = threading.Thread(
            target=self._process_events, daemon=True
        )
        self.processing_thread.start()
        self.stop_event = threading.Event()

    def add_event(self, event: "StepDict"):
        self.event_queue.put(event)

    async def a_add_events(self, event: "StepDict"):
        await to_thread(self.event_queue.put, event)

    def _process_events(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            batch: List["StepDict"] = []
            try:
                # Try to fill the batch up to the batch_size
                while len(batch) < self.batch_size:
                    # Attempt to get events with a timeout
                    event = self.event_queue.get(timeout=0.5)
                    batch.append(event)
            except queue.Empty:
                # No more events at the moment, proceed with processing what's in the batch
                pass

            # Process the batch if any events are present
            if batch:
                loop.run_until_complete(self._process_batch(batch))

            # Stop if the stop_event is set and the queue is empty
            if self.stop_event.is_set() and self.event_queue.empty():
                break

    async def _process_batch(self, batch):
        res = await self.api.send_steps(
            steps=batch,
        )
        # simple one-try retry in case of network failure (no retry on graphql errors)
        if not res:
            await asyncio.sleep(0.5)
            await self.api.send_steps(
                steps=batch,
            )
            return

    def wait_until_queue_empty(self):
        self.stop_event.set()
        self.processing_thread.join()

    def __del__(self):
        self.wait_until_queue_empty()

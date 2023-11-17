import asyncio
import json
import unittest
from unittest.mock import patch, MagicMock
import chainlit_sdk


class TestExample(unittest.IsolatedAsyncioTestCase):
    @patch("chainlit_sdk.EventProcessor._process_events")
    async def test_simple(self, mock_process_events):
        event_processor = chainlit_sdk.EventProcessor()
        observer = chainlit_sdk.ObservabilityAgent(processor=event_processor)

        @observer.run
        def sync_function():
            observer.set_step_parameter("key", "value")
            return "result"

        result = sync_function()

        mock_process_events.assert_called_once()

    @patch("chainlit_sdk.EventProcessor.add_event")
    async def test_with(self, mock_add_event):
        observer = chainlit_sdk.ObservabilityAgent()
        with observer.step() as step:
            step.set_parameter("key", "value")
            pass

        mock_add_event.assert_called_once()


if __name__ == "__main__":
    unittest.main()

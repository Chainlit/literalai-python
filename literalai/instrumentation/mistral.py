import time
from typing import Dict, Union
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from literalai.client import LiteralClient

from literalai.helper import ensure_values_serializable
from literalai.my_types import ChatGeneration, CompletionGeneration, GenerationType
import logging

from literalai.context import active_steps_var, active_thread_var

from literalai.wrappers import AfterContext, BeforeContext, wrap_sync

logger = logging.getLogger(__name__)

is_mistral_instrumented = False

TO_WRAP = [
    {
        "module": "mistralai.client",
        "object": "MistralClient",
        "method": "chat",
        "metadata": {
            "type": GenerationType.CHAT,
        },
        "async": False,
    },
    {
        "module": "mistralai.client",
        "object": "MistralClient",
        "method": "completion",
        "metadata": {
            "type": GenerationType.COMPLETION,
        },
        "async": False,
    },
]

def instrument_mistral(client: "LiteralClient"):
    global is_mistral_instrumented
    if is_mistral_instrumented:
        return
    
    # TODO: Check if we have specific requirements on mistralai version
    # TODO: Check if need for on_new_generation callback

    def init_generation(generation_type: "GenerationType", kwargs):
        model = kwargs.get("model")
        tools = kwargs.get("tools")

        if generation_type == GenerationType.CHAT:
            orig_messages = kwargs.get("messages")
        
            messages = ensure_values_serializable(orig_messages)
            # TODO: add prompt_id and variables

            settings = {
                "model": model,
                "temperature": kwargs.get("temperature"),
                "top_p": kwargs.get("top_p"),
                "max_tokens": kwargs.get("max_tokens"),
                
                # "stream": kwargs.get("stream"),
                # "stop": kwargs.get("stop"), # TODO: add stop for completion only
                # "seed": kwargs.get("seed"), # TODO: check Mistral's random seed
                # TODO: check safe_prompt parameter

                # "tool_choice": kwargs.get("tool_choice"),
                # TODO: check stream for function calling
                # "frequency_penalty": kwargs.get("frequency_penalty"),
                # "logit_bias": kwargs.get("logit_bias"),
                # "logprobs": kwargs.get("logprobs"),
                # "top_logprobs": kwargs.get("top_logprobs"),
                # "n": kwargs.get("n"),
                # "presence_penalty": kwargs.get("presence_penalty"),
                # "response_format": kwargs.get("response_format"),
            }
            settings = {k: v for k, v in settings.items() if v is not None}
            return ChatGeneration(
                # TODOL add prompt id and variables
                provider="mistral",
                model=model,
                settings=settings,
                messages=messages,
            )
        # TODO: Add Completion API

    def update_step_after(
        generation: Union[ChatGeneration, CompletionGeneration], result
    ):
        if generation and isinstance(generation, ChatGeneration):
            generation.message_completion = result.choices[0].message.model_dump()
        elif generation and isinstance(generation, CompletionGeneration):
            if generation and generation.type == GenerationType.COMPLETION:
                generation.completion = result.choices[0].text

        if generation:
            generation.input_token_count = result.usage.prompt_tokens
            generation.output_token_count = result.usage.completion_tokens
            generation.token_count = result.usage.total_tokens


    def before_wrapper(metadata: Dict):
        def before(context: BeforeContext, *args, **kwargs):
            active_thread = active_thread_var.get()
            active_steps = active_steps_var.get()
            generation = init_generation(metadata["type"], kwargs)

            if (active_thread or active_steps): # TODO check callable on_new_gen
                step = client.start_step(
                    name=context["original_func"].__name__, type="llm"
                )
                step.name = generation.model or "mistral"
                if isinstance(generation, ChatGeneration):
                    step.input = {"content": generation.messages}
                else:
                    step.input = {"content": generation.prompt}

                context["step"] = step

            context["generation"] = generation
            context["start"] = time.time()

        return before
    
    def after_wrapper(metadata: Dict):
        # Needs to be done in a separate function to avoid transforming all returned data into generators
        def after(result, context: AfterContext, *args, **kwargs):
            step = context.get("step")
            generation = context.get("generation")

            if not generation:
                return

            if model := getattr(result, "model", None):
                generation.model = model
                if generation.settings:
                    generation.settings["model"] = model

            # TODO: handle streaming
            generation.duration = time.time() - context["start"]
            update_step_after(generation, result)

            # TODO: handle on gen callable
            if step:
                if isinstance(generation, ChatGeneration):
                    step.output = generation.message_completion  # type: ignore
                step.generation = generation
                step.end()
            else:
                client.api.create_generation(generation)
            return result

        return after

    wrap_sync(
        TO_WRAP,
        before_wrapper,
        after_wrapper,
    )

    is_mistral_instrumented = True
import time
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypedDict, Union

from literalai.helper import ensure_values_serializable
from literalai.my_types import ChatGeneration, CompletionGeneration, GenerationMessage
from literalai.step import Step

if TYPE_CHECKING:
    from uuid import UUID

    from literalai.client import LiteralClient
    from literalai.step import TrueStepType


def process_content(content: Any) -> Tuple[Dict, Optional[str]]:
    if content is None:
        return {}, None
    if isinstance(content, dict):
        return content, "json"
    elif isinstance(content, str):
        return {"content": content}, "text"
    else:
        return {"content": str(content)}, "text"


def get_langchain_callback():
    try:
        version("langchain")
    except Exception:
        raise Exception(
            "Please install langchain to use the langchain callback. "
            "You can install it with `pip install langchain`"
        )

    from langchain.callbacks.tracers.base import BaseTracer
    from langchain.callbacks.tracers.schemas import Run
    from langchain.schema import BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, GenerationChunk

    class ChatGenerationStart(TypedDict):
        input_messages: List[BaseMessage]
        start: float
        token_count: int
        tt_first_token: Optional[float]

    class CompletionGenerationStart(TypedDict):
        prompt: str
        start: float
        token_count: int
        tt_first_token: Optional[float]

    class GenerationHelper:
        chat_generations: Dict[str, ChatGenerationStart]
        completion_generations: Dict[str, CompletionGenerationStart]
        generation_inputs: Dict[str, Dict]

        def __init__(self) -> None:
            self.chat_generations = {}
            self.completion_generations = {}
            self.generation_inputs = {}

        def _convert_message_role(self, role: str):
            if "human" in role.lower():
                return "user"
            elif "system" in role.lower():
                return "system"
            elif "function" in role.lower():
                return "function"
            elif "tool" in role.lower():
                return "tool"
            else:
                return "assistant"

        def _convert_message_dict(
            self,
            message: Dict,
        ):
            class_name = message["id"][-1]
            kwargs = message.get("kwargs", {})
            function_call = kwargs.get("additional_kwargs", {}).get("function_call")

            msg = GenerationMessage(
                name=kwargs.get("name"),
                role=self._convert_message_role(class_name),
                content="",
            )

            if function_call:
                msg["function_call"] = function_call
            else:
                msg["content"] = kwargs.get("content", "")

            return msg

        def _convert_message(
            self,
            message: Union[Dict, BaseMessage],
        ):
            if isinstance(message, dict):
                return self._convert_message_dict(
                    message,
                )
            function_call = message.additional_kwargs.get("function_call")
            msg = GenerationMessage(
                name=getattr(message, "name", None),
                role=self._convert_message_role(message.type),
                content="",
            )

            if literal_uuid := message.additional_kwargs.get("uuid"):
                msg["uuid"] = literal_uuid
                msg["templated"] = True

            if function_call:
                msg["function_call"] = function_call
            else:
                msg["content"] = message.content  # type: ignore

            return msg

        def _build_llm_settings(
            self,
            serialized: Dict,
            invocation_params: Optional[Dict] = None,
        ):
            # invocation_params = run.extra.get("invocation_params")
            if invocation_params is None:
                return None, None

            provider = invocation_params.pop("_type", "")  # type: str

            model_kwargs = invocation_params.pop("model_kwargs", {})

            if model_kwargs is None:
                model_kwargs = {}

            merged = {
                **invocation_params,
                **model_kwargs,
                **serialized.get("kwargs", {}),
            }

            # make sure there is no api key specification
            settings = {k: v for k, v in merged.items() if not k.endswith("_api_key")}
            model_keys = ["azure_deployment", "deployment_name", "model", "model_name"]
            model = next((settings[k] for k in model_keys if k in settings), None)
            tools = None
            if "functions" in settings:
                tools = [
                    {"type": "function", "function": f} for f in settings["functions"]
                ]
            if "tools" in settings:
                tools = settings["tools"]
            return provider, model, tools, settings

    DEFAULT_TO_IGNORE = ["RunnableSequence", "RunnableParallel", "<lambda>"]
    DEFAULT_TO_KEEP = ["retriever", "llm", "agent", "chain", "tool"]

    class LangchainTracer(BaseTracer, GenerationHelper):
        steps: Dict[str, Step]
        parent_id_map: Dict[str, str]
        ignored_runs: set
        client: "LiteralClient"

        def __init__(
            self,
            client: "LiteralClient",
            # Runs to ignore to enhance readability
            to_ignore: Optional[List[str]] = None,
            # Runs to keep within ignored runs
            to_keep: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> None:
            BaseTracer.__init__(self, **kwargs)
            GenerationHelper.__init__(self)

            self.client = client
            self.steps = {}
            self.parent_id_map = {}
            self.ignored_runs = set()

            if to_ignore is None:
                self.to_ignore = DEFAULT_TO_IGNORE
            else:
                self.to_ignore = to_ignore

            if to_keep is None:
                self.to_keep = DEFAULT_TO_KEEP
            else:
                self.to_keep = to_keep

        def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: List[List[BaseMessage]],
            *,
            run_id: "UUID",
            parent_run_id: Optional["UUID"] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> Any:
            lc_messages = messages[0]
            self.chat_generations[str(run_id)] = {
                "input_messages": lc_messages,
                "start": time.time(),
                "token_count": 0,
                "tt_first_token": None,
            }

            return super().on_chat_model_start(
                serialized,
                messages,
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
                metadata=metadata,
                **kwargs,
            )

        def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            *,
            run_id: "UUID",
            tags: Optional[List[str]] = None,
            parent_run_id: Optional["UUID"] = None,
            metadata: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None,
            **kwargs: Any,
        ) -> Run:
            self.completion_generations[str(run_id)] = {
                "prompt": prompts[0],
                "start": time.time(),
                "token_count": 0,
                "tt_first_token": None,
            }
            return super().on_llm_start(
                serialized,
                prompts,
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
                metadata=metadata,
                name=name,
                **kwargs,
            )

        def on_llm_new_token(
            self,
            token: str,
            *,
            chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
            run_id: "UUID",
            parent_run_id: Optional["UUID"] = None,
            **kwargs: Any,
        ) -> Run:
            if isinstance(chunk, ChatGenerationChunk):
                start = self.chat_generations[str(run_id)]
            else:
                start = self.completion_generations[str(run_id)]  # type: ignore
            start["token_count"] += 1
            if start["tt_first_token"] is None:
                start["tt_first_token"] = (time.time() - start["start"]) * 1000

            return super().on_llm_new_token(
                token,
                chunk=chunk,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )

        def _persist_run(self, run: Run) -> None:
            pass

        def _get_run_parent_id(self, run: Run):
            parent_id = str(run.parent_run_id) if run.parent_run_id else None

            return parent_id

        def _get_non_ignored_parent_id(self, current_parent_id: Optional[str] = None):
            if not current_parent_id or current_parent_id not in self.parent_id_map:
                return None

            while current_parent_id in self.parent_id_map:
                # If the parent id is in the ignored runs, we need to get the parent id of the ignored run
                if current_parent_id in self.ignored_runs:
                    current_parent_id = self.parent_id_map[current_parent_id]
                else:
                    return current_parent_id

            return None

        def _should_ignore_run(self, run: Run):
            parent_id = self._get_run_parent_id(run)

            if parent_id:
                # Add the parent id of the ignored run in the mapping
                # so we can re-attach a kept child to the right parent id
                self.parent_id_map[str(run.id)] = parent_id

            ignore_by_name = False
            ignore_by_parent = parent_id in self.ignored_runs

            for filter in self.to_ignore:
                if filter in run.name:
                    ignore_by_name = True
                    break

            ignore = ignore_by_name or ignore_by_parent

            # If the ignore cause is the parent being ignored, check if we should nonetheless keep the child
            if ignore_by_parent and not ignore_by_name and run.run_type in self.to_keep:
                return False, self._get_non_ignored_parent_id(parent_id)
            else:
                if ignore:
                    # Tag the run as ignored
                    self.ignored_runs.add(str(run.id))
                return ignore, parent_id

        def _start_trace(self, run: Run) -> None:
            super()._start_trace(run)

            ignore, parent_id = self._should_ignore_run(run)

            if run.run_type in ["chain", "prompt"]:
                self.generation_inputs[str(run.id)] = ensure_values_serializable(
                    run.inputs
                )

            if ignore:
                return

            step_type: "TrueStepType" = "undefined"
            if run.run_type == "agent":
                step_type = "run"
            elif run.run_type == "chain":
                pass
            elif run.run_type == "llm":
                step_type = "llm"
            elif run.run_type == "retriever":
                step_type = "retrieval"
            elif run.run_type == "tool":
                step_type = "tool"
            elif run.run_type == "embedding":
                step_type = "embedding"

            if not self.steps and step_type != "llm":
                step_type = "run"

            step = self.client.start_step(
                id=str(run.id), name=run.name, type=step_type, parent_id=parent_id
            )
            step.input, language = process_content(run.inputs)
            if language is not None:
                if step.metadata is None:
                    step.metadata = {}
                step.metadata["language"] = language

            self.steps[str(run.id)] = step

        def _on_run_update(self, run: Run) -> None:
            """Process a run upon update."""

            ignore, parent_id = self._should_ignore_run(run)

            if ignore:
                return

            current_step = self.steps.get(str(run.id), None)

            if run.run_type == "llm" and current_step:
                provider, model, tools, llm_settings = self._build_llm_settings(
                    (run.serialized or {}), (run.extra or {}).get("invocation_params")
                )

                generations = (run.outputs or {}).get("generations", [])
                generation = generations[0][0]
                variables = self.generation_inputs.get(str(run.parent_run_id), {})
                if message := generation.get("message"):
                    chat_start = self.chat_generations[str(run.id)]
                    duration = time.time() - chat_start["start"]
                    if duration and chat_start["token_count"]:
                        throughput = chat_start["token_count"] / duration
                    else:
                        throughput = None
                    message_completion = self._convert_message(message)
                    current_step.generation = ChatGeneration(
                        provider=provider,
                        model=model,
                        tools=tools,
                        variables=variables,
                        settings=llm_settings,
                        duration=duration,
                        token_throughput_in_s=throughput,
                        tt_first_token=chat_start.get("tt_first_token"),
                        messages=[
                            self._convert_message(m)
                            for m in chat_start["input_messages"]
                        ],
                        message_completion=message_completion,
                    )
                    # find first message with prompt_id
                    prompt_id = None
                    variables_with_defaults: Optional[Dict] = None
                    for m in chat_start["input_messages"]:
                        if m.additional_kwargs.get("prompt_id"):
                            prompt_id = m.additional_kwargs["prompt_id"]
                            variables_with_defaults = m.additional_kwargs.get(
                                "variables"
                            )
                            break
                    if prompt_id:
                        current_step.generation.prompt_id = prompt_id
                    if variables_with_defaults:
                        current_step.generation.variables = variables_with_defaults

                    current_step.output = message_completion
                else:
                    completion_start = self.completion_generations[str(run.id)]
                    duration = time.time() - completion_start["start"]
                    if duration and completion_start["token_count"]:
                        throughput = completion_start["token_count"] / duration
                    else:
                        throughput = None
                    completion = generation.get("text", "")
                    current_step.generation = CompletionGeneration(
                        provider=provider,
                        model=model,
                        settings=llm_settings,
                        variables=variables,
                        duration=duration,
                        token_throughput_in_s=throughput,
                        tt_first_token=completion_start.get("tt_first_token"),
                        prompt=completion_start["prompt"],
                        completion=completion,
                    )
                    current_step.output = {"content": completion}

                if current_step:
                    if current_step.metadata is None:
                        current_step.metadata = {}
                    current_step.end()

                return

            outputs = run.outputs or {}
            output_keys = list(outputs.keys())
            output = outputs
            if output_keys:
                output = outputs.get(output_keys[0], outputs)

            if current_step:
                current_step.output, _ = process_content(output)
                current_step.end()

        def _on_error(self, error: BaseException, *, run_id: "UUID", **kwargs: Any):
            if current_step := self.steps.get(str(run_id), None):
                if current_step.metadata is None:
                    current_step.metadata = {}
                current_step.error = str(error)
                current_step.end()
                self.client.flush()

        on_llm_error = _on_error
        on_chain_error = _on_error
        on_tool_error = _on_error
        on_retriever_error = _on_error

    return LangchainTracer

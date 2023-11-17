# Draft, could be replaced by dynamically-generated code to stay in sync with the graphql schema

from datetime import datetime
from enum import Enum, unique
from typing import Dict, List, Optional


@unique
class PromptType(Enum):
    CHAT = "CHAT"
    COMPLETION = "COMPLETION"


@unique
class FeedbackStrategy(Enum):
    BINARY = "BINARY"
    STARS = "STARS"
    BIG_STARS = "BIG_STARS"
    LIKERT = "LIKERT"
    CONTINUOUS = "CONTINUOUS"
    LETTERS = "LETTERS"
    PERCENTAGE = "PERCENTAGE"


class ParticipantPayload:
    name: Optional[str]
    metadata: Optional[Dict]


class PromptPayload:
    provider: Optional[str]
    type: Optional[PromptType]
    settings: Optional[Dict]
    inputs: Optional[Dict]
    completion: Optional[str]
    template_format: Optional[str]
    token_count: Optional[int]
    template: Optional[str]
    formatted: Optional[str]
    messages: Optional[Dict]


class AttachmentPayload:
    type: Optional[str]
    name: Optional[str]
    display: Optional[str]
    mime: Optional[str]
    object_key: Optional[str]
    url: Optional[str]
    size: Optional[str]
    language: Optional[str]


class FeedbackPayload:
    value: Optional[int]
    strategy: Optional[FeedbackStrategy]
    comment: Optional[str]


class TracePayload:
    id: str
    metadata: Optional[Dict]
    tags: Optional[List[str]]


class StepOperation(Enum):
    # Assuming StepOperation is an Enum with predefined values
    pass


class StepPayload:
    id: str
    trace_id: str
    metadata: Optional[Dict]
    tags: Optional[List[str]]
    service_version: Optional[str]
    input: Optional[str]
    output: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    parent_id: Optional[str]
    operation: Optional[StepOperation]
    operator_name: Optional[str]
    participant: Optional[ParticipantPayload]
    prompt: Optional[PromptPayload]
    attachments: Optional[List[AttachmentPayload]]
    feedback: Optional[FeedbackPayload]
    trace: Optional[TracePayload]

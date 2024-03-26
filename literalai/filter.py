import sys

if sys.version_info < (3, 11):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

from typing import Any, Generic, Literal, Optional, TypeVar, Union

Field = TypeVar("Field")
Operators = TypeVar("Operators")
Value = TypeVar("Value")

BOOLEAN_OPERATORS = Literal["is", "nis"]
STRING_OPERATORS = Literal["eq", "neq", "ilike", "nilike"]
NUMBER_OPERATORS = Literal["eq", "neq", "gt", "gte", "lt", "lte"]
STRING_LIST_OPERATORS = Literal["in", "nin"]
DATETIME_OPERATORS = Literal["gte", "lte", "gt", "lt"]

OPERATORS = Union[
    BOOLEAN_OPERATORS,
    STRING_OPERATORS,
    NUMBER_OPERATORS,
    STRING_LIST_OPERATORS,
    DATETIME_OPERATORS,
]


class Filter(Generic[Field], TypedDict):
    field: Field
    operator: OPERATORS
    value: Any
    path: Optional[str]


class OrderBy(Generic[Field], TypedDict):
    column: Field
    direction: Literal["ASC", "DESC"]

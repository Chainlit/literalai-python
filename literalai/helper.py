from datetime import datetime

from pydantic import BaseModel


def filter_none_values(data):
    return {key: value for key, value in data.items() if value is not None}


def ensure_values_serializable(data):
    """
    Recursively ensures that all values in the input (dict or list) are JSON serializable.
    """

    if isinstance(data, BaseModel):
        return filter_none_values(data.model_dump())

    if isinstance(data, dict):
        return {key: ensure_values_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [ensure_values_serializable(item) for item in data]
    elif isinstance(data, (str, int, float, bool, type(None))):
        return data
    elif isinstance(data, (tuple, set)):
        return ensure_values_serializable(
            list(data)
        )  # Convert tuples and sets to lists
    else:
        return str(data)  # Fallback: convert other types to string


def utc_now():
    dt = datetime.utcnow()
    return dt.isoformat() + "Z"


def timestamp_utc(timestamp: float):
    dt = datetime.utcfromtimestamp(timestamp)
    return dt.isoformat() + "Z"

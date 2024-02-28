from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict


class DatasetDict(TypedDict, total=False):
    id: str
    createdAt: str
    metadata: Dict
    name: Optional[str]
    description: Optional[str]
    items: Optional[List[Dict]]


@dataclass
class Dataset:
    id: str
    created_at: str
    metadata: Dict
    name: Optional[str] = None
    description: Optional[str] = None
    items: Optional[List[Dict]] = None

    def to_dict(self):
        return {
            "id": self.id,
            "createdAt": self.created_at,
            "metadata": self.metadata,
            "name": self.name,
            "description": self.description,
            "items": self.items,
        }

    @classmethod
    def from_dict(cls, dataset: DatasetDict) -> "Dataset":
        return cls(
            id=dataset.get("id", ""),
            created_at=dataset.get("createdAt", ""),
            metadata=dataset.get("metadata", {}),
            name=dataset.get("name"),
            description=dataset.get("description"),
            items=dataset.get("items"),
        )

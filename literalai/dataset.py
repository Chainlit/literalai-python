import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

if TYPE_CHECKING:
    from literalai.api import API

from literalai.dataset_item import DatasetItemDict


class DatasetDict(TypedDict, total=False):
    id: str
    createdAt: str
    metadata: Dict
    name: Optional[str]
    description: Optional[str]
    items: Optional[List[DatasetItemDict]]


@dataclass
class Dataset:
    api: "API"
    id: str
    created_at: str
    metadata: Dict
    name: Optional[str] = None
    description: Optional[str] = None
    items: List[DatasetItemDict] = field(default_factory=lambda: [])

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
    def from_dict(cls, api: "API", dataset: DatasetDict) -> "Dataset":
        return cls(
            api=api,
            id=dataset.get("id", ""),
            created_at=dataset.get("createdAt", ""),
            metadata=dataset.get("metadata", {}),
            name=dataset.get("name"),
            description=dataset.get("description"),
            items=dataset.get("items") or [],
        )

    async def update(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        updated_dataset = await self.api.update_dataset(
            self.id, name=name, description=description, metadata=metadata
        )
        self.name = updated_dataset.name
        self.description = updated_dataset.description
        self.metadata = updated_dataset.metadata

    def update_sync(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        updated_dataset = self.api.update_dataset_sync(
            self.id, name=name, description=description, metadata=metadata
        )
        self.name = updated_dataset.name
        self.description = updated_dataset.description
        self.metadata = updated_dataset.metadata

    async def delete(self):
        await self.api.delete_dataset(self.id)

    def delete_sync(self):
        self.api.delete_dataset_sync(self.id)

    async def create_item(
        self,
        input: Dict,
        expected_output: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> DatasetItemDict:
        """
        Create a new dataset item and add it to this dataset.
        :param input: The input data for the dataset item.
        :param expected_output: The output data for the dataset item (optional).
        :param metadata: Metadata for the dataset item (optional).
        :return: The created DatasetItem instance.
        """
        dataset_item = await self.api.create_dataset_item(
            self.id, input, expected_output, metadata
        )
        if self.items is None:
            self.items = []
        dataset_item_dict = dataset_item.to_dict()
        self.items.append(dataset_item_dict)
        return dataset_item_dict

    def create_item_sync(
        self,
        input: Dict,
        expected_output: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> DatasetItemDict:
        """
        Synchronous version of the create_item method.
        """
        dataset_item = self.api.create_dataset_item_sync(
            self.id, input, expected_output, metadata
        )
        if self.items is None:
            self.items = []

        dataset_item_dict = dataset_item.to_dict()
        self.items.append(dataset_item_dict)
        return dataset_item_dict

    async def delete_item(self, item_id: str):
        """
        Delete a dataset item from this dataset.
        :param api: An instance of the DatasetAPI to make the call.
        :param item_id: The ID of the dataset item to delete.
        """
        await self.api.delete_dataset_item(item_id)
        if self.items is not None:
            self.items = [item for item in self.items if item["id"] != item_id]

    def delete_item_sync(self, item_id: str):
        """
        Synchronous version of the delete_item method.
        """
        self.api.delete_dataset_item_sync(item_id)
        if self.items is not None:
            self.items = [item for item in self.items if item["id"] != item_id]

    async def add_step(
        self, step_id: str, metadata: Optional[Dict] = None
    ) -> DatasetItemDict:
        """
        Create a new dataset item based on a step and add it to this dataset.
        :param step_id: The id of the step to add to the dataset.
        :param metadata: Metadata for the dataset item (optional).
        :return: The created DatasetItem instance.
        """
        dataset_item = await self.api.add_step_to_dataset(self.id, step_id, metadata)
        if self.items is None:
            self.items = []
        dataset_item_dict = dataset_item.to_dict()
        self.items.append(dataset_item_dict)
        return dataset_item_dict

    def add_step_sync(
        self, step_id: str, metadata: Optional[Dict] = None
    ) -> DatasetItemDict:
        """
        Synchronous version of the add_step method.
        """
        dataset_item = self.api.add_step_to_dataset_sync(self.id, step_id, metadata)
        if self.items is None:
            self.items = []
        dataset_item_dict = dataset_item.to_dict()
        self.items.append(dataset_item_dict)
        return dataset_item_dict

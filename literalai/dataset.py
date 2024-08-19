import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

from literalai.my_types import Utils

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

if TYPE_CHECKING:
    from literalai.api import LiteralAPI

from literalai.dataset_experiment import DatasetExperiment
from literalai.dataset_item import DatasetItem, DatasetItemDict

DatasetType = Literal["key_value", "generation"]


class DatasetDict(TypedDict, total=False):
    """
    A dictionary representation of a Dataset. All attributes are optional.

    Attributes:
        id (str): The unique identifier for the dataset.
        createdAt (str): The timestamp when the dataset was created.
        metadata (Dict[str, Any]): Additional metadata for the dataset.
        name (str): The name of the dataset.
        description (str): The description of the dataset.
        items (List[DatasetItemDict]): The items in the dataset.
        type (DatasetType): The type of the dataset.
    """
    id: str
    createdAt: str
    metadata: Dict
    name: Optional[str]
    description: Optional[str]
    items: Optional[List[DatasetItemDict]]
    type: DatasetType


@dataclass(repr=False)
class Dataset(Utils):
    """
    A class representing a dataset of items.

    Attributes:
        api (LiteralAPI): The API client to handle the dataset.
        id (str): The unique identifier for the dataset.
        created_at (str): The timestamp when the dataset was created.
        metadata (Dict[str, Any]): Additional metadata for the dataset.
        name (Optional[str]): The name of the dataset. Defaults to None.
        description (Optional[str]): The description of the dataset. Defaults to None.
        items (List[DatasetItem]): The items in the dataset. Defaults to [].
        type (DatasetType): The type of the dataset. Defaults to "key_value".
    """

    api: "LiteralAPI"
    id: str
    created_at: str
    metadata: Dict
    name: Optional[str] = None
    description: Optional[str] = None
    items: List[DatasetItem] = field(default_factory=lambda: [])
    type: DatasetType = "key_value"

    def to_dict(self):
        """
        Converts the Dataset object to a DatasetDict dictionary.

        Returns:
            DatasetDict: The dictionary representation of the Dataset object.
        """
        return {
            "id": self.id,
            "createdAt": self.created_at,
            "metadata": self.metadata,
            "name": self.name,
            "description": self.description,
            "items": [item.to_dict() for item in self.items],
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, api: "LiteralAPI", dataset: DatasetDict) -> "Dataset":
        """
        Creates a Dataset object from a DatasetDict dictionary.

        Args:
            api (LiteralAPI): The API client to handle the dataset.
            dataset (DatasetDict): The dictionary representation of the Dataset object.

        Returns:
            Dataset: The Dataset object created from the dictionary.
        """
        items = dataset.get("items", [])
        if not isinstance(items, list):
            raise Exception("Dataset items should be an array")

        return cls(
            api=api,
            id=dataset.get("id", ""),
            created_at=dataset.get("createdAt", ""),
            metadata=dataset.get("metadata", {}),
            name=dataset.get("name"),
            description=dataset.get("description"),
            items=[DatasetItem.from_dict(item) for item in items],
            type=dataset.get("type", "key_value"),
        )

    def update(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Updates the name, description, and metadata of the dataset.

        Args:
            name (Optional[str], optional): The new name for the dataset. Defaults to None.
            description (Optional[str], optional): The new description for the dataset. Defaults to None.
            metadata (Optional[Dict], optional): The new metadata for the dataset. Defaults to None.
        """
        updated_dataset = self.api.update_dataset(
            self.id, name=name, description=description, metadata=metadata
        )
        self.name = updated_dataset.name
        self.description = updated_dataset.description
        self.metadata = updated_dataset.metadata

    def delete(self):
        """
        Deletes the dataset.
        """
        self.api.delete_dataset(self.id)

    def create_item(
        self,
        input: Dict,
        expected_output: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> DatasetItem:
        """
        Create a new dataset item and add it to this dataset.

        Args:
            input (Dict): The input data for the dataset item.
            expected_output (Optional[Dict], optional): The output data for the dataset item. Defaults to None.
            metadata (Optional[Dict], optional): Metadata for the dataset item. Defaults to None.

        Returns:
            DatasetItem: The created DatasetItem instance.
        """
        dataset_item = self.api.create_dataset_item(
            self.id, input, expected_output, metadata
        )
        if self.items is None:
            self.items = []
        self.items.append(dataset_item)
        return dataset_item

    def create_experiment(
        self, name: str, prompt_id: Optional[str] = None, params: Optional[Dict] = None
    ) -> DatasetExperiment:
        """
        Creates a new dataset experiment based on this dataset.

        Args:
            name (str): The name of the experiment.
            prompt_id (Optional[str], optional): The Prompt ID used on LLM calls. Defaults to None.
            params (Optional[Dict], optional): The params used on the experiment. Defaults to None.

        Returns:
            DatasetExperiment: The created DatasetExperiment instance.
        """
        experiment = self.api.create_experiment(
            self.id, name, prompt_id, params
        )
        return experiment

    def delete_item(self, item_id: str):
        """
        Delete a dataset item from this dataset.

        Args:
            item_id (str): The ID of the dataset item to delete.
        """
        self.api.delete_dataset_item(item_id)
        if self.items is not None:
            self.items = [item for item in self.items if item.id != item_id]

    def add_step(
        self, step_id: str, metadata: Optional[Dict] = None
    ) -> DatasetItem:
        """
        Create a new dataset item based on a step and add it to this dataset.

        Args:
            step_id (str): The id of the step to add to the dataset.
            metadata (Optional[Dict], optional): Metadata for the dataset item. Defaults to None.

        Returns:
            DatasetItem: The created DatasetItem instance.

        Raises:
            ValueError: If the dataset type is "generation".
        """
        if self.type == "generation":
            raise ValueError("Cannot add a step to a generation dataset")

        dataset_item = self.api.add_step_to_dataset(self.id, step_id, metadata)
        if self.items is None:
            self.items = []
        self.items.append(dataset_item)
        return dataset_item

    def add_generation(
        self, generation_id: str, metadata: Optional[Dict] = None
    ) -> DatasetItem:
        """
        Create a new dataset item based on a generation and add it to this dataset.

        Args:
            generation_id (str): The id of the generation to add to the dataset.
            metadata (Optional[Dict], optional): Metadata for the dataset item. Defaults to None.

        Returns:
            DatasetItem: The created DatasetItem instance.
        """
        dataset_item = self.api.add_generation_to_dataset(
            self.id, generation_id, metadata
        )
        if self.items is None:
            self.items = []
        self.items.append(dataset_item)
        return dataset_item

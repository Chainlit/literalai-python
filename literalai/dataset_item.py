from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict

from literalai.my_types import Utils


class DatasetItemDict(TypedDict, total=False):
    """
    A dictionary representation of a DatasetItem.

    Attributes (all optional):
        id (str): The unique identifier for the dataset item.
        createdAt (str): The timestamp when the dataset item was created.
        datasetId (str): The identifier of the dataset the item belongs to.
        metadata (Dict[str, Any]): Additional metadata for the dataset item.
        input (Dict[str, Any]): The input data for the dataset item.
        expectedOutput (Dict[str, Any]): The expected output data for the dataset item.
        intermediarySteps (List[Dict[str, Any]]): The intermediary steps for the dataset item.
    """
    id: str
    createdAt: str
    datasetId: str
    metadata: Dict
    input: Dict
    expectedOutput: Optional[Dict]
    intermediarySteps: Optional[List[Dict]]


@dataclass(repr=False)
class DatasetItem(Utils):
    """
    A class representing a dataset item.

    Attributes:
        id (str): The unique identifier for the dataset item.
        created_at (str): The timestamp when the dataset item was created.
        dataset_id (str): The identifier of the dataset the item belongs to.
        metadata (Dict[str, Any]): Additional metadata for the dataset item.
        input (Dict[str, Any]): The input data for the dataset item.
        expected_output (Optional[Dict[str, Any]]): The expected output data for the dataset item. Defaults to None.
        intermediary_steps (Optional[List[Dict[str, Any]]]): The intermediary steps for the dataset item. Defaults to None.
    """
    id: str
    created_at: str
    dataset_id: str
    metadata: Dict
    input: Dict
    expected_output: Optional[Dict] = None
    intermediary_steps: Optional[List[Dict]] = None

    def to_dict(self):
        """
        Converts the DatasetItem object to a DatasetItemDict dictionary.

        Returns:
            DatasetItemDict: The dictionary representation of the DatasetItem object.
        """
        return {
            "id": self.id,
            "createdAt": self.created_at,
            "datasetId": self.dataset_id,
            "metadata": self.metadata,
            "input": self.input,
            "expectedOutput": self.expected_output,
            "intermediarySteps": self.intermediary_steps,
        }

    @classmethod
    def from_dict(cls, dataset_item: DatasetItemDict) -> "DatasetItem":
        """
        Creates a DatasetItem object from a DatasetItemDict dictionary.

        Args:
            dataset_item (DatasetItemDict): The dictionary representation of the DatasetItem object.

        Returns:
            DatasetItem: The DatasetItem object created from the dictionary.
        """
        return cls(
            id=dataset_item.get("id", ""),
            created_at=dataset_item.get("createdAt", ""),
            dataset_id=dataset_item.get("datasetId", ""),
            metadata=dataset_item.get("metadata", {}),
            input=dataset_item.get("input", {}),
            expected_output=dataset_item.get("expectedOutput"),
            intermediary_steps=dataset_item.get("intermediarySteps"),
        )

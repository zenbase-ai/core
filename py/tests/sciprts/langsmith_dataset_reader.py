from typing import Any, Dict

from dotenv import load_dotenv
from langsmith import Client

load_dotenv("../../../py/.env.test", override=True)


def get_langsmith_dataset(dataset_name: str) -> Dict[str, Any]:
    # Initialize LangSmith client
    client = Client()

    # Fetch the dataset by name
    datasets = client.list_datasets(dataset_name=dataset_name)
    if not datasets:
        raise ValueError(f"Dataset '{dataset_name}' not found")

    dataset = [i for i in datasets][0]  # Assume the first match is the one we want

    # Fetch all examples for this dataset
    examples = list(client.list_examples(dataset_id=dataset.id))

    # Prepare the return dictionary
    dataset_dict = {
        "dataset_info": {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "created_at": str(dataset.created_at),
        },
        "examples": [],
    }

    # Add examples to the dictionary
    for example in examples:
        dataset_dict["examples"].append(
            {
                "id": example.id,
                "inputs": example.inputs,
                "outputs": example.outputs,
                "created_at": str(example.created_at),
            }
        )

    return dataset_dict


# Usage example
dataset_dict = get_langsmith_dataset("GSM8K_Dataset_2iWgvn2G0fnxxkInl5cCfoBL2SY")
print(f"Dataset '{dataset_dict['dataset_info']['name']}' retrieved successfully.")
print(f"Number of examples: {len(dataset_dict['examples'])}")

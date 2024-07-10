from dotenv import load_dotenv
from langsmith import Client


def remove_all_datasets():
    # Initialize LangSmith client
    client = Client()

    # Fetch all datasets
    datasets = list(client.list_datasets())

    if not datasets:
        print("No datasets found in LangSmith.")
        return

    # Confirm with the user
    confirm = input(f"Are you sure you want to delete all {len(datasets)} datasets? (yes/no): ")
    if confirm.lower() != "yes":
        print("Operation cancelled.")
        return

    # Delete each dataset
    for dataset in datasets:
        try:
            client.delete_dataset(dataset_id=dataset.id)
            print(f"Deleted dataset: {dataset.name} (ID: {dataset.id})")
        except Exception as e:
            print(f"Error deleting dataset {dataset.name} (ID: {dataset.id}): {str(e)}")

    print("All datasets have been deleted.")


# Run the function
if __name__ == "__main__":
    load_dotenv("../../.env.test")
    remove_all_datasets()

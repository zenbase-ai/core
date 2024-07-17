from pathlib import Path

# Define the base directory as the parent of the parent directory of this file
BASE_DIR = Path(__file__).resolve().parent.parent

# Define the test directory as a subdirectory of the base directory's parent
TEST_DIR = BASE_DIR.parent / "tests"

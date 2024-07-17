import os

import nbformat


def fix_notebooks_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ipynb"):
                notebook_path = os.path.join(root, file)
                print(f"Fixing notebook: {notebook_path}")

                with open(notebook_path, "r", encoding="utf-8") as f:
                    notebook = nbformat.read(f, as_version=4)

                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code" and "execution_count" not in cell:
                        cell["execution_count"] = None

                with open(notebook_path, "w", encoding="utf-8") as f:
                    nbformat.write(notebook, f)

                print(f"Fixed notebook: {notebook_path}")


fix_notebooks_in_directory("../../cookbooks")

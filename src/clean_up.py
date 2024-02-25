"""Script to clean up the data directory by deleting all the newly generated files.
"""
import os

from config import DATA


if __name__ == "__main__":
    file_paths = [
        DATA["cond"], DATA["corrected"], DATA["masks"],
        DATA["preds"], DATA["scores"], DATA["trues"],
    ]  # List of file paths to delete
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print(f"The file {file_path} does not exist")

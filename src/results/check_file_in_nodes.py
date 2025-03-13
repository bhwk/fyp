import os
import json


def check_file_in_nodes(json_file_path):
    match_count = 0
    file_count = 0
    with open(json_file_path, "r") as file:
        data = json.load(file)

    for entry in data:
        file_path = entry.get("file")
        nodes = entry.get("nodes", [])

        if file_path and nodes:
            file_name = os.path.basename(file_path)
            if file_name in [node for node in nodes]:
                match_count += 1
        file_count += 1

    return match_count, file_count


def check_directory(directory_path):
    total_matches = 0
    total_file_count = (
        0  # Renamed to avoid confusion with file_count inside check_file_in_nodes
    )
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".json"):
                json_file_path = os.path.join(root, file)
                matches, file_count = check_file_in_nodes(json_file_path)
                total_matches += matches
                total_file_count += file_count  # Correct accumulation here

    print(
        f"Total matches across all files in the directory: {total_matches}/{total_file_count}"
    )


# Example usage
directory_path = "."
check_directory(directory_path)

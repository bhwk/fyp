import json
import argparse


def is_file_in_nodes(data):
    """Check if the file name in the 'file' field is present in the 'nodes' array for each entry."""
    results = {}
    total_matches = 0
    total_files = 0

    for entry in data:
        file_name = entry["file"].split("/")[-1]  # Extract file name from path
        total_files += 1
        if file_name in entry.get("nodes", []):
            results[file_name] = True
            total_matches += 1
        else:
            results[file_name] = False

    return results, total_files, total_matches


def check_file_in_json(json_file):
    """Load JSON data from file and check if each file is in its respective nodes array."""
    with open(json_file, "r") as f:
        data = json.load(f)

    # Check if each file is in its respective nodes array
    results, total_files, total_matches = is_file_in_nodes(data)

    # Print results
    for file, found in results.items():
        print(f"File '{file}' found in nodes: {found}")

    # Print totals
    print(f"\nTotal files: {total_files}")
    print(f"Total matches: {total_matches}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Check if the file name in the 'file' field is present in the 'nodes' array."
    )
    parser.add_argument("json_file", help="Path to the input JSON file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided file
    check_file_in_json(args.json_file)


if __name__ == "__main__":
    main()

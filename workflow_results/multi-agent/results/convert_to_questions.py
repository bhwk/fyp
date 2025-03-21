import os
import json
import re


def convert_to_question_format(data):
    result = {}
    for item in data:
        file = item["file"]
        query = item["query"]

        if file not in result:
            result[file] = []
        result[file].append(query)

    formatted_result = [
        {"file": file, "questions": questions} for file, questions in result.items()
    ]
    return {"files": formatted_result}


def process_directory(directory):
    combined_data = []
    json_files = [f for f in os.listdir(directory) if re.match(r"batch_\d+\.json$", f)]
    json_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))

    for filename in json_files:
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
                combined_data.extend(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {filename}: {e}")
    return combined_data


def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    data = process_directory(directory)
    formatted_output = convert_to_question_format(data)
    output_file = os.path.join(directory, "formatted_output.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_output, f, indent=4)
    print(f"Formatted data written to {output_file}")


if __name__ == "__main__":
    main()

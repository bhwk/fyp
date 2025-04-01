import json
import evaluate
import torch
import bert_score
from transformers import AutoTokenizer, AutoModel


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_responses(data):
    """Extract 'information' as references and 'response' as predictions, filtering out empty responses."""
    filtered_data = [item for item in data if item.get("response")]
    references = [" ".join(item["information"]) for item in filtered_data]
    predictions = [item["response"] for item in filtered_data]
    return references, predictions


def compute_bleu_rouge(references, predictions):
    """Compute BLEU and ROUGE scores."""
    rouge = evaluate.load("rouge")

    # For ROUGE, keep the list of lists structure
    rouge_score = rouge.compute(predictions=predictions, references=references)

    return rouge_score


def main(reference_file):
    """Load responses, compute scores, and print results."""
    reference_data = load_json(reference_file)

    references, predictions = extract_responses(reference_data)

    if len(references) != len(predictions):
        print("Warning: The number of reference and predicted responses do not match!")

    if not references or not predictions:
        print(
            "No valid entries with non-empty responses found for either references or predictions."
        )
        return

    rouge_score = compute_bleu_rouge(references, predictions)

    print("ROUGE Scores:", rouge_score)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute BLEU, ROUGE, between system response and information"
    )
    parser.add_argument(
        "reference_file",
        type=str,
        help="Path to the JSON file.",
    )

    args = parser.parse_args()
    main(args.reference_file)


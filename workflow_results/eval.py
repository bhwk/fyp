import evaluate
import json


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_responses(data):
    return [item["response"] for item in data]


def compute_metrics(references, predictions):
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    rouge_metric = rouge_metric.compute(predictions=predictions, references=references)

    return bleu_score, rouge_metric


def main(reference_file, prediction_file):
    reference_data = load_json(reference_file)
    prediction_data = load_json(prediction_file)

    references = extract_responses(reference_data)
    predictions = extract_responses(prediction_data)

    if len(references) != len(predictions):
        print("Lengths not the same")

    bleu_score, rouge_score = compute_metrics(references, predictions)

    print(f"BLEU Score: {bleu_score['bleu'] * 100:.2f}")  # type: ignore
    print(f"ROUGE-1 F1 Score: {rouge_score['rouge1']:.2f}")  # type: ignore
    print(f"ROUGE-L F1 Score: {rouge_score['rougeL']:.2f}")  # type: ignore


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare responses using BLEU and ROUGE metrics."
    )

    parser.add_argument(
        "reference_file",
        type=str,
        help="Path to the JSON file containing reference responses.",
    )
    parser.add_argument(
        "prediction_file",
        type=str,
        help="Path to the JSON file containing predicted responses.",
    )

    args = parser.parse_args()
    main(args.reference_file, args.prediction_file)

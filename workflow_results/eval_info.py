import json
import evaluate
import torch
import torch.nn.functional as F
import bert_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_information(data):
    """Extract 'information' and 'synthesized information' fields, excluding missing ones."""
    valid_pairs = [
        (
            " ".join(item.get("information", "")),
            item.get("synthesized information", "").strip(),
        )
        for item in data
    ]
    # Filter out empty references and corresponding synthesized information
    filtered_pairs = [
        (info, synth_info) for info, synth_info in valid_pairs if info and synth_info
    ]

    return zip(*filtered_pairs)


def compute_bleu_rouge(references, predictions):
    """Compute BLEU and ROUGE scores."""
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)

    return bleu_score, rouge_score


def compute_bert_score(references, predictions):
    from statistics import mean

    """Compute BERTScore using the evaluate package."""
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(
        predictions=predictions, references=references, lang="en"
    )
    return {
        "precision": mean(results["precision"]),  # type: ignore
        "recall": mean(results["recall"]),  # type: ignore
        "f1": mean(results["f1"]),  # type: ignore
    }


def compute_embeddings(texts, model, tokenizer, device):
    # Tokenize the input texts
    inputs = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Forward pass to get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract token embeddings and attention mask
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]

    # Perform mean pooling
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask

    return mean_pooled.cpu().numpy()


def compute_sem_score(
    references, predictions, model_name="sentence-transformers/all-mpnet-base-v2"
):
    """Compute SemScore using cosine similarity of mean-pooled embeddings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # Compute embeddings for references and predictions
    ref_embeddings = compute_embeddings(references, model, tokenizer, device)
    pred_embeddings = compute_embeddings(predictions, model, tokenizer, device)

    # Compute cosine similarity between corresponding embeddings
    scores = [
        cosine_similarity([pred], [ref])[0][0]
        for pred, ref in zip(pred_embeddings, ref_embeddings)
    ]
    return sum(scores) / len(scores)


def main(reference_file):
    """Load information, compute scores, and print results."""
    data = load_json(reference_file)

    information, synthesized_information = extract_information(data)
    if not information:
        print("No valid synthesized information found. Exiting.")
        return

    bleu_score, rouge_score = compute_bleu_rouge(information, synthesized_information)
    bert_score_result = compute_bert_score(information, synthesized_information)
    sem_score = compute_sem_score(information, synthesized_information)

    print("BLEU Score:", bleu_score)
    print("ROUGE Scores:", rouge_score)
    print("BERTScore:", bert_score_result)
    print(f"SemScore: {sem_score:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute BLEU, ROUGE, BERTScore, and SemScore between information and synthesized information."
    )
    parser.add_argument(
        "reference_file",
        type=str,
        help="Path to the JSON file with reference information.",
    )

    args = parser.parse_args()
    main(args.reference_file)

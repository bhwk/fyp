import json
import evaluate
import torch
import bert_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_responses(data):
    """Extract only the 'response' field from the JSON objects."""
    return [item["response"] for item in data]


def extract_queries(data):
    """Extract 'query' and 'synthetic query' fields, excluding missing ones."""
    valid_pairs = [
        (item["query"], item.get("synthetic query", "").strip()) for item in data
    ]
    return zip(*[(q, sq) for q, sq in valid_pairs if sq])


def compute_bleu_rouge(references, predictions):
    """Compute BLEU and ROUGE scores."""
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)

    return bleu_score, rouge_score


def compute_bert_score(references, predictions):
    """Compute BERTScore using the bert_score package."""
    P, R, F1 = bert_score.score(
        predictions, references, lang="en", rescale_with_baseline=True
    )
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }


def compute_embeddings(texts, model, tokenizer, device):
    """Compute embeddings for a list of texts."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
        device
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def compute_sem_score(
    references, predictions, model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    """Compute SemScore using cosine similarity of embeddings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    ref_embeddings = compute_embeddings(references, model, tokenizer, device)
    pred_embeddings = compute_embeddings(predictions, model, tokenizer, device)

    scores = cosine_similarity(pred_embeddings, ref_embeddings)
    return scores.diagonal().mean()


def main(reference_file):
    """Load queries, compute scores, and print results."""
    data = load_json(reference_file)

    queries, synthetic_queries = extract_queries(data)
    if not queries:
        print("No valid synthetic queries found. Exiting.")
        return

    bleu_score, rouge_score = compute_bleu_rouge(queries, synthetic_queries)
    bert_score_result = compute_bert_score(queries, synthetic_queries)
    sem_score = compute_sem_score(queries, synthetic_queries)

    print("BLEU Score:", bleu_score)
    print("ROUGE Scores:", rouge_score)
    print("BERTScore:", bert_score_result)
    print(f"SemScore: {sem_score:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute BLEU, ROUGE, BERTScore, and SemScore between queries and synthetic queries."
    )
    parser.add_argument(
        "reference_file", type=str, help="Path to the JSON file with reference queries."
    )

    args = parser.parse_args()
    main(args.reference_file)

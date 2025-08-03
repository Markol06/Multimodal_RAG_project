import json

import numpy as np

from config import QUERY_EXPANSION_PATH, TOP_K
from rag import retriever_instance

def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    return len(set(retrieved_k) & set(relevant)) / len(retrieved_k)

def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / len(relevant)

def mean_reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def average_precision(retrieved: list[str], relevant: list[str]) -> float:
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)

def top1_accuracy(retrieved: list[str], relevant: list[str]) -> float:
    if not retrieved:
        return 0.0
    return 1.0 if retrieved[0] in relevant else 0.0

def evaluate_system():
    retriever = retriever_instance

    with open(QUERY_EXPANSION_PATH, "r", encoding="utf-8") as f:
        test_queries: dict[str, list[str]] = json.load(f)

    metrics = {
        "precision": [],
        "recall": [],
        "mrr": [],
        "ndcg": [],
        "map": [],
        "top1_acc": []
    }

    per_query_results = {}

    for query, relevant_ids in test_queries.items():
        results = retriever.search_multimodal(query, top_k=TOP_K)
        retrieved_ids = [r["id"] for r in results["text"]]

        p = precision_at_k(retrieved_ids, relevant_ids, TOP_K)
        r = recall_at_k(retrieved_ids, relevant_ids, TOP_K)
        mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, TOP_K)
        ap = average_precision(retrieved_ids, relevant_ids)
        t1 = top1_accuracy(retrieved_ids, relevant_ids)

        metrics["precision"].append(p)
        metrics["recall"].append(r)
        metrics["mrr"].append(mrr)
        metrics["ndcg"].append(ndcg)
        metrics["map"].append(ap)
        metrics["top1_acc"].append(t1)

        per_query_results[query] = {
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            f"precision@{TOP_K}": p,
            f"recall@{TOP_K}": r,
            "mrr": mrr,
            f"ndcg@{TOP_K}": ndcg,
            "ap": ap,
            "top1_accuracy": t1
        }

    avg_metrics = {
        f"mean_precision@{TOP_K}": np.mean(metrics["precision"]),
        f"mean_recall@{TOP_K}": np.mean(metrics["recall"]),
        "mean_mrr": np.mean(metrics["mrr"]),
        f"mean_ndcg@{TOP_K}": np.mean(metrics["ndcg"]),
        "map": np.mean(metrics["map"]),
        "mean_top1_accuracy": np.mean(metrics["top1_acc"])
    }

    results_to_save = {
        "per_query": per_query_results,
        "average_metrics": avg_metrics
    }

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)

    print("\n=== AVERAGE METRICS ===")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    evaluate_system()

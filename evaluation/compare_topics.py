#!/usr/bin/env python3
"""
Compare predicted topics from TB/LG systems against gold standard.

Calculates precision, recall, and F1 score using exact string matching
after normalization to snake_case.

Usage:
    python -m evaluation.compare_topics \
        --pred evaluation/runs/bench_finance_*.jsonl \
        --gold evaluation/finance/data/finance_gold.jsonl \
        --out evaluation/runs/fin_quality_XXXX.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Any, Optional


def normalize_topic(topic: str) -> str:
    """
    Normalize topic string to snake_case for consistent matching.
    
    Examples:
        "Acme Corp Q2 revenue growth" -> "acme_corp_q2_revenue_growth"
        "Full-Year EPS Guidance" -> "full_year_eps_guidance"
        "AI chip demand" -> "ai_chip_demand"
    """
    # Convert to lowercase
    topic = topic.lower().strip()
    
    # Replace multiple spaces/hyphens with single space
    topic = re.sub(r'[\s\-]+', '_', topic)
    
    # Remove special characters except underscores
    topic = re.sub(r'[^\w_]', '', topic)
    
    # Remove multiple underscores
    topic = re.sub(r'_+', '_', topic)
    
    # Strip leading/trailing underscores
    topic = topic.strip('_')
    
    return topic


def parse_topic_list(topic_str: str) -> List[str]:
    """
    Parse semicolon-separated topic string from benchmark output.
    
    Args:
        topic_str: String like "earnings;strategy;markets"
    
    Returns:
        List of normalized topic strings
    """
    if not topic_str or not isinstance(topic_str, str):
        return []
    
    topics = [t.strip() for t in topic_str.split(';') if t.strip()]
    return [normalize_topic(t) for t in topics]


def load_gold_topics(gold_path: Path) -> Dict[str, Set[str]]:
    """
    Load gold standard topics from JSONL file.
    
    Expected format:
        {"id": "F001", "topics": ["earnings", "guidance", "margin"]}
    
    Returns:
        Dict mapping item_id -> set of normalized topics
    """
    gold_topics: Dict[str, Set[str]] = {}
    
    with open(gold_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            item = json.loads(line)
            item_id = item.get("id")
            topics = item.get("topics", [])
            
            if item_id:
                gold_topics[item_id] = {normalize_topic(t) for t in topics if t}
    
    return gold_topics


def load_predictions(jsonl_paths: List[Path]) -> Dict[str, Set[str]]:
    """
    Load predicted topics from benchmark output JSONL files.
    
    Args:
        jsonl_paths: List of paths to bench_finance_*.jsonl files
    
    Returns:
        Dict mapping item_id -> set of normalized predicted topics
    """
    predictions: Dict[str, Set[str]] = defaultdict(set)
    
    for path in jsonl_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                item = json.loads(line)
                item_id = item.get("item_id")
                item_system = item.get("system", "")
                topic_str = item.get("topic", "")
                
                # Only process TB system entries
                if item_system != "TB":
                    continue
                
                if item_id:
                    topics = parse_topic_list(topic_str)
                    predictions[item_id].update(topics)
    
    return predictions


def calculate_metrics(
    gold: Set[str],
    predicted: Set[str]
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        gold: Set of gold standard topics (normalized)
        predicted: Set of predicted topics (normalized)
    
    Returns:
        Dict with keys: precision, recall, f1, tp, fp, fn
    """
    # True positives: topics in both gold and predicted
    tp = len(gold & predicted)
    
    # False positives: topics in predicted but not in gold
    fp = len(predicted - gold)
    
    # False negatives: topics in gold but not in predicted
    fn = len(gold - predicted)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compare_topics(
    gold_path: Path,
    jsonl_paths: List[Path],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Main comparison function.
    
    Returns:
        Dict with global metrics and per-item metrics
    """
    # Load data
    gold_topics = load_gold_topics(gold_path)
    predicted_topics = load_predictions(jsonl_paths)
    
    # Calculate per-item metrics
    per_item_metrics: List[Dict[str, Any]] = []
    global_tp = 0
    global_fp = 0
    global_fn = 0
    
    # Get only item IDs that are in predictions (benchmark file)
    # No point in comparing items that weren't processed
    all_item_ids = set(predicted_topics.keys())
    
    for item_id in sorted(all_item_ids):
        gold = gold_topics.get(item_id, set())
        predicted = predicted_topics.get(item_id, set())
        
        metrics = calculate_metrics(gold, predicted)
        
        # Accumulate for global metrics
        global_tp += metrics["tp"]
        global_fp += metrics["fp"]
        global_fn += metrics["fn"]
        
        per_item_metrics.append({
            "item_id": item_id,
            "gold_topics": sorted(gold),
            "predicted_topics": sorted(predicted),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })
    
    # Calculate global metrics
    global_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    global_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0
    
    return {
        "system": "TB",
        "global_metrics": {
            "precision": global_precision,
            "recall": global_recall,
            "f1": global_f1,
            "tp": global_tp,
            "fp": global_fp,
            "fn": global_fn,
            "n_items": len(per_item_metrics),
        },
        "per_item_metrics": per_item_metrics if verbose else [],
    }


def print_results(results: Dict[str, Any], verbose: bool = False) -> None:
    """
    Pretty print comparison results.
    """
    system = results["system"]
    global_metrics = results["global_metrics"]
    
    print(f"\n{'='*70}")
    print(f"Topic Comparison Results - System: {system}")
    print(f"{'='*70}\n")
    
    print(f"Global Metrics (across {global_metrics['n_items']} items):")
    print(f"  Precision: {global_metrics['precision']:.3f}")
    print(f"  Recall:    {global_metrics['recall']:.3f}")
    print(f"  F1 Score:  {global_metrics['f1']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {global_metrics['tp']}")
    print(f"  False Positives: {global_metrics['fp']}")
    print(f"  False Negatives: {global_metrics['fn']}")
    
    if verbose and results["per_item_metrics"]:
        print(f"\n{'-'*70}")
        print("Per-Item Metrics:")
        print(f"{'-'*70}\n")
        
        for item in results["per_item_metrics"]:
            print(f"Item: {item['item_id']}")
            print(f"  Gold:      {', '.join(item['gold_topics']) or '(none)'}")
            print(f"  Predicted: {', '.join(item['predicted_topics']) or '(none)'}")
            print(f"  Precision: {item['precision']:.3f} | Recall: {item['recall']:.3f} | F1: {item['f1']:.3f}")
            print(f"  TP: {item['tp']} | FP: {item['fp']} | FN: {item['fn']}")
            print()
    
    print(f"{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare predicted topics against gold standard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--pred",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to benchmark output JSONL files (supports wildcards)"
    )
    parser.add_argument(
        "--gold",
        type=str,
        required=True,
        help="Path to gold standard JSONL file"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-item metrics in addition to global metrics"
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Output JSON file path (optional)"
    )
    
    args = parser.parse_args()
    
    # Convert paths
    gold_path = Path(args.gold)
    jsonl_paths = [Path(p) for p in args.pred]
    
    # Validate paths
    if not gold_path.exists():
        print(f"Error: Gold file not found: {gold_path}")
        return
    
    for path in jsonl_paths:
        if not path.exists():
            print(f"Error: Benchmark file not found: {path}")
            return
    
    # Run comparison (TB only)
    results = compare_topics(gold_path, jsonl_paths, args.verbose)
    print_results(results, args.verbose)
    
    # Save to file if requested
    if args.out:
        output_path = Path(args.out)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

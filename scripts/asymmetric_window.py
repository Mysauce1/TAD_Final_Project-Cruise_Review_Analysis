import os
import pandas as pd
from scripts.utils import load_jsonl, split_sentences, detect_ports, get_sentiment

# Resolve base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input data path (JSONL file)
DATA_PATH = os.path.join(BASE_DIR, "data", "cruise_reviews.jsonl")

# Output directory for this approach
RESULT_DIR = os.path.join(BASE_DIR, "results", "asymmetric_window")

# Ensure results folder exists
os.makedirs(RESULT_DIR, exist_ok=True)

# Window size before and after port mention
BEFORE = 1
AFTER = 2


def run():
    """
    Execute asymmetric window propagation approach.

    Logic:
    - Assign sentences within ±window of port mentions
    - Allow overlapping port influence
    """

    # Load dataset into DataFrame
    df = load_jsonl(DATA_PATH)

    records = []

    # Process each review independently
    for _, row in df.iterrows():
        review_id = row["review_id"]

        # Split review into sentences
        sentences = split_sentences(row["review"])

        # Map sentence index to detected ports
        port_positions = {}

        for i, sent in enumerate(sentences):
            ports = detect_ports(sent)
            if ports:
                for p in ports:
                    port_positions.setdefault(i, []).append(p)

        # Assign sentiment using asymmetric window
        for i, sent in enumerate(sentences):

            sentiment = get_sentiment(sent)

            # Check window around each sentence
            for idx in range(i - BEFORE, i + AFTER + 1):

                if idx < 0 or idx >= len(sentences):
                    continue

                if idx in port_positions:
                    for p in port_positions[idx]:
                        records.append(
                            {
                                "review_id": review_id,
                                "port": p,
                                "sentence": sent,
                                "sentiment": sentiment,
                            }
                        )

    # Convert to DataFrame
    out = pd.DataFrame(records)

    # Compute sentiment per port
    summary = out.groupby("port")["sentiment"].mean().reset_index()
    summary.rename(columns={"sentiment": "avg_sentiment"}, inplace=True)

    summary.to_csv(os.path.join(RESULT_DIR, "avg_sentiment_by_port.csv"), index=False)

    # Compute proportions
    port_counts = out.groupby("port").size().reset_index(name="count")

    total = port_counts["count"].sum()
    port_counts["proportion"] = port_counts["count"] / total

    port_counts.to_csv(os.path.join(RESULT_DIR, "port_proportions.csv"), index=False)

    # Save summary stats
    summary_stats = pd.DataFrame(
        [
            {
                "total_assignments": len(out),
                "unique_sentences": out["sentence"].nunique(),
                "unique_reviews": out["review_id"].nunique(),
            }
        ]
    )

    summary_stats.to_csv(os.path.join(RESULT_DIR, "method_summary.csv"), index=False)

    # Save detailed output
    out.to_csv(os.path.join(RESULT_DIR, "detailed_output.csv"), index=False)


if __name__ == "__main__":
    run()

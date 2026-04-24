import os
import pandas as pd
from scripts.utils import (
    load_jsonl,
    split_sentences,
    split_clauses,
    detect_ports,
    get_sentiment,
)

# Resolve base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input data path (JSONL file)
DATA_PATH = os.path.join(BASE_DIR, "data", "cruise_reviews.jsonl")

# Output directory for this approach
RESULT_DIR = os.path.join(BASE_DIR, "results", "until_next_port")

# Ensure results folder exists
os.makedirs(RESULT_DIR, exist_ok=True)


def run():
    """
    Execute the "until next port" propagation approach.

    Logic:
    - Maintain active port context
    - Assign all sentences until a new port appears
    - Clause splitting improves attribution granularity
    """

    # Load dataset into DataFrame
    df = load_jsonl(DATA_PATH)

    records = []

    # Process each review independently
    for _, row in df.iterrows():
        review_id = row["review_id"]

        # Split review into sentences
        sentences = split_sentences(row["review"])

        # Track most recent active port(s)
        current_ports = []

        for sent in sentences:

            # Split sentence into clauses for finer-grained attribution
            clauses = split_clauses(sent)

            for clause in clauses:
                text = clause["text"]
                ports = clause["ports"]

                # Compute sentiment for clause
                sentiment = get_sentiment(text)

                # Update active port context if ports are detected
                if ports:
                    current_ports = ports

                # Skip until a port context exists
                elif not current_ports:
                    continue

                # Assign clause to active ports
                for p in current_ports:
                    records.append(
                        {
                            "review_id": review_id,
                            "port": p,
                            "sentence": text,
                            "sentiment": sentiment,
                        }
                    )

    # Convert to DataFrame
    out = pd.DataFrame(records)

    # Compute average sentiment per port
    summary = out.groupby("port")["sentiment"].mean().reset_index()
    summary.rename(columns={"sentiment": "avg_sentiment"}, inplace=True)

    # Save sentiment summary
    summary.to_csv(os.path.join(RESULT_DIR, "avg_sentiment_by_port.csv"), index=False)

    # Count assignments per port
    port_counts = out.groupby("port").size().reset_index(name="count")

    # Compute proportions
    total = port_counts["count"].sum()
    port_counts["proportion"] = port_counts["count"] / total

    # Save proportions
    port_counts.to_csv(os.path.join(RESULT_DIR, "port_proportions.csv"), index=False)

    # Save method-level summary stats
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

    # Save detailed outputs
    out.to_csv(os.path.join(RESULT_DIR, "detailed_output.csv"), index=False)


if __name__ == "__main__":
    run()

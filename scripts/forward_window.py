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
RESULT_DIR = os.path.join(BASE_DIR, "results", "forward_window")

# Ensure results folder exists
os.makedirs(RESULT_DIR, exist_ok=True)

# Number of sentences to include after port mention
WINDOW = 2

# Minimum sentiment threshold for filtering noise
MIN_SENTIMENT = 0.05


def run():
    """
    Execute forward window propagation approach.

    Logic:
    - Port opens forward window
    - Window includes current + next N sentences
    - New port overrides previous window
    """

    # Load dataset into DataFrame
    df = load_jsonl(DATA_PATH)

    records = []

    # Process each review independently
    for _, row in df.iterrows():
        review_id = row["review_id"]

        # Split review into sentences
        sentences = split_sentences(row["review"])

        # Track active window (end index, ports)
        active_window = None

        for i, sent in enumerate(sentences):

            # Split sentence into clauses
            clauses = split_clauses(sent)

            for clause in clauses:
                text = clause["text"]
                ports = clause["ports"]

                sentiment = get_sentiment(text)

                # Filter weak sentiment signals
                if abs(sentiment) < MIN_SENTIMENT:
                    continue

                # If ports are detected, start new window
                if ports:
                    active_window = (i + WINDOW, ports)

                    for p in ports:
                        records.append(
                            {
                                "review_id": review_id,
                                "port": p,
                                "sentence": text,
                                "sentiment": sentiment,
                            }
                        )
                    continue

                # Skip if no active window exists
                if active_window is None:
                    continue

                window_end, window_ports = active_window

                # Close window if exceeded
                if i > window_end:
                    active_window = None
                    continue

                # Assign to active ports
                for p in window_ports:
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

    # Compute sentiment by port
    summary = out.groupby("port")["sentiment"].mean().reset_index()
    summary.rename(columns={"sentiment": "avg_sentiment"}, inplace=True)

    summary.to_csv(os.path.join(RESULT_DIR, "avg_sentiment_by_port.csv"), index=False)

    # Compute port proportions
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

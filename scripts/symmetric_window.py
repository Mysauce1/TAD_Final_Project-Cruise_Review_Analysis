import os
import pandas as pd
from scripts.utils import load_jsonl, split_sentences, detect_ports, get_sentiment

# Resolve base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input data path (JSONL file)
DATA_PATH = os.path.join(BASE_DIR, "data", "cruise_reviews.jsonl")

# Output directory for this approach
RESULT_DIR = os.path.join(BASE_DIR, "results", "symmetric_window")

# Ensure results folder exists
os.makedirs(RESULT_DIR, exist_ok=True)

# Window size: sentences before and after port mention
BEFORE = 2
AFTER = 2


def run():
    """
    Execute the symmetric window propagation approach.

    Logic:
    - Identify all sentence indices where ports are mentioned
    - For each sentence, check if it falls within ±window of any port mention
    - Assign sentiment to those nearby ports
    - A sentence can be linked to multiple ports (overlapping windows)

    Outputs:
        avg_sentiment_by_port.csv: mean sentiment per port
        detailed_output.csv: sentence-level assignments
    """
    # Load dataset
    df = load_jsonl(DATA_PATH)

    records = []

    # Process each review independently
    for _, row in df.iterrows():
        review_id = row["review_id"]

        # Split review into sentences
        sentences = split_sentences(row["review"])

        # Map sentence index → ports mentioned at that index
        port_positions = {}

        for i, sent in enumerate(sentences):
            ports = detect_ports(sent)
            if ports:
                for p in ports:
                    # Store ports at their sentence positions
                    port_positions.setdefault(i, []).append(p)

        # Assign sentiment using symmetric window
        for i, sent in enumerate(sentences):
            # Compute sentiment for current sentence
            sentiment = get_sentiment(sent)

            # Check surrounding window for port mentions
            for idx in range(i - BEFORE, i + AFTER + 1):
                # Skip out-of-bounds indices
                if idx < 0 or idx >= len(sentences):
                    continue

                # If a port appears in the window, assign sentiment
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

    # Compute average sentiment per port
    summary = out.groupby("port")["sentiment"].mean().reset_index()
    summary.rename(columns={"sentiment": "avg_sentiment"}, inplace=True)

    # Save aggregated results
    summary.to_csv(os.path.join(RESULT_DIR, "avg_sentiment_by_port.csv"), index=False)

    # Save detailed sentence-level results
    out.to_csv(os.path.join(RESULT_DIR, "detailed_output.csv"), index=False)


# Run script directly
if __name__ == "__main__":
    run()

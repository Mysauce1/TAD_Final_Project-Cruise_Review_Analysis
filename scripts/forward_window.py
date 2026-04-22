import os
import pandas as pd
from scripts.utils import load_jsonl, split_sentences, detect_ports, get_sentiment

# Obtain base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input data path (JSONL file)
DATA_PATH = os.path.join(BASE_DIR, "data", "cruise_reviews.jsonl")

# Output directory for this approach
RESULT_DIR = os.path.join(BASE_DIR, "results", "forward_window")

# Ensure results folder exists
os.makedirs(RESULT_DIR, exist_ok=True)

# Number of sentences to include after a port mention
WINDOW = 2


def run():
    """
    Execute the forward window propagation approach.

    Logic:
    - When a port appears, open a forward window of size WINDOW
    - Assign sentiment from the port sentence and next WINDOW sentences
    - Multiple windows can overlap if ports appear close together
    - Each sentence can be assigned to multiple ports

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

        # Store active windows as (start_index, ports)
        active_windows = []

        for i, sent in enumerate(sentences):
            # Detect ports in current sentence
            ports = detect_ports(sent)

            if ports:
                # Start a new window at this sentence index
                active_windows.append((i, ports))

            # Compute sentiment for current sentence
            sentiment = get_sentiment(sent)

            # Assign sentence to all active windows within range
            for start_idx, ports_in_window in active_windows:
                if start_idx <= i <= start_idx + WINDOW:
                    for p in ports_in_window:
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

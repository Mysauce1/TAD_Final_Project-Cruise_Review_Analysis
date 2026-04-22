import os
import pandas as pd
from scripts.utils import load_jsonl, split_sentences, detect_ports, get_sentiment

# Obtain base directory (project root)
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
    - Track the current active port(s)
    - Assign each sentence to the most recent port until a new port appears
    - Compute sentiment per sentence using VADER
    - Aggregate average sentiment per port

    Outputs:
        avg_sentiment_by_port.csv: mean sentiment per port
        detailed_output.csv: sentence-level assignments
    """
    # Load dataset into DataFrame
    df = load_jsonl(DATA_PATH)

    records = []

    # Process each review independently
    for _, row in df.iterrows():
        review_id = row["review_id"]

        # Split review into sentences
        sentences = split_sentences(row["review"])

        # Track most recent port(s)
        current_ports = []

        for sent in sentences:
            # Detect ports mentioned in sentence
            ports = detect_ports(sent)

            if ports:
                # Update active port(s)
                current_ports = ports
            elif not current_ports:
                # Skip sentences before any port appears
                continue

            # Compute sentiment for sentence
            sentiment = get_sentiment(sent)

            # Assign sentence to all active ports
            for p in current_ports:
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

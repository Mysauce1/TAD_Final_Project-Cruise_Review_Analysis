import json
import re
import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ftfy import fix_text

# Load NLP model and sentiment analyzer
nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()

# Port dictionary (normalized → canonical form)
PORTS = {"cozumel": "Cozumel", "roatan": "Roatan", "costa maya": "Costa Maya"}


def normalize_text(text):
    """
    Lowercase input text for consistent matching.

    Args:
        text (str): Raw text.

    Returns:
        str: Lowercased text.
    """
    return text.lower()


def detect_ports(text):
    """
    Identify port mentions in text using regex.

    Matches full port names and simple possessives (e.g., "Cozumel's").

    Args:
        text (str): Input text.

    Returns:
        list: Unique list of detected ports (canonical names).
    """
    text = normalize_text(text)
    found = set()

    for key, canonical in PORTS.items():
        # Match full word + optional "'s"
        pattern = r"\b" + re.escape(key) + r"(?:'s)?\b"
        if re.search(pattern, text):
            found.add(canonical)

    return list(found)


def load_jsonl(path):
    """
    Load a JSONL file into a DataFrame with robust parsing.

    Uses standard JSON parsing first. If a line fails (e.g., due to
    unescaped quotes), falls back to regex extraction.

    Args:
        path (str): Path to JSONL file.

    Returns:
        pd.DataFrame: DataFrame with review_id, name, review.
    """
    data = []

    # Regex patterns for fallback parsing (handles broken JSON)
    review_pattern = re.compile(r'"review"\s*:\s*"(.*)"\s*,\s*"name"')
    id_pattern = re.compile(r'"review_id"\s*:\s*(\d+)')
    name_pattern = re.compile(r'"name"\s*:\s*"(.*?)"')

    # Read file line-by-line (memory efficient)
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):

            try:
                # Fast path: parse valid JSON line
                obj = json.loads(line)

            except json.JSONDecodeError:
                # Fallback: extract fields manually if JSON is broken
                try:
                    review_id = int(id_pattern.search(line).group(1))
                    name = name_pattern.search(line).group(1)

                    # Greedy match captures full review (incl. inner quotes)
                    review_match = review_pattern.search(line)
                    review = review_match.group(1) if review_match else ""

                    obj = {"review_id": review_id, "name": name, "review": review}

                except Exception:
                    # Skip line if extraction fails
                    print(f"Skipping bad line {i+1}")
                    continue

            # Clean and fix encoding issues in review text
            if "review" in obj:
                text = obj["review"].replace("\n", " ").replace("\r", " ")

                # Fix encoding issues (quotes, mojibake, etc.)
                text = fix_text(text)

                # Normalize ALL dash variants (including em dash) to hyphen
                text = re.sub(r"[–—−]", "-", text)

                # Normalize smart quotes to standard quotes
                text = re.sub(r"[“”]", '"', text)
                text = re.sub(r"[‘’]", "'", text)

                obj["review"] = text

            data.append(obj)

    return pd.DataFrame(data)


def split_sentences(text):
    """
    Split text into sentences using spaCy.

    Args:
        text (str): Input text.

    Returns:
        list: List of sentence strings.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def split_clauses(sentence):
    """
    Split sentence into clauses based on 'but'.

    If 'but' exists:
    - Split into 2 clauses
    - Second clause may introduce a new port (port switch)

    Args:
        sentence (str): Input sentence.

    Returns:
        list: List of clause dictionaries with text and port info.
    """
    sentence_lower = sentence.lower()

    # Split only on first occurrence of "but"
    if " but " in sentence_lower:
        parts = re.split(r"\bbut\b", sentence, maxsplit=1, flags=re.IGNORECASE)

        first = parts[0].strip()
        second = parts[1].strip()

        first_ports = detect_ports(first)
        second_ports = detect_ports(second)

        # True if second clause introduces a new port context
        has_switch = len(second_ports) > 0

        return [
            {"text": first, "ports": first_ports, "has_port_switch": False},
            {"text": second, "ports": second_ports, "has_port_switch": has_switch},
        ]

    # No split needed
    return [
        {
            "text": sentence.strip(),
            "ports": detect_ports(sentence),
            "has_port_switch": False,
        }
    ]


def get_sentiment(text):
    """
    Compute sentiment using VADER.

    Args:
        text (str): Input text.

    Returns:
        float: compound sentiment score
    """
    return vader.polarity_scores(text)["compound"]

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Resolve project base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Root directory where all method-specific result folders are stored
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Directory where all generated plots will be saved
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Ensure plot output directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)


def format_method(name):
    """
    Convert raw folder-style method names into readable labels.

    Example:
        "until_next_port" to "Until Next Port"
    """
    return name.replace("_", " ").title()


def load_sentiment_data():
    """
    Load and aggregate sentiment results across all methods.

    For each method:
    - Read sentence-level outputs
    - Compute mean sentiment per (method, port)

    Returns:
        pd.DataFrame: aggregated sentiment values
    """

    # List of all experiment method folders
    methods = [
        "until_next_port",
        "forward_window",
        "asymmetric_window",
    ]

    frames = []

    # Iterate through each method folder
    for m in methods:

        # Path to sentence-level output for that method
        path = os.path.join(RESULTS_DIR, m, "detailed_output.csv")

        # Skip if results are missing
        if not os.path.exists(path):
            continue

        # Load sentence-level data
        df = pd.read_csv(path)

        # Attach readable method label for plotting
        df["method"] = format_method(m)

        # Aggregate: compute mean sentiment per port per method
        agg = df.groupby(["method", "port"])["sentiment"].mean().reset_index()

        # Store for final concatenation
        frames.append(agg)

    # Combine all methods into one dataframe
    return pd.concat(frames, ignore_index=True)


def load_proportions():
    """
    Load precomputed port assignment proportions from each method.

    Each method already computes:
        - count per port
        - normalized proportion of assignments

    Returns:
        pd.DataFrame: combined proportions across methods
    """

    methods = [
        "until_next_port",
        "forward_window",
        "asymmetric_window",
    ]

    frames = []

    # Load each method's precomputed proportion file
    for m in methods:

        path = os.path.join(RESULTS_DIR, m, "port_proportions.csv")

        # Skip missing results safely
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        # Add readable method label for visualization
        df["method"] = format_method(m)

        frames.append(df)

    # Merge all methods together
    return pd.concat(frames, ignore_index=True)


def plot_avg_sentiment(df):
    """
    Plot average sentiment by method and port.

    Structure:
        x-axis → methods
        hue → ports
    """

    plt.figure(figsize=(10, 6))

    # Create grouped bar plot
    ax = sns.barplot(data=df, x="method", y="sentiment", hue="port")

    # Add numeric labels on top of each bar for readability
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=9)

    # Plot styling
    plt.title("Average Sentiment by Method and Port")
    plt.ylabel("Average Sentiment")
    plt.xlabel("Assignment Method")

    # Legend shows port categories
    plt.legend(title="Port")

    plt.tight_layout()

    # Save figure to disk
    plt.savefig(os.path.join(PLOTS_DIR, "avg_sentiment_by_method.png"))
    plt.close()


def plot_proportions(df):
    """
    Plot proportion of sentence/clause assignments per port and method.

    Structure:
        x-axis → methods
        hue → ports
    """

    plt.figure(figsize=(10, 6))

    # Create grouped bar plot of proportions
    ax = sns.barplot(data=df, x="method", y="proportion", hue="port")

    # Add value labels to bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=9)

    # Plot styling
    plt.title("Port Assignment Proportions by Method")
    plt.ylabel("Proportion of Assignments")
    plt.xlabel("Assignment Method")

    # Place legend in bottom-right corner for readability
    plt.legend(title="Port", loc="lower right", bbox_to_anchor=(1, 0))

    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(PLOTS_DIR, "port_proportions_by_method.png"))
    plt.close()


def run():
    """
    Main plotting pipeline.

    Steps:
    1. Load sentiment aggregation across methods
    2. Load assignment proportions across methods
    3. Generate both plots
    """

    sentiment_df = load_sentiment_data()
    proportion_df = load_proportions()

    plot_avg_sentiment(sentiment_df)
    plot_proportions(proportion_df)


if __name__ == "__main__":
    run()

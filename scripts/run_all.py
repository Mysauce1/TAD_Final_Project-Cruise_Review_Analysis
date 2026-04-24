from scripts import until_next_port
from scripts import forward_window
from scripts import asymmetric_window
from scripts import plots


def run_all():
    """
    Run all sentiment propagation approaches and generate plots.

    Execution order:
    - until_next_port: baseline propagation until next port mention
    - forward_window: forward-looking window-based propagation
    - asymmetric_window: bidirectional window propagation
    - plots: generate visualizations across all methods

    Each pipeline saves:
    - avg_sentiment_by_port.csv
    - detailed_output.csv
    - port_proportions.csv
    - method_summary.csv
    """

    # Run baseline approach (until next port appears)
    until_next_port.run()

    # Run forward window approach
    forward_window.run()

    # Run asymmetric window approach
    asymmetric_window.run()

    # Generate plots across all methods
    plots.run()


# Execute all pipelines when script is run directly
if __name__ == "__main__":
    run_all()

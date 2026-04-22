from scripts import until_next_port
from scripts import forward_window
from scripts import symmetric_window


def run_all():
    """
    Run all sentiment propagation approaches.

    Executes:
    - until_next_port: propagate until next port mention
    - forward_window: forward window (fixed size)
    - symmetric_window: bidirectional window

    Each script saves its own results to its respective folder.
    """
    # Run unbounded propagation (until next port appears)
    until_next_port.run()

    # Run forward-only window approach
    forward_window.run()

    # Run symmetric (before + after) window approach
    symmetric_window.run()


# Execute all approaches when script is run directly
if __name__ == "__main__":
    run_all()

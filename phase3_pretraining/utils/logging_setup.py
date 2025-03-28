import logging

def setup_logging(log_file: str, level=logging.DEBUG):
    """Set up logging with a file handler and console handler."""
    logging.basicConfig(
        level=level,  # Configurable log level
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
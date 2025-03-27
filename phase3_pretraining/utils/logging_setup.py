import logging

def setup_logging(log_file="pretrain.log"):
    # Configure the root logger
    logger = logging.getLogger()
    
    # Check if the logger already has handlers to avoid duplicates
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Add new handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        # Disable propagation to prevent duplicate logging
        logger.propagate = False

# Call the setup function to configure the logger
setup_logging()
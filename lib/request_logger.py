import logging
import time
from datetime import datetime
import os
import uuid


class RequestLogger:
    def __init__(self, log_dir="request_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def create_request_log(self, request_data=None):
        # Generate unique request ID and timestamp
        request_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create log filename with timestamp
        log_filename = f"request_{timestamp}_{request_id}.log"
        log_path = os.path.join(self.log_dir, log_filename)

        # Create logger for this request
        logger = logging.getLogger(f"request_{request_id}")
        logger.setLevel(logging.DEBUG)

        # Remove any existing handlers
        logger.handlers = []

        # Create file handler
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger, log_path

# # Usage example
# logger_manager = RequestLogger()
#
# # Simulate different requests
# for i in range(3):
#     logger, path = logger_manager.create_request_log()
#     logger.info(f"Request {i + 1} started")
#     logger.debug(f"Processing data for request {i + 1}")
#     logger.info(f"Request {i + 1} completed")
#     print(f"Log saved to: {path}")
#     time.sleep(1)


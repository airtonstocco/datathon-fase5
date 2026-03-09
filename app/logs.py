import logging
import os

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("model_monitor")

logger.setLevel(logging.INFO)

handler = logging.FileHandler("logs/api.log")

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)

handler.setFormatter(formatter)

logger.addHandler(handler)
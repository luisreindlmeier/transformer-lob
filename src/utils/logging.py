import os
import warnings
import logging


def suppress_warnings() -> None:
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)

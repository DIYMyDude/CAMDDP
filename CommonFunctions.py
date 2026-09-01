#========================= CommonFunctions.py =========================
#This file contains all the common functions shared application wide.
#======================================================================

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score

from pathlib import Path
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)

# =========================
# Log Functions
# =========================

def log_create(log_dir):
    
    log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_file = log_dir / f"{timestamp}.log"

    return log_file

def log_message(
    log_file,
    message,
    show_timestamp=True
):

    if show_timestamp:

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        line = f"[{timestamp}] {message}"

    else:

        line = str(message)

    print(line)

    with open(
        log_file,
        "a",
        encoding="utf-8"
    ) as f:
        f.write(line + "\n")


def log_print_message(
    log_file,
    message,
    show_timestamp=True
):
    """
    Print message to console and append to log file.

    Parameters
    ----------
    log_file : Path
        Log file path

    message : str
        Message to print/log

    show_timestamp : bool, default=True
        Prepend timestamp if True
    """

    if show_timestamp:

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        line = f"[{timestamp}] {message}"

    else:

        line = str(message)

    # Console output
    print(line)

    # Log file output
    with open(
        log_file,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(line + "\n")
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix(cm: list[list[int]], title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.array(cm, dtype=int)
    plt.figure()
    plt.imshow(arr)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1, 2], ["SELL", "HOLD", "BUY"])
    plt.yticks([0, 1, 2], ["SELL", "HOLD", "BUY"])

    # write counts in cells
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, str(arr[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
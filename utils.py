import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===============================
# OUTPUT PATH
# ===============================
BASE_DIR = Path(r"D:\Abhiijith\trader-sentiment-analysis")

OUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"

# create folders automatically
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ===============================
# SAVE FIGURE
# ===============================
def save_fig(name: str):

    path = FIG_DIR / name

    # save CURRENT ACTIVE FIGURE
    plt.savefig(path, bbox_inches="tight", dpi=300)

    print(f"Saved figure → {path}")

    plt.close()   # prevents overwrite issues
    return path


# ===============================
# SAVE DATAFRAME REPORT
# ===============================
def save_report(df: pd.DataFrame, name: str):

    path = REPORT_DIR / name

    df.to_csv(path, index=False)

    print(f"Saved report → {path}")

    return path


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ===============================
# QUICK STATS
# ===============================
def quick_stats(df, name="df"):

    print(f"\n--- {name} shape: {df.shape} ---")

    print(df.describe(include='all').transpose())

    print("\nMissing values:\n")
    print(df.isnull().sum())
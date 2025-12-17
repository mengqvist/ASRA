

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from asra.core import asra_ordering



def plot_asra_paper_data(data_path: str, outfile: str):
    # --- 1. Load data -----------------------------------------------------------
    df = pd.read_csv(data_path, sep="\t")

    X = df[["idxat215", "idxat217"]].to_numpy(int)
    y = df["eAverage"].to_numpy(float)
    levels = 20

    # ------------------------------
    # 2. Build original grid
    # ------------------------------
    grid_orig = np.full((levels, levels), np.nan)
    for aa1, aa2, val in zip(df["idxat215"], df["idxat217"], df["eAverage"]):
        grid_orig[aa1, aa2] = val

    # ------------------------------
    # 3. ASRA ordering (paper style)
    # ------------------------------
    ordering, Q = asra_ordering(
        X=X, y=y, levels=levels, sigma=None, w=0.0, mode="paper"
    )

    ord_pos1 = ordering[:, 0]
    ord_pos2 = ordering[:, 1]
    grid_asra = grid_orig[np.ix_(ord_pos1, ord_pos2)]

    # ------------------------------
    # 4. Plot — paper-style
    # ------------------------------

    # jet colormap (red = max)
    cmap = plt.cm.get_cmap("rainbow").copy()
    cmap.set_bad("white")

    masked_orig = np.ma.masked_invalid(grid_orig)
    masked_asra = np.ma.masked_invalid(grid_asra)

    vmin = np.nanmin(grid_orig)
    vmax = np.nanmax(grid_orig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # --- Left panel: original ordering ---
    im0 = axes[0].imshow(
        masked_orig,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper"  # makes row 0 appear at top
    )
    axes[0].set_title("Original index ordering")
    axes[0].set_xlabel("Position 2")
    axes[0].set_ylabel("Position 1")

    # --- Right panel: ASRA ordering ---
    im1 = axes[1].imshow(
        masked_asra,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper"
    )
    axes[1].set_title("ASRA re-ordering")
    axes[1].set_xlabel("Position 2 (ASRA)")
    axes[1].set_ylabel("Position 1 (ASRA)")

    # --- Colorbar: place to the right cleanly ---
    cbar = fig.colorbar(im1, ax=axes, location="right", shrink=0.8)
    cbar.set_label("Activity / fitness")

    fig.savefig(outfile)



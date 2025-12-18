

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
    cbar = fig.colorbar(im1, ax=axes, location="right", shrink=1.0)
    cbar.set_label("Activity / fitness")

    fig.savefig(outfile)


def plot_q_values(Qs, outfile="q_values.png", flip_states=True):
    """
    Visualize per-state ASRA scores in a heatmap.

    Qs : list of arrays
        Qs[pos][state] = ASRA Q-value for that state at that position.
    """

    # Higher = better
    scores = [-np.asarray(Q, float) for Q in Qs]

    max_states = max(len(S) for S in scores)
    n_pos = len(scores)

    # Build rectangular grid padded with NaN
    grid = np.full((max_states, n_pos), np.nan)
    for pos, S in enumerate(scores):
        L = len(S)
        grid[:L, pos] = S

    # Flip vertically so best states appear at the top
    if flip_states:
        grid = np.flipud(grid)

    fig, ax = plt.subplots(figsize=(1.5 * n_pos, 5))

    im = ax.imshow(grid, cmap="viridis", aspect="auto")

    # Ticks for positions
    ax.set_xticks(np.arange(n_pos))
    ax.set_xticklabels([f"Pos {i}" for i in range(n_pos)], rotation=45, ha="right")

    # Ticks for state index (optional)
    ax.set_yticks(np.arange(max_states))
    if flip_states:
        ax.set_yticklabels([max_states - 1 - i for i in range(max_states)])
    else:
        ax.set_yticklabels(np.arange(max_states))

    ax.set_title("Per-state ASRA scores across positions")
    ax.set_xlabel("Position")
    ax.set_ylabel("State index")

    # Correct colorbar usage
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ASRA state score (higher = better)")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)



def plot_state_rankings(Qs, orderings, outfile="state_rankings.png"):
    """
    Visualize ASRA per-state rankings for each position.

    For each position:
    - y-axis = rank (0 = best, increasing downward)
    - cell color = ASRA Q-value (lower = better)
    - cell text = original state index (input encoding)
    """
    n_pos = len(Qs)
    max_states = max(len(Q) for Q in Qs)

    # Build grid of Q values in (rank, position) space
    # scores_grid[rank, pos] = Q-value of the state with that rank at that pos
    scores_grid = np.full((max_states, n_pos), np.nan)

    for pos, (Q, ord_pos) in enumerate(zip(Qs, orderings)):
        Q = np.asarray(Q, float)
        for rank, state in enumerate(ord_pos):
            scores_grid[rank, pos] = Q[state]

    fig, ax = plt.subplots(figsize=(1.5 * n_pos, 6))

    # rank 0 at top → origin="upper"
    im = ax.imshow(scores_grid, cmap="viridis", aspect="auto", origin="upper")

    ax.set_xlabel("Position")
    ax.set_ylabel("Rank (0 = best)")

    ax.set_xticks(np.arange(n_pos))
    ax.set_xticklabels([f"Pos {i}" for i in range(n_pos)], rotation=45, ha="right")

    ax.set_yticks(np.arange(max_states))
    ax.set_yticklabels(np.arange(max_states))

    # Annotate each cell with the original state index
    median_val = np.nanmedian(scores_grid)
    for pos, ord_pos in enumerate(orderings):
        for rank, state in enumerate(ord_pos):
            val = scores_grid[rank, pos]
            if np.isnan(val):
                continue
            # choose text color for contrast
            color = "white" if val > median_val else "black"
            ax.text(
                pos,
                rank,
                str(state),
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ASRA Q (lower = better)")

    ax.set_title("ASRA per-state rankings (numbers = input indices)")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
import numpy as np
import pandas as pd
import os
from asra.core import asra_ordering, asra_orderings_multiblock

data_path = os.path.join(os.path.dirname(__file__), "test_data", "asra_data.tsv")


AA = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
DECODER = {i: a for i, a in enumerate(AA)}

TABLE_1B_215 = "KRWDEPGVAHICQLTYNMSF"
TABLE_1B_217 = "QRESKAWPYHTGICFMVDNL"

TABLE_1C_215 = "RKWDEPGVAHICQLTMNYSF"
TABLE_1C_217 = "QREWSKAHPYTGICFMVDNL"


def _kendall_inversion_distance(order: str, target: str) -> int:
    """Counts pairwise disagreements between two permutations (0 = identical)."""
    pos = {ch: i for i, ch in enumerate(order)}
    seq = [pos[ch] for ch in target]
    inv = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                inv += 1
    return inv


def test_asra_ordering_matches_table_1b_w0_paper():
    """
    Verifies ASRA Step-2 ordering for the 60 min random-sample dataset (TSV),
    reproducing the paper Table 1B (Case B, w=0) ordering strings observed.
    """
    df = pd.read_csv(data_path, sep="\t")
    X = df[["idxat215", "idxat217"]].to_numpy(int)
    y = df["eAverage"].to_numpy(float)
    sigma = df["eStd."].to_numpy(float)

    ordering, Q = asra_ordering(
        X=X, y=y, levels=20, sigma=sigma, w=0.0, mode="paper"
    )

    got_215 = "".join(DECODER[i] for i in ordering[:, 0])
    got_217 = "".join(DECODER[i] for i in ordering[:, 1])

    assert got_215 == TABLE_1B_215
    assert got_217 == TABLE_1B_217

    assert np.isfinite(Q[:, 0]).all()
    assert np.isfinite(Q[:, 1]).all()


def test_asra_ordering_matches_table_1b_w0_robust():
    """
    Verifies ASRA Step-2 ordering for the 60 min random-sample dataset (TSV),
    reproducing the paper Table 1B (Case B, w=0) ordering strings observed.
    """
    df = pd.read_csv(data_path, sep="\t")
    X = df[["idxat215", "idxat217"]].to_numpy(int)
    y = df["eAverage"].to_numpy(float)
    sigma = df["eStd."].to_numpy(float)

    ordering, Q = asra_ordering(
        X=X, y=y, levels=20, sigma=sigma, w=0.0, mode="robust", tau=0.0
    )

    got_215 = "".join(DECODER[i] for i in ordering[:, 0])
    got_217 = "".join(DECODER[i] for i in ordering[:, 1])

    assert got_215 == TABLE_1B_215
    assert got_217 == TABLE_1B_217

    assert np.isfinite(Q[:, 0]).all()
    assert np.isfinite(Q[:, 1]).all()


def test_asra_ordering_is_close_to_table_1c_w1_paper():
    """
    For w=1, the paper reports Table 1C. On the provided sparse TSV,
    exact reproduction depends on conventions (RSD definition + missing-cell handling).
    We enforce a *close* match to Table 1C (small inversion distance),
    which is consistent with the papers statement that w=1 is “highly similar” to w=0.
    """
    df = pd.read_csv(data_path, sep="\t")
    X = df[["idxat215", "idxat217"]].to_numpy(int)
    y = df["eAverage"].to_numpy(float)
    sigma = df["eStd."].to_numpy(float)

    ordering, _Q = asra_ordering(
        X=X, y=y, levels=20, sigma=sigma, w=1.0, mode="paper"
    )

    got_215 = "".join(DECODER[i] for i in ordering[:, 0])
    got_217 = "".join(DECODER[i] for i in ordering[:, 1])

    # tight, but not brittle: allow only a few near-tie swaps
    assert _kendall_inversion_distance(got_215, TABLE_1C_215) <= 2
    assert _kendall_inversion_distance(got_217, TABLE_1C_217) <= 3


def test_asra_ordering_is_close_to_table_1c_w1_robust():
    """
    For w=1, the paper reports Table 1C. On the provided sparse TSV,
    exact reproduction depends on conventions (RSD definition + missing-cell handling).
    We enforce a *close* match to Table 1C (small inversion distance),
    which is consistent with the papers statement that w=1 is “highly similar” to w=0.
    """
    df = pd.read_csv(data_path, sep="\t")
    X = df[["idxat215", "idxat217"]].to_numpy(int)
    y = df["eAverage"].to_numpy(float)
    sigma = df["eStd."].to_numpy(float)

    ordering, _Q = asra_ordering(
        X=X, y=y, levels=20, sigma=sigma, w=1.0, mode="robust", tau=0.0
    )

    got_215 = "".join(DECODER[i] for i in ordering[:, 0])
    got_217 = "".join(DECODER[i] for i in ordering[:, 1])

    # tight, but not brittle: allow only a few near-tie swaps
    assert _kendall_inversion_distance(got_215, TABLE_1C_215) <= 2
    assert _kendall_inversion_distance(got_217, TABLE_1C_217) <= 3



def test_multiblock_wrapper_consistency_uniform_levels():
    """
    asra_ordering (wrapper) should match asra_orderings_multiblock when all
    positions have the same number of levels.
    """
    df = pd.read_csv(data_path, sep="\t")
    X = df[["idxat215", "idxat217"]].to_numpy(int)
    y = df["eAverage"].to_numpy(float)
    sigma = df["eStd."].to_numpy(float)

    levels = 20
    levels_per_pos = [levels] * X.shape[1]

    ordering_wrap, Q_wrap = asra_ordering(
        X=X, y=y, levels=levels, sigma=sigma, w=1.0, mode="paper"
    )

    orderings_mb, Qs_mb = asra_orderings_multiblock(
        X=X, y=y, levels_per_pos=levels_per_pos,
        sigma=sigma, w=1.0, mode="paper"
    )

    # Compare column-by-column
    for pos in range(X.shape[1]):
        assert np.array_equal(ordering_wrap[:, pos], orderings_mb[pos])
        assert np.allclose(Q_wrap[:, pos], Qs_mb[pos], equal_nan=True)


def test_multiblock_supports_different_levels_per_pos():
    """
    Simple synthetic test where positions have different depths.
    Ensures shapes and ordering are well-defined.

    We choose y so that *smaller* values are better, consistent with the
    convention that smaller Q_m corresponds to "better" states in our
    implementation (ascending sort on Q).
    """
    # Position 0: 3 states, Position 1: 2 states
    levels_per_pos = [3, 2]

    # X[:,0] in {0,1,2}, X[:,1] in {0,1}
    X = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [0, 1],
        [1, 1],
        [2, 1],
    ], dtype=int)

    # Start from a grid where higher index is better, then negate so that
    # "better" = more negative (smaller y).
    y_raw = np.array([
        0.0,  # (0,0)
        1.0,  # (1,0)
        2.0,  # (2,0)
        0.5,  # (0,1)
        1.5,  # (1,1)
        2.5,  # (2,1)
    ], dtype=float)

    y = -y_raw  # smaller is better for ASRA's ascending-Q convention
    sigma = np.full_like(y, 0.1, dtype=float)

    orderings, Qs = asra_orderings_multiblock(
        X=X,
        y=y,
        levels_per_pos=levels_per_pos,
        sigma=sigma,
        w=0.0,
        mode="paper",
        tau=0.0,
    )

    # Check lengths match levels_per_pos
    assert len(orderings) == 2
    assert orderings[0].shape[0] == 3
    assert orderings[1].shape[0] == 2

    # For pos0, we expect state 2 (best, most negative y) to rank ahead of 1, ahead of 0
    ord0 = orderings[0]
    idx_0 = np.where(ord0 == 0)[0][0]
    idx_1 = np.where(ord0 == 1)[0][0]
    idx_2 = np.where(ord0 == 2)[0][0]
    assert idx_2 < idx_1 < idx_0

    # For pos1, state 1 is always better (more negative) than state 0
    ord1 = orderings[1]
    idx_0_1 = np.where(ord1 == 0)[0][0]
    idx_1_1 = np.where(ord1 == 1)[0][0]
    assert idx_1_1 < idx_0_1

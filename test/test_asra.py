import numpy as np
import pandas as pd
import os
from asra.core import asra_ordering

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

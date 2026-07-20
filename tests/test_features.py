# Testing the binning/labeling logic against small hand-built spike
# arrays rather than a full simulated session -- faster, and I can check
# the numbers by hand instead of just trusting the output.

import pandas as pd

from features import bin_spikes
from simulate import behavior_at


def small_bouts():
    return pd.DataFrame([
        {"start_s": 0.0, "end_s": 2.0, "behavior": "rest"},
        {"start_s": 2.0, "end_s": 4.0, "behavior": "move"},
    ])


def test_behavior_at_picks_correct_bout():
    bouts = small_bouts()
    assert behavior_at(bouts, 1.0) == "rest"
    assert behavior_at(bouts, 3.0) == "move"


def test_behavior_at_falls_back_at_the_edge():
    # this is the off-by-one case that bit me once -- a bin midpoint
    # landing right at the very end of the session
    bouts = small_bouts()
    assert behavior_at(bouts, 4.0) == "move"


def test_bin_spikes_counts_correctly():
    bouts = small_bouts()
    spikes = {0: [0.1, 0.5], 1: [2.2]}  # neuron 0 fires twice early, neuron 1 once later
    df = bin_spikes(spikes, bouts, bin_size_s=1.0)

    assert len(df) == 4  # 0-4s in 1s bins
    assert df.loc[0, "neuron_0_rate"] == 2.0  # 2 spikes / 1s bin = 2 Hz
    assert df.loc[2, "neuron_1_rate"] == 1.0
    assert df.loc[0, "behavior"] == "rest"
    assert df.loc[2, "behavior"] == "move"


def test_bin_spikes_labels_every_bin():
    # a neuron with zero spikes shouldn't break labeling for the others
    bouts = small_bouts()
    df = bin_spikes({0: []}, bouts, bin_size_s=1.0)
    assert df["behavior"].isin(["rest", "move"]).all()

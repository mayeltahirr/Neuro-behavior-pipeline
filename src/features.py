# features.py
#
# Turns raw spikes + the ethogram into rows a model can actually use: one
# row per time bin, one column per neuron's firing rate, plus the
# behavior label for that bin.

import numpy as np
import pandas as pd

from simulate import behavior_at


def bin_spikes(spikes, bouts_df, bin_size_s=1.0):
    duration_s = bouts_df["end_s"].iloc[-1]
    bin_edges = np.arange(0, duration_s + bin_size_s, bin_size_s)
    n_bins = len(bin_edges) - 1
    n_neurons = len(spikes)

    rate_matrix = np.zeros((n_bins, n_neurons))
    for n, times in spikes.items():
        counts, _ = np.histogram(times, bins=bin_edges)
        rate_matrix[:, n] = counts / bin_size_s

    bin_starts = bin_edges[:-1]
    # label each bin by whatever was happening at its midpoint, not its
    # start -- start felt like it'd bias toward whichever behavior tends
    # to begin bouts, midpoint seemed more honest
    labels = [behavior_at(bouts_df, t + bin_size_s / 2) for t in bin_starts]

    df = pd.DataFrame(rate_matrix, columns=[f"neuron_{n}_rate" for n in range(n_neurons)])
    df.insert(0, "bin_start_s", bin_starts)
    df["behavior"] = labels
    return df

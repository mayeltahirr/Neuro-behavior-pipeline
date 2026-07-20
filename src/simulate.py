# simulate.py
#
# I don't have real neural recordings to work with, so this file fakes a
# session: a population of neurons spiking, plus a rough ethogram (what
# the animal was doing, minute to minute). I made sure firing rates
# actually shift with behavior -- otherwise the model down the line would
# just be learning noise, and that felt like cheating the whole point of
# the demo.

import numpy as np
import pandas as pd

BEHAVIORS = ["rest", "groom", "move"]


def generate_behavior_bouts(duration_s=600, seed=0, min_bout_s=5, max_bout_s=30):
    # just alternates between behaviors for a random stretch of time until
    # we've filled up the whole session. nothing clever, just needs to be
    # a believable-looking ethogram
    rng = np.random.default_rng(seed)
    bouts = []
    t = 0.0
    while t < duration_s:
        behavior = rng.choice(BEHAVIORS)
        bout_len = rng.uniform(min_bout_s, max_bout_s)
        end = min(t + bout_len, duration_s)
        bouts.append({"start_s": t, "end_s": end, "behavior": behavior})
        t = end
    return pd.DataFrame(bouts)


def behavior_at(bouts_df, t):
    """What was the animal doing at time t? Falls back to the last bout
    if t lands right at (or just past) the edge of the session -- ran
    into an off-by-one here early on where the very last bin had no
    match at all."""
    row = bouts_df[(bouts_df.start_s <= t) & (t < bouts_df.end_s)]
    if row.empty:
        return bouts_df.iloc[-1]["behavior"]
    return row.iloc[0]["behavior"]


def _neuron_rate_profile(n_neurons, seed):
    # baseline firing rate (Hz) per neuron, per behavior. some neurons
    # care a lot about "move", some about "groom", some don't seem to
    # care much either way -- wanted a mix instead of every neuron being
    # perfectly behavior-tuned, since that's not really how a real
    # population looks
    rng = np.random.default_rng(seed + 1)
    profiles = []
    for _ in range(n_neurons):
        base = rng.uniform(1.0, 4.0)
        profiles.append({
            "rest": base * rng.uniform(0.7, 1.0),
            "groom": base * rng.uniform(0.8, 1.6),
            "move": base * rng.uniform(1.0, 2.5),
        })
    return profiles


def generate_spike_trains(bouts_df, n_neurons=20, seed=0):
    """Poisson spikes per neuron, rate depends on whatever behavior bout
    we're in at the time. Returns {neuron_id: array of spike times}."""
    rng = np.random.default_rng(seed + 2)
    profiles = _neuron_rate_profile(n_neurons, seed)

    spikes = {}
    for n in range(n_neurons):
        neuron_spikes = []
        for bout in bouts_df.itertuples():
            rate = profiles[n][bout.behavior]
            bout_dur = bout.end_s - bout.start_s
            n_spikes = rng.poisson(rate * bout_dur)
            times = rng.uniform(bout.start_s, bout.end_s, n_spikes)
            neuron_spikes.append(times)
        # concatenate can choke on an empty list of arrays, hence the guard
        spikes[n] = np.sort(np.concatenate(neuron_spikes)) if neuron_spikes else np.array([])
    return spikes

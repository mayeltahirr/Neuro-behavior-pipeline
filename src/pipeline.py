# pipeline.py
#
# Run this and it does the whole thing:
#   python src/pipeline.py --neurons 20 --duration 600
#
# simulate a session -> bin into firing-rate features -> train a decoder
# -> store results -> print a summary. Each step lives in its own file
# and gets tested on its own, so this file just wires them together.

import argparse

from features import bin_spikes
from model import train_behavior_classifier
from repository import init_db, store_bouts, store_model_results, store_session
from simulate import generate_behavior_bouts, generate_spike_trains


def run(n_neurons=20, duration_s=600, bin_size_s=1.0, seed=0, db_path="research_repository.sqlite"):
    print(f"Simulating a {duration_s:.0f}s session with {n_neurons} neurons...")
    bouts = generate_behavior_bouts(duration_s=duration_s, seed=seed)
    spikes = generate_spike_trains(bouts, n_neurons=n_neurons, seed=seed)

    print("Extracting firing-rate features per time bin...")
    features = bin_spikes(spikes, bouts, bin_size_s=bin_size_s)

    print("Training a classifier to decode behavior from firing rates...")
    clf, acc, report = train_behavior_classifier(features, seed=seed)
    print(f"Held-out accuracy: {acc:.2%}  (chance level for 3 behaviors is ~33%)")

    print(f"Storing session, ethogram, and results in {db_path}...")
    conn = init_db(db_path)
    session_id = store_session(conn, duration_s, n_neurons, bin_size_s)
    store_bouts(conn, session_id, bouts)
    store_model_results(conn, session_id, acc, report)
    conn.close()

    print(f"Done. session_id={session_id}")
    return acc


def main():
    ap = argparse.ArgumentParser(description="Simulated neural + behavior decoding pipeline.")
    ap.add_argument("--neurons", type=int, default=20)
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--bin-size", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--db", default="research_repository.sqlite")
    args = ap.parse_args()
    run(args.neurons, args.duration, args.bin_size, args.seed, args.db)


if __name__ == "__main__":
    main()

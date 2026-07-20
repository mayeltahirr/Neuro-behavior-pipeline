# repository.py
#
# Storing results in an actual database instead of leaving CSVs scattered
# around felt like the right call, even for a toy project -- one schema,
# one file, queryable later. Real lab infrastructure would be bigger than
# this (something like DataJoint, probably), but the idea's the same.

import json
import sqlite3


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        duration_s REAL,
        n_neurons INTEGER,
        bin_size_s REAL
    );

    CREATE TABLE IF NOT EXISTS behavior_bouts (
        bout_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        start_s REAL,
        end_s REAL,
        behavior TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    );

    CREATE TABLE IF NOT EXISTS model_results (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        accuracy REAL,
        report_json TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    );
    """)
    conn.commit()
    return conn


def store_session(conn, duration_s, n_neurons, bin_size_s):
    cur = conn.execute(
        "INSERT INTO sessions (duration_s, n_neurons, bin_size_s) VALUES (?, ?, ?)",
        (duration_s, n_neurons, bin_size_s),
    )
    conn.commit()
    return cur.lastrowid


def store_bouts(conn, session_id, bouts_df):
    rows = [(session_id, r.start_s, r.end_s, r.behavior) for r in bouts_df.itertuples()]
    conn.executemany(
        "INSERT INTO behavior_bouts (session_id, start_s, end_s, behavior) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def store_model_results(conn, session_id, accuracy, report):
    # dumping the sklearn report as JSON in a text column rather than
    # normalizing it into its own table -- didn't feel worth designing a
    # whole schema for per-class precision/recall on a project this size
    conn.execute(
        "INSERT INTO model_results (session_id, accuracy, report_json) VALUES (?, ?, ?)",
        (session_id, accuracy, json.dumps(report)),
    )
    conn.commit()

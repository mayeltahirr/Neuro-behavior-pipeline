# Neural Activity + Behavior Decoding Pipeline

This project is a small computer program that pretends to be a brain recording experiment. It makes fake data that looks like real brain data, figures out patterns in it, and saves the results in a database.

## Why the data is fake

I don't have real brain recordings from a real animal. Nobody just hands those out. So instead, I wrote code that makes up realistic-looking fake data: pretend brain cells that fire more or less depending on what the animal is "doing" (resting, grooming, or moving around).

I made sure the fake firing rates actually connect to the fake behavior, the same way real brain cells connect to real behavior. That way, when the computer tries to guess the behavior from the brain activity, there's a real pattern to find, not just random noise.

I built this to show the kind of work a computational neuroscience role would involve: taking messy data, organizing it, and using a model to find patterns in it. I'm not claiming I've worked with real neuroscience data before. I'm showing that I can build the pipeline that this kind of work depends on.

## How it works, step by step

```
fake brain + behavior data --> turn into numbers --> guess the behavior --> save results
      (simulate.py)              (features.py)         (model.py)         (repository.py)
```

- `src/simulate.py` makes the fake data: brain cell activity and a behavior timeline (rest, groom, or move).
- `src/features.py` counts how often each brain cell fired in every short time window, and labels each window with the behavior happening then.
- `src/model.py` trains a simple model to guess the behavior just from the brain cell activity, and checks how often it guesses right.
- `src/repository.py` saves everything into a small database, so the results don't just disappear after the program finishes.
- `src/pipeline.py` runs all four steps in order.
- `data/` has two small example files so you can see what the fake data actually looks like, without running any code.

## Real output from actually running this

This isn't just a description of what it should do. I ran it myself and checked the results directly in the database, not just trusted what printed on screen.

Running:
```
python src/pipeline.py --neurons 20 --duration 600
```

printed:
```
Simulating a 600s session with 20 neurons...
Extracting firing-rate features per time bin...
Training a classifier to decode behavior from firing rates...
Held-out accuracy: 91.33%  (chance level for 3 behaviors is ~33%)
Storing session, ethogram, and results in research_repository.sqlite...
Done. session_id=1
```

91.33% means the model correctly guessed the behavior from brain activity alone, way more often than random guessing (33%) would. That's expected, since I built the fake data so behavior really does connect to firing rate -- the point is proving the pipeline can find that connection, not claiming a real scientific discovery.

Then I checked the database directly, instead of just trusting the printed text:

```
sqlite3 research_repository.sqlite "SELECT * FROM sessions;"
1|600.0|20|1.0

sqlite3 research_repository.sqlite "SELECT behavior, COUNT(*) FROM behavior_bouts GROUP BY behavior;"
groom|24
move|22
rest|22

sqlite3 research_repository.sqlite "SELECT session_id, accuracy FROM model_results;"
1|0.9133333333333333
```

That confirms the database really did store a real session, a real mix of behavior bouts and the real accuracy score -- not just something printed to the screen and thrown away.

## Running it yourself

```bash
pip install -r requirements.txt
python src/pipeline.py --neurons 20 --duration 600
pytest tests/ -v
```

## Troubleshooting

A few things worth knowing if you run this yourself:

- `pip: command not found` -- use `pip3` or `python3 -m pip install -r requirements.txt` instead.
- `ValueError: This solver needs samples of at least 2 classes` -- happens if `--duration` is too short to include more than one behavior. Keep it at 300+ seconds.
- `database is locked` -- close any other program that has `research_repository.sqlite` open, then try again.
- Accuracy looks different if you change the code -- that's expected, not a bug. The accuracy depends directly on how strongly the fake data ties firing rate to behavior.

## What I'd add next

- Right now each time window only gets one behavior label. Real behavior data often overlaps, which would need a different approach.
- The model here is simple (logistic regression). A more advanced version could look at the exact timing of brain activity instead of just counting spikes per window.
- Nothing here touches gene expression data. This would need their own fake data and their own database tables.

## Background

This project was my final project for my Intro to Bioinformatics class. I actually used this bioinformatics project as an outline for the precision oncology pipeline. That one was for a real database systems class. Both projects use the same basic idea. First, you take raw data. Then you organize it. Next, you run a model on it. Last, you save the results in a database so people can look them up later. It helped me plan out the steps before I built the other one. I used Gemini to help build the simulation and to make fake data. This fake data was needed so the project could show real looking outcomes.

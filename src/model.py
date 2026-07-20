# model.py
#
# Question: can we tell what the animal was doing just from population
# firing rates in that time bin? Kept this deliberately simple -- plain
# logistic regression, no hyperparameter tuning -- since the point was to
# get a clean pipeline working end to end, not chase accuracy.

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_behavior_classifier(features_df, test_size=0.25, seed=0):
    feature_cols = [c for c in features_df.columns if c.endswith("_rate")]
    X = features_df[feature_cols]
    y = features_df["behavior"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # zero_division=0 just so a class with no predicted samples doesn't
    # throw a warning wall -- can happen with short test sessions
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return clf, acc, report

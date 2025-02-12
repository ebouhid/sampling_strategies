from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
import argparse
import os


def get_train_df(segs_dir, train_df):
    X_train = []
    y_train = []

    for index, row in train_df.iterrows():
        SA = row["Study_Area"]
        region = f"x{SA:02d}"
        segment_id = row["Segment_id"]
        segment_class = 'forest' if row["Majority_response"] == "Forest" else 'nonforest'
        segment = np.load(os.path.join(segs_dir, f"{region}_{segment_id}.npy"))
        X_train.append(segment.flatten())

        if segment_class == 'forest':
            y_train.append(0)
        else:
            y_train.append(1)

    if len(X_train) != len(train_df):
        print("Error: X_train and train_df should have the same length")

    return X_train, y_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segs_csv", "-s", type=str, required=True)
    parser.add_argument("--segs_dir", "-a", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)

    args = parser.parse_args()

    segs_df = pd.read_csv(args.segs_csv)

    forest_df = segs_df[segs_df["Majority_response"] == "Forest"].copy()
    non_forest_df = segs_df[segs_df["Majority_response"] == "Non-forest"].copy()

    forest_df.sort_values(["Response_time"], ascending=False, inplace=True)
    non_forest_df.sort_values(["Response_time"], ascending=False, inplace=True)

    # Get top and bottom half_pct from forest_df
    forest_train = forest_df

    # Get top and bottom half_pct from non_forest_df
    non_forest_train = non_forest_df

    train_df = pd.concat([forest_train, non_forest_train])

    X_train, y_train = get_train_df(
        os.path.join(args.segs_dir, "train"), train_df)
    print(f'Train size: {len(X_train)}')

    clf = SVC()
    clf.fit(X_train, y_train)

    # Get distances from decision boundary
    distances = clf.decision_function(X_train)

    # Update df
    train_df["Distance"] = distances
    train_df["Distance"] = train_df["Distance"].abs()

    train_df = train_df.sort_values(["Distance"], ascending=False)
    train_df.drop(columns=["Response_time"], inplace=True)
    train_df.to_csv(args.output, index=False)

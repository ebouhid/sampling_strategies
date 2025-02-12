from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
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

def get_test_df(segs_dir):
    X_test = []
    y_test = []

    for file in os.listdir(segs_dir):
        segment = np.load(os.path.join(segs_dir, file))
        X_test.append(segment.flatten())

        if "forest" in file:
            y_test.append(0)
        else:
            y_test.append(1)

    return X_test, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--classifier", "-c", type=str, choices=['svm', 'mlp'], required=True)
    parser.add_argument("--segs_csv", type=str, required=True)
    parser.add_argument("--segs_dir", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    

    args = parser.parse_args()

    segs_df = pd.read_csv(args.segs_csv)

    X_test, y_test = get_test_df(os.path.join(args.segs_dir, "test"))

    res_df = []

    # Define criteria
    criteria = segs_df.columns[-1]

    for classifier_class in [SVC, lambda: MLPClassifier(max_iter=500)]:
        # Random sampling
        for train_pct in [x/100 for x in range(5, 101, 5)]:
            forest_df = segs_df[segs_df["Majority_response"] == "Forest"].copy()
            non_forest_df = segs_df[segs_df["Majority_response"] == "Non-forest"].copy()

            forest_train = forest_df.sample(frac=train_pct, random_state=42)
            non_forest_train = non_forest_df.sample(frac=train_pct, random_state=42)

            train_df = pd.concat([forest_train, non_forest_train])

            X_train, y_train = get_train_df(os.path.join(args.segs_dir, "train"), train_df)

            print(f"Train pct: {train_pct} | Train size: {len(train_df)} | Test size: {len(X_test)}")

            clf = classifier_class()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            bal_acc = balanced_accuracy_score(y_test, y_pred)

            print(f"Balanced accuracy: {bal_acc}")

            res_df.append({
                "Classifier": clf.__class__.__name__,
                "Ordering": "Random",
                "Train_pct": train_pct,
                "Balanced_accuracy": bal_acc,
                "Train_size": len(train_df),
                "Test_size": len(X_test)
            })

        # Decreasing criteria
        for train_pct in [x/100 for x in range(5, 101, 5)]:
            forest_df = segs_df[segs_df["Majority_response"] == "Forest"].copy()
            non_forest_df = segs_df[segs_df["Majority_response"] == "Non-forest"].copy()

            forest_df.sort_values([criteria], ascending=False, inplace=True)
            non_forest_df.sort_values([criteria], ascending=False, inplace=True)
            
            # Get top and bottom half_pct from forest_df
            forest_train = forest_df.iloc[:int(len(forest_df) * train_pct)]

            # Get top and bottom half_pct from non_forest_df
            non_forest_train = non_forest_df.iloc[:int(len(non_forest_df) * train_pct)]

            train_df = pd.concat([forest_train, non_forest_train])

            X_train, y_train = get_train_df(os.path.join(args.segs_dir, "train"), train_df)

            print(f"Train pct: {train_pct} | Train size: {len(train_df)} | Test size: {len(X_test)}")

            clf = classifier_class()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            bal_acc = balanced_accuracy_score(y_test, y_pred)

            print(f"Balanced accuracy: {bal_acc}")

            res_df.append({
                "Classifier": clf.__class__.__name__,
                "Ordering": "Decreasing",
                "Train_pct": train_pct,
                "Balanced_accuracy": bal_acc,
                "Train_size": len(train_df),
                "Test_size": len(X_test)
            })
        
        # Increasing criteria
        for train_pct in [x/100 for x in range(5, 101, 5)]:
            forest_df = segs_df[segs_df["Majority_response"] == "Forest"].copy()
            non_forest_df = segs_df[segs_df["Majority_response"] == "Non-forest"].copy()

            forest_df.sort_values([criteria], ascending=True, inplace=True)
            non_forest_df.sort_values([criteria], ascending=True, inplace=True)
            
            # Get top and bottom half_pct from forest_df
            forest_train = forest_df.iloc[:int(len(forest_df) * train_pct)]

            # Get top and bottom half_pct from non_forest_df
            non_forest_train = non_forest_df.iloc[:int(len(non_forest_df) * train_pct)]

            train_df = pd.concat([forest_train, non_forest_train])
            print(f"Duplicate rows: {len(train_df[train_df.duplicated()])}")


            X_train, y_train = get_train_df(os.path.join(args.segs_dir, "train"), train_df)

            print(f"Train pct: {train_pct} | Train size: {len(train_df)} | Test size: {len(X_test)}")

            clf = classifier_class()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            bal_acc = balanced_accuracy_score(y_test, y_pred)

            print(f"Balanced accuracy: {bal_acc}")

            res_df.append({
                "Classifier": clf.__class__.__name__,
                "Ordering": "Increasing",
                "Train_pct": train_pct,
                "Balanced_accuracy": bal_acc,
                "Train_size": len(train_df),
                "Test_size": len(X_test)
            })


        # Edges
        for train_pct in [x/100 for x in range(5, 101, 5)]:
            forest_df = segs_df[segs_df["Majority_response"] == "Forest"].copy()
            non_forest_df = segs_df[segs_df["Majority_response"] == "Non-forest"].copy()

            forest_df.sort_values([criteria], ascending=True, inplace=True)
            non_forest_df.sort_values([criteria], ascending=True, inplace=True)
            half_pct = int(len(forest_df) * (train_pct / 2))
            
            # Get top and bottom half_pct from forest_df
            forest_train = pd.concat([forest_df.iloc[:half_pct], forest_df.iloc[-half_pct:]])
            # Get top and bottom half_pct from non_forest_df
            non_forest_train = pd.concat([non_forest_df.iloc[:half_pct], non_forest_df.iloc[-half_pct:]])

            train_df = pd.concat([forest_train, non_forest_train])
            # Print duplicates
            train_df = train_df.drop_duplicates(subset=["Segment_id", "Study_Area"])
            print(f"Duplicate rows: {len(train_df[train_df.duplicated()])}")

            X_train, y_train = get_train_df(os.path.join(args.segs_dir, "train"), train_df)

            print(f"Train pct: {train_pct} | Train size: {len(train_df)} | Test size: {len(X_test)}")

            clf = classifier_class()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            bal_acc = balanced_accuracy_score(y_test, y_pred)

            print(f"Balanced accuracy: {bal_acc}")

            res_df.append({
                "Classifier": clf.__class__.__name__,
                "Ordering": "Edges",
                "Train_pct": train_pct,
                "Balanced_accuracy": bal_acc,
                "Train_size": len(train_df),
                "Test_size": len(X_test)
            })



    res_df = pd.DataFrame.from_records(res_df)
    res_df.to_csv(args.output, index=False)


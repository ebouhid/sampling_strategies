from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
import joblib
import pandas as pd
import numpy as np
import argparse
import os
import time


def get_train_data(segs_dir, train_df):
    X_train = []
    y_train = []

    for index, row in train_df.iterrows():
        SA = row["Study_Area"]
        region = f"x{SA:02d}"
        segment_id = row["Segment_id"]
        segment_class = 'forest' if row["Majority_response"] == "Forest" else 'nonforest'

        if segment_class == 'forest':
            segment = np.load(os.path.join(
                segs_dir, f"{region}_{segment_id}.npy"))
            segment = segment.flatten()
            # segment = np.expand_dims(segment, axis=0)
            X_train.append(segment)
            y_train.append(0)
        elif segment_class == 'nonforest':
            segment = np.load(os.path.join(
                segs_dir, f"{region}_{segment_id}.npy"))
            segment = segment.flatten()
            # segment = np.expand_dims(segment, axis=0)
            X_train.append(segment)
            y_train.append(1)

    assert len(X_train) == len(y_train), "X_train and y_train have different lengths ({} vs {})".format(
        len(X_train), len(y_train))
    return X_train, y_train


def get_test_data(segs_dir):
    X_test = []
    y_test = []

    for file in os.listdir(segs_dir):
        if "-forest" in file:
            segment = np.load(os.path.join(segs_dir, file))
            segment = segment.flatten()
            # segment = np.expand_dims(segment, axis=0)
            X_test.append(segment)
            y_test.append(0)
        else:
            segment = np.load(os.path.join(segs_dir, file))
            segment = segment.flatten()
            # segment = np.expand_dims(segment, axis=0)
            X_test.append(segment)
            y_test.append(1)

    return X_test, y_test


def create_checkpoint_dir(base_dir, criteria, ordering):
    path = os.path.join(base_dir, criteria, ordering)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_train_df(segs_df, ordering, n_samples, prev_samples, seed=42):
    forest_df = segs_df[segs_df["Majority_response"] == "Forest"].copy()
    nonforest_df = segs_df[segs_df["Majority_response"] == "Non-forest"].copy()
    criteria = segs_df.columns[-1]

    total_forest = len(forest_df)
    total_nonforest = len(nonforest_df)

    # Calculate the number of samples to take from each class
    remaining_forest = total_forest - prev_samples['forest']
    remaining_nonforest = total_nonforest - prev_samples['nonforest']

    n_samples_forest = min(n_samples, remaining_forest)
    n_samples_nonforest = min(n_samples, remaining_nonforest)

    if ordering == "Random":
        forest_df = forest_df.sample(frac=1, random_state=seed).copy()
        nonforest_df = nonforest_df.sample(frac=1, random_state=seed).copy()

        forest_sample = forest_df.iloc[prev_samples['forest']                                       :prev_samples['forest'] + n_samples_forest]
        nonforest_sample = nonforest_df.iloc[prev_samples['nonforest']                                             :prev_samples['nonforest'] + n_samples_nonforest]

    elif ordering == "Decreasing":
        forest_df.sort_values(criteria, ascending=False, inplace=True)
        nonforest_df.sort_values(criteria, ascending=False, inplace=True)
        forest_sample = forest_df.iloc[prev_samples['forest']                                       :prev_samples['forest'] + n_samples_forest]
        nonforest_sample = nonforest_df.iloc[prev_samples['nonforest']                                             :prev_samples['nonforest'] + n_samples_nonforest]

    elif ordering == "Increasing":
        forest_df.sort_values(criteria, ascending=True, inplace=True)
        nonforest_df.sort_values(criteria, ascending=True, inplace=True)
        forest_sample = forest_df.iloc[prev_samples['forest']                                       :prev_samples['forest'] + n_samples_forest]
        nonforest_sample = nonforest_df.iloc[prev_samples['nonforest']                                             :prev_samples['nonforest'] + n_samples_nonforest]

    elif ordering == "Edges":
        # Sort DataFrames by the criteria.
        forest_df.sort_values(by=criteria, ascending=False, inplace=True)
        nonforest_df.sort_values(by=criteria, ascending=False, inplace=True)

        forest_df = forest_df.reset_index(drop=True)
        nonforest_df = nonforest_df.reset_index(drop=True)

        # Calculate the number of samples for top and bottom edges
        half_samples_forest = n_samples_forest  // 2 if n_samples_forest % 2 == 0 else n_samples_forest  // 2 + 1
        half_samples_nonforest = n_samples_nonforest  // 2 if n_samples_nonforest % 2 == 0 else n_samples_nonforest  // 2 + 1

        # Select samples from the top edge
        forest_top_edge = forest_df.iloc[prev_samples['forest'] // 2:prev_samples['forest'] // 2 + half_samples_forest]
        nonforest_top_edge = nonforest_df.iloc[prev_samples['nonforest'] // 2:prev_samples['nonforest'] // 2 + half_samples_nonforest]

        forest_bottomlim = -prev_samples['forest'] // 2 if prev_samples['forest'] > 0 else None
        nonforest_bottomlim = -prev_samples['nonforest'] // 2 if prev_samples['nonforest'] > 0 else None

        # Select samples from the bottom edge ensuring no overlap with the top edge
        forest_bottom_edge = forest_df.iloc[-(half_samples_forest + prev_samples['forest'] // 2):forest_bottomlim].drop(forest_top_edge.index, errors='ignore')
        nonforest_bottom_edge = nonforest_df.iloc[-(half_samples_nonforest + prev_samples['nonforest'] // 2):nonforest_bottomlim].drop(nonforest_top_edge.index, errors='ignore')

        print(f"prev_samples_forest: {prev_samples['forest']}, prev_samples_nonforest: {prev_samples['nonforest']}")
        print(f"n_samples_forest: {n_samples_forest}, n_samples_nonforest: {n_samples_nonforest}")

        print(f"Top edge: {len(forest_top_edge)} forest samples, {len(nonforest_top_edge)} nonforest samples")
        print(f"Bottom edge: {len(forest_bottom_edge)} forest samples, {len(nonforest_bottom_edge)} nonforest samples")
        print(f"Attempted bottom edge: {-(half_samples_forest + prev_samples['forest'] // 2)}:{forest_bottomlim}, {-(half_samples_nonforest + prev_samples['nonforest'] // 2)}:{nonforest_bottomlim}")
        

        # Combine top and bottom edges into one DataFrame
        forest_sample = pd.concat([forest_top_edge, forest_bottom_edge]).drop_duplicates()
        nonforest_sample = pd.concat([nonforest_top_edge, nonforest_bottom_edge]).drop_duplicates()

        print(f"Combined: {len(forest_sample)} forest samples, {len(nonforest_sample)} nonforest samples")

        # # Ensure the correct number of samples are selected
        # if len(forest_sample) < n_samples_forest:
        #     needed = n_samples_forest - len(forest_sample)
        #     additional_forest_samples = forest_df.drop(forest_sample.index, errors='ignore').head(needed)
        #     forest_sample = pd.concat([forest_sample, additional_forest_samples])

        # if len(nonforest_sample) < n_samples_nonforest:
        #     needed = n_samples_nonforest - len(nonforest_sample)
        #     additional_nonforest_samples = nonforest_df.drop(nonforest_sample.index, errors='ignore').head(needed)
        #     nonforest_sample = pd.concat([nonforest_sample, additional_nonforest_samples])

        # Reset indices to handle duplicates and index continuity
        forest_sample.reset_index(drop=True, inplace=True)
        nonforest_sample.reset_index(drop=True, inplace=True)

        # Combine into final training DataFrame
        train_df = pd.concat([forest_sample, nonforest_sample])

    else:
        raise ValueError(
            "Invalid ordering! Must be one of: 'Decreasing', 'Increasing', 'Edges'")

    train_df = pd.concat([forest_sample, nonforest_sample])

    # Update previous samples count
    prev_samples['forest'] += len(forest_sample)
    prev_samples['nonforest'] += len(nonforest_sample)

    return train_df, prev_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segs_csv", type=str, required=True)
    parser.add_argument("--segs_dir", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--exp_name", '-e', type=str, default="default")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    args = parser.parse_args()

    segs_df = pd.read_csv(args.segs_csv)
    test_gathering_start = time.time()
    X_test, y_test = get_test_data(os.path.join(args.segs_dir, "test"))
    test_gathering_end = time.time()
    print(
        f"Test data gathering took {test_gathering_end - test_gathering_start} seconds")

    res_df = []
    criteria = segs_df.columns[-1]
    base_checkpoint_dir = args.checkpoint_dir

    step = 5  # Step in percentage
    n_samples = np.ceil(
        max(segs_df["Majority_response"].value_counts()) * (step / 100)).astype(np.uint8)

    for order_method in ["Increasing", "Decreasing", "Edges"]:
        prev_samples = {'forest': 0, 'nonforest': 0}
        prev_train_df = pd.DataFrame()
        checkpoint_dir = create_checkpoint_dir(
            base_checkpoint_dir, criteria, order_method)
        for i, train_pct in enumerate([x/100 for x in range(step, 101, step)]):
            if i == 0:
                clf = SVC(kernel="linear", random_state=42)
                print("Initializing SVC")
            else:
                model_path_to_load = os.path.join(
                    checkpoint_dir, f"svc_checkpoint_{prev_pct}.pkl")
                clf = joblib.load(model_path_to_load)
                print(f"Loaded model from {model_path_to_load}")

            train_df, prev_samples = get_train_df(
                segs_df, order_method, n_samples, prev_samples)
            train_df = pd.concat([train_df, prev_train_df]).drop_duplicates()
            
            model_path_to_save = os.path.join(
                checkpoint_dir, f"svc_checkpoint_{train_pct}.pkl")
            print(train_df["Majority_response"].value_counts())

            X_train, y_train = get_train_data(
                os.path.join(args.segs_dir, "train"), train_df)

            fit_start = time.time()
            clf.fit(X_train, y_train)
            fit_end = time.time()
            print(f"Training took {fit_end - fit_start} seconds")
            joblib.dump(clf, model_path_to_save)

            y_pred = clf.predict(X_test)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            print(
                f"Ordering: {order_method}, Train pct: {train_pct}, Balanced accuracy: {bal_acc} | Train size: {len(train_df)} | Test size: {len(X_test)}")

            res_df.append({
                "Classifier": "SVC",
                "Ordering": order_method,
                "Train_pct": train_pct,
                "Balanced_accuracy": bal_acc,
                "Train_size": len(train_df),
                "Test_size": len(X_test),
                "kernel": "linear",
            })

            prev_pct = train_pct
            prev_train_df = train_df.copy()
            print(30 * "-")

    res_df = pd.DataFrame.from_records(res_df)
    res_df.to_csv(args.output.replace(
        ".csv", f"_{args.exp_name}.csv"), index=False)

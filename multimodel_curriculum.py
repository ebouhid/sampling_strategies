from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
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

        segment = np.load(os.path.join(segs_dir, f"{region}_{segment_id}.npy"))
        segment = segment.flatten()
        X_train.append(segment)
        y_train.append(0 if segment_class == 'forest' else 1)

    assert len(X_train) == len(y_train), "X_train and y_train have different lengths ({} vs {})".format(
        len(X_train), len(y_train))
    return X_train, y_train


def get_test_data(segs_dir):
    X_test = []
    y_test = []

    for file in os.listdir(segs_dir):
        segment = np.load(os.path.join(segs_dir, file))
        segment = segment.flatten()
        X_test.append(segment)
        # If filename contains "-forest", label as 0; else label as 1
        y_test.append(0 if "-forest" in file else 1)

    return X_test, y_test


def create_checkpoint_dir(base_dir, model_name, criteria, ordering):
    path = os.path.join(base_dir, model_name, criteria, ordering)
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

    # Casting n_samples_* to be signed integers
    n_samples_forest = np.int8(n_samples_forest)
    n_samples_nonforest = np.int8(n_samples_nonforest)

    if ordering == "Random":
        forest_df = forest_df.sample(frac=1, random_state=seed).copy()
        nonforest_df = nonforest_df.sample(frac=1, random_state=seed).copy()

        forest_sample = forest_df.iloc[prev_samples['forest']:prev_samples['forest'] + n_samples_forest]
        nonforest_sample = nonforest_df.iloc[prev_samples['nonforest']:prev_samples['nonforest'] + n_samples_nonforest]

    elif ordering == "Decreasing":
        forest_df.sort_values(criteria, ascending=False, inplace=True)
        nonforest_df.sort_values(criteria, ascending=False, inplace=True)
        forest_sample = forest_df.iloc[prev_samples['forest']:prev_samples['forest'] + n_samples_forest]
        nonforest_sample = nonforest_df.iloc[prev_samples['nonforest']:prev_samples['nonforest'] + n_samples_nonforest]

    elif ordering == "Increasing":
        forest_df.sort_values(criteria, ascending=True, inplace=True)
        nonforest_df.sort_values(criteria, ascending=True, inplace=True)
        forest_sample = forest_df.iloc[prev_samples['forest']:prev_samples['forest'] + n_samples_forest]
        nonforest_sample = nonforest_df.iloc[prev_samples['nonforest']:prev_samples['nonforest'] + n_samples_nonforest]

    elif ordering == "Edges":
        forest_df.sort_values(by=criteria, ascending=False, inplace=True)
        nonforest_df.sort_values(by=criteria, ascending=False, inplace=True)

        forest_df = forest_df.reset_index(drop=True)
        nonforest_df = nonforest_df.reset_index(drop=True)

        half_samples_forest = n_samples_forest // 2 if n_samples_forest % 2 == 0 else n_samples_forest // 2 + 1
        half_samples_nonforest = n_samples_nonforest // 2 if n_samples_nonforest % 2 == 0 else n_samples_nonforest // 2 + 1

        forest_top_edge = forest_df.iloc[prev_samples['forest'] // 2:prev_samples['forest'] // 2 + half_samples_forest]
        nonforest_top_edge = nonforest_df.iloc[prev_samples['nonforest'] // 2:prev_samples['nonforest'] // 2 + half_samples_nonforest]

        forest_bottomlim = -prev_samples['forest'] // 2 if prev_samples['forest'] > 0 else None
        nonforest_bottomlim = -prev_samples['nonforest'] // 2 if prev_samples['nonforest'] > 0 else None

        forest_bottom_edge = forest_df.iloc[-(half_samples_forest + prev_samples['forest'] // 2):forest_bottomlim].drop(forest_top_edge.index, errors='ignore')
        nonforest_bottom_edge = nonforest_df.iloc[-(half_samples_nonforest + prev_samples['nonforest'] // 2):nonforest_bottomlim].drop(nonforest_top_edge.index, errors='ignore')

        print(f"prev_samples_forest: {prev_samples['forest']}, prev_samples_nonforest: {prev_samples['nonforest']}")
        print(f"n_samples_forest: {n_samples_forest}, n_samples_nonforest: {n_samples_nonforest}")
        print(f"Top edge: {len(forest_top_edge)} forest samples, {len(nonforest_top_edge)} nonforest samples")
        print(f"Bottom edge: {len(forest_bottom_edge)} forest samples, {len(nonforest_bottom_edge)} nonforest samples")
        print(f"Attempted bottom edge: {-(half_samples_forest + prev_samples['forest'] // 2)}:{forest_bottomlim}, {-(half_samples_nonforest + prev_samples['nonforest'] // 2)}:{nonforest_bottomlim}")

        forest_sample = pd.concat([forest_top_edge, forest_bottom_edge]).drop_duplicates()
        nonforest_sample = pd.concat([nonforest_top_edge, nonforest_bottom_edge]).drop_duplicates()

        print(f"Combined: {len(forest_sample)} forest samples, {len(nonforest_sample)} nonforest samples")

        forest_sample.reset_index(drop=True, inplace=True)
        nonforest_sample.reset_index(drop=True, inplace=True)
        train_df = pd.concat([forest_sample, nonforest_sample])
    else:
        raise ValueError("Invalid ordering! Must be one of: 'Decreasing', 'Increasing', 'Edges', 'Random'")

    train_df = pd.concat([forest_sample, nonforest_sample])
    prev_samples['forest'] += len(forest_sample)
    prev_samples['nonforest'] += len(nonforest_sample)

    return train_df, prev_samples


def run_training_iterations(model_name, model_class, model_params, ordering, segs_df,
                            n_samples, segs_dir, X_test, y_test, checkpoint_dir, step, seed=None):
    """
    Execute the inner training loop for a given ordering (and optional seed).
    Returns a list of result dictionaries.
    """
    prev_samples = {'forest': 0, 'nonforest': 0}
    prev_train_df = pd.DataFrame()
    prev_pct = None
    iteration_results = []
    for i, train_pct in enumerate([x/100 for x in range(step, 101, step)]):
        # Initialize or load model
        if i == 0:
            clf = model_class(**model_params)
        else:
            model_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint_{prev_pct}.pkl")
            clf = joblib.load(model_path)

        # Get training data (pass seed if provided)
        train_df, prev_samples = get_train_df(segs_df, ordering, n_samples, prev_samples, seed if seed is not None else 42)
        train_df = pd.concat([train_df, prev_train_df]).drop_duplicates()

        # Prepare data
        X_train, y_train = get_train_data(os.path.join(segs_dir, "train"), train_df)

        # MLP-specific adjustments
        if model_name == "MLP":
            class_counts = np.bincount(y_train)
            if len(class_counts) < 2:
                print(f"Skipping {model_name} - only one class present")
                continue

            min_val_samples = 2
            val_fraction = 0.3
            current_val_samples = int(len(X_train) * val_fraction)
            if current_val_samples < min_val_samples:
                clf.set_params(early_stopping=False, validation_fraction=0.0, max_iter=200)
            else:
                clf.set_params(early_stopping=True, validation_fraction=val_fraction, max_iter=200)

        # Train model
        start_time = time.time()
        try:
            clf.fit(X_train, y_train)
        except ValueError as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
        train_time = time.time() - start_time

        # Save checkpoint
        joblib.dump(clf, os.path.join(checkpoint_dir, f"{model_name}_checkpoint_{train_pct}.pkl"))

        # Evaluate
        y_pred = clf.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        iteration_results.append({
            "Classifier": model_name,
            "Ordering": ordering,
            "Seed": seed,
            "Train_pct": train_pct,
            "Balanced_accuracy": bal_acc,
            "Train_time": train_time,
            "Train_size": len(train_df),
            "Test_size": len(X_test)
        })

        prev_pct = train_pct
        prev_train_df = train_df.copy()

        seed_info = f" (seed {seed})" if seed is not None else ""
        print(f"{model_name} | {ordering}{seed_info} | {train_pct:.0%} | Acc: {bal_acc:.4f} | Time: {train_time:.1f}s")
    return iteration_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segs_csv", type=str, required=True)
    parser.add_argument("--segs_dir", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--exp_name", '-e', type=str, default="default")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    args = parser.parse_args()

    # Define models with their parameters
    models = {
        "Ridge": (RidgeClassifier, {
            'alpha': 0.9,
            'random_state': 42,
            'solver': 'auto'
        }),
        "AdaBoost": (AdaBoostClassifier, {
            'n_estimators': 100,
            'learning_rate': 0.8,
            'random_state': 42
        }),
        "MLP": (MLPClassifier, {
            'hidden_layer_sizes': (200, 150, 100),
            'max_iter': 100,
            'early_stopping': True,
            'random_state': 42
        }),
        "KNN": (KNeighborsClassifier, {
            'n_neighbors': 7,
            'n_jobs': 6
        }),
        "SVM": (SVC, {
            'kernel': 'linear',
            'random_state': 42
        })
    }

    segs_df = pd.read_csv(args.segs_csv)
    X_test, y_test = get_test_data(os.path.join(args.segs_dir, "test"))
    res_df = []
    criteria = segs_df.columns[-1]

    # Calculate sample size increment
    step = 5
    n_samples = np.ceil(max(segs_df["Majority_response"].value_counts()) * (step / 100)).astype(np.uint8)

    # Main training loop
    for model_name, (model_class, model_params) in models.items():
        print(f"\n=== Training {model_name} classifier ===")
        for order_method in ["Random", "Increasing", "Decreasing", "Edges"]:
            if order_method == "Random":
                # For "Random", run multiple seeds and then aggregate the results.
                random_results = []
                seeds = range(10)
                for seed in seeds:
                    chkpt_dir = create_checkpoint_dir(
                        args.checkpoint_dir, model_name, criteria, order_method + f"_seed_{seed}"
                    )
                    seed_results = run_training_iterations(model_name, model_class, model_params,
                                                           order_method, segs_df, n_samples,
                                                           os.path.join(args.segs_dir), X_test, y_test,
                                                           chkpt_dir, step, seed)
                    random_results.extend(seed_results)
                # Aggregate results across seeds by train_pct (discarding std, etc.)
                rand_df = pd.DataFrame(random_results)
                agg_rand = rand_df.groupby("Train_pct").agg({
                    "Balanced_accuracy": "mean",
                    "Train_time": "mean",
                    "Train_size": "mean",
                    "Test_size": "mean"
                }).reset_index()
                for idx, row in agg_rand.iterrows():
                    res_df.append({
                        "Classifier": model_name,
                        "Ordering": order_method,
                        "Train_pct": row["Train_pct"],
                        "Balanced_accuracy": row["Balanced_accuracy"],
                        "Train_time": row["Train_time"],
                        "Train_size": row["Train_size"],
                        "Test_size": row["Test_size"]
                    })
            else:
                chkpt_dir = create_checkpoint_dir(
                    args.checkpoint_dir, model_name, criteria, order_method
                )
                results = run_training_iterations(model_name, model_class, model_params,
                                                  order_method, segs_df, n_samples,
                                                  os.path.join(args.segs_dir), X_test, y_test,
                                                  chkpt_dir, step)
                res_df.extend(results)

    # Save final results
    pd.DataFrame(res_df).to_csv(
        args.output.replace(".csv", f"_{args.exp_name}.csv"),
        index=False
    )

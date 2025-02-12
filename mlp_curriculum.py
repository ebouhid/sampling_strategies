# from sklearn.neural_network import MLPClassifier
from mlp_model import SimpleMLPWrapper
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
            segment = np.expand_dims(segment, axis=0)
            X_train.append(segment)
            y_train.append(0)
        elif segment_class == 'nonforest':
            segment = np.load(os.path.join(
                segs_dir, f"{region}_{segment_id}.npy"))
            segment = segment.flatten()
            segment = np.expand_dims(segment, axis=0)
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
            segment = np.expand_dims(segment, axis=0)
            X_test.append(segment)
            y_test.append(0)
        elif "-nonforest" in file:
            segment = np.load(os.path.join(segs_dir, file))
            segment = segment.flatten()
            segment = np.expand_dims(segment, axis=0)
            X_test.append(segment)
            y_test.append(1)

    return X_test, y_test


def create_checkpoint_dir(base_dir, criteria, ordering):
    path = os.path.join(base_dir, criteria, ordering)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_train_df(segs_df, ordering, n_samples, prev_samples):
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
        forest_df = forest_df.sample(frac=1, random_state=42).copy()
        nonforest_df = nonforest_df.sample(frac=1, random_state=42).copy()

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
        # Ensure the DataFrame is sorted by the criteria.
        forest_df.sort_values(criteria, ascending=False, inplace=True)
        nonforest_df.sort_values(criteria, ascending=False, inplace=True)

        # Calculate the indices for the top and bottom edges.
        half_samples_forest = np.ceil(n_samples_forest / 2).astype(np.uint8)
        half_samples_nonforest = np.ceil(
            n_samples_nonforest / 2).astype(np.uint8)

        # Calculate starting points for the next set of samples.
        half_prev_samples_forest = np.floor(
            prev_samples['forest'] / 2).astype(np.uint8)
        half_prev_samples_nonforest = np.floor(
            prev_samples['nonforest'] / 2).astype(np.uint8)

        # Calculate indices for top samples.
        forest_top_edge = forest_df.iloc[half_prev_samples_forest:
                                         half_prev_samples_forest + half_samples_forest]
        nonforest_top_edge = nonforest_df.iloc[half_prev_samples_nonforest:
                                               half_prev_samples_nonforest + half_samples_nonforest]

        # Calculate indices for bottom samples; handle negative slicing carefully.
        forest_bottom_edge = forest_df.iloc[-(half_prev_samples_forest + half_samples_forest)                                            :-half_prev_samples_forest if half_prev_samples_forest > 0 else None]
        nonforest_bottom_edge = nonforest_df.iloc[-(half_prev_samples_nonforest + half_samples_nonforest)                                                  :-half_prev_samples_nonforest if half_prev_samples_nonforest > 0 else None]

        # Combine top and bottom edges into one DataFrame.
        forest_sample = pd.concat([forest_top_edge, forest_bottom_edge])
        nonforest_sample = pd.concat(
            [nonforest_top_edge, nonforest_bottom_edge])

    else:
        raise ValueError(
            "Invalid ordering! Must be one of 'Random', 'Decreasing', 'Increasing', 'Edges'")

    train_df = pd.concat([forest_sample, nonforest_sample])

    # Update previous samples count
    prev_samples['forest'] += n_samples_forest
    prev_samples['nonforest'] += n_samples_nonforest

    return train_df, prev_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segs_csv", type=str, required=True)
    parser.add_argument("--segs_dir", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--experiment_name", "-e", type=str, required=True)

    args = parser.parse_args()

    segs_df = pd.read_csv(args.segs_csv)
    test_gathering_start = time.time()
    X_test, y_test = get_test_data(os.path.join(args.segs_dir, "test"))
    test_gathering_end = time.time()

    sensor = str.upper((args.output.split("_")[0])).split("/")[-1]

    print(
        f"Test data gathering took {test_gathering_end - test_gathering_start} seconds")

    res_df = []
    criteria = segs_df.columns[-1]
    base_checkpoint_dir = "model_checkpoints"

    step = 5
    n_samples = np.ceil(
        max(segs_df["Majority_response"].value_counts()) * (step / 100)).astype(np.uint8)

    for order_method in ["Random", "Increasing"]:
        prev_samples = {'forest': 0, 'nonforest': 0}
        prev_train_df = pd.DataFrame()
        checkpoint_dir = create_checkpoint_dir(
            base_checkpoint_dir, criteria, order_method)
        for i, train_pct in enumerate([x/100 for x in range(step, 101, step)]):
            if i == 0:
                input_dim = X_test[0].shape[1]
                output_dim = len(np.unique(np.stack(y_test, axis=0)))
                print(f"Input dim: {input_dim}, Output dim: {output_dim}")
                clf = SimpleMLPWrapper(input_dim=input_dim, hidden_dim=128, output_dim=output_dim,
                                       batch_size=256, exp_name=args.experiment_name, learning_rate=1e-3)
                print("Initializing HaralickClassifier")
            else:
                model_path_to_load = os.path.join(
                    checkpoint_dir, f"mlp_checkpoint_{prev_pct}.pkl")
                clf.load_model(model_path_to_load)
                print(f"Loaded model from {model_path_to_load}")

            train_df, prev_samples = get_train_df(
                segs_df, order_method, n_samples, prev_samples)
            train_df = pd.concat([train_df, prev_train_df]).drop_duplicates()
            
            model_path_to_save = os.path.join(
                checkpoint_dir, f"mlp_checkpoint_{train_pct}.pkl")
            print(train_df["Majority_response"].value_counts())
            X_train, y_train = get_train_data(
                os.path.join(args.segs_dir, "train"), train_df)

            clf.model.reset_max_val_acc()
            clf.fit(X_train, y_train, X_test, y_test, num_epochs=50,
                    log_name=f"{sensor}_{criteria}_{order_method}_{train_pct}")
            max_val_acc = clf.model.max_val_acc

            # Save the model
            clf.save_model(model_path_to_save)
            print(f'Saved model to {model_path_to_save}')

            # Evaluate the model
            bal_acc = clf.score(X_test, y_test)
            print(
                f"Ordering: {order_method}, Train pct: {train_pct}, Balanced accuracy: {bal_acc} | Train size: {len(train_df)} | Test size: {len(X_test)}")

            res_df.append({
                "Classifier": "MLPClassifier",
                "Ordering": order_method,
                "Train_pct": train_pct,
                "Balanced_accuracy": bal_acc,
                "Max_val_acc": max_val_acc,
                "Train_size": len(train_df),
                "Test_size": len(X_test)
            })

            prev_pct = train_pct
            prev_train_df = train_df.copy()
            print(30 * "-")

    res_df = pd.DataFrame.from_records(res_df)
    res_df.to_csv(args.output, index=False)

from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
import joblib
import pandas as pd
import numpy as np
import argparse
import os
import time
from svm_curriculum import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segs_csv", type=str, required=True)
    parser.add_argument("--segs_dir", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--exp_name", '-e', type=str, default="default")
    args = parser.parse_args()

    segs_df = pd.read_csv(args.segs_csv)
    test_gathering_start = time.time()
    X_test, y_test = get_test_data(os.path.join(args.segs_dir, "test"))
    test_gathering_end = time.time()
    print(f"Test data gathering took {test_gathering_end - test_gathering_start} seconds")

    res_df = []
    criteria = segs_df.columns[-1]
    base_checkpoint_dir = "model_checkpoints"
    step = 5  # Step in percentage
    max_samples = max(segs_df["Majority_response"].value_counts())
    n_samples = np.ceil(max_samples * (step / 100)).astype(np.uint8)
    
    seeds = range(10)  # Different seeds for random sampling
    results = []

    for seed in seeds:
        for order_method in ["Random"]:
            prev_samples = {'forest': 0, 'nonforest': 0}
            prev_train_df = pd.DataFrame()
            checkpoint_dir = create_checkpoint_dir(base_checkpoint_dir, criteria, order_method + f"_seed_{seed}")
            for i, train_pct in enumerate([x/100 for x in range(step, 101, step)]):
                if i == 0:
                    clf = SVC(kernel="linear", random_state=42)
                    print("Initializing SVC")
                else:
                    model_path_to_load = os.path.join(checkpoint_dir, f"svc_checkpoint_{prev_pct}.pkl")
                    clf = joblib.load(model_path_to_load)
                    print(f"Loaded model from {model_path_to_load}")

                train_df, prev_samples = get_train_df(segs_df, order_method, n_samples, prev_samples, seed)
                train_df = pd.concat([train_df, prev_train_df]).drop_duplicates()

                model_path_to_save = os.path.join(checkpoint_dir, f"svc_checkpoint_{train_pct}.pkl")
                print(train_df["Majority_response"].value_counts())

                X_train, y_train = get_train_data(os.path.join(args.segs_dir, "train"), train_df)
                
                fit_start = time.time()
                clf.fit(X_train, y_train)
                fit_end = time.time()
                print(f"Training took {fit_end - fit_start} seconds")
                joblib.dump(clf, model_path_to_save)

                y_pred = clf.predict(X_test)
                bal_acc = balanced_accuracy_score(y_test, y_pred)
                print(f"Seed: {seed}, Ordering: {order_method}, Train pct: {train_pct}, Balanced accuracy: {bal_acc}")

                results.append({
                    "Seed": seed,
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

    # Aggregate results across seeds
    final_df = pd.DataFrame.from_records(results)
    final_agg_df = final_df.groupby(["Ordering", "Train_pct"]).agg({
        "Balanced_accuracy": ['mean', 'std'],
        "Train_size": 'mean',
        "Test_size": 'mean'
    }).reset_index()
    final_agg_df.columns = ['Ordering', 'Train_pct', 'Bal_Accuracy_Mean', 'Bal_Accuracy_Std', 'Train_Size_Mean', 'Test_Size_Mean']

    final_agg_df.to_csv(args.output.replace(".csv", f"_{args.exp_name}_aggregated.csv"), index=False)

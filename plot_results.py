import pandas as pd
import matplotlib.pyplot as plt

def plot_balanced_accuracy(csv_path, classifier_name=None, title=None):
    """
    Reads a CSV of the form:
      Classifier,Ordering,Train_pct,Balanced_accuracy,Train_time,Train_size,Test_size
    and plots Balanced Accuracy vs. Train_pct for each Ordering.
    
    :param csv_path: Path to the CSV file.
    :param classifier_name: If given, only rows matching this classifier are plotted.
    :param title: Plot title (if None, a default is used).
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Optionally filter on a given classifier
    if classifier_name is not None:
        df = df[df['Classifier'] == classifier_name]

    # If there's nothing left after filtering, just return
    if df.empty:
        print(f"No data for classifier '{classifier_name}' in {csv_path}")
        return

    # Get unique orderings in the data
    orderings = df['Ordering'].unique()

    criteria = os.path.basename(csv_path).split("_")[1]
    sensor = os.path.basename(csv_path).split("_")[0]
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each ordering
    colors = {
        "Increasing": "blue",
        "Decreasing": "red",
        "Edges": "green",
        "Random": "gold"
    }

    for ordering in ["Increasing", "Decreasing", "Edges", "Random"]:
        if ordering in orderings:
            subset = df[df['Ordering'] == ordering].copy()
            subset.sort_values(by='Train_pct', inplace=True)
            ax.plot(
                subset['Train_pct'],
                subset['Balanced_accuracy'],
                marker='o',
                label=ordering,
                color=colors[ordering]
            )
        else:
            print(f"No data for ordering '{ordering}' with classifier '{classifier_name}' in {csv_path}")

    # Label axes
    ax.set_xlabel("Training Percentage", fontsize=18)
    ax.set_ylabel("Balanced Accuracy", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Set a title
    if title is None:
        title = f"Balanced Accuracy vs Training Percentage"
        if classifier_name is not None:
            title += f" for {classifier_name}"
        # if criteria is not None:
        #     title += f" ({criteria})"
        # if sensor is not None:
        #     title += f" | Sensor: {sensor.capitalize()}"
    ax.set_title(title, fontsize=18)

    # Show a legend
    ax.legend(title="Ordering")
    ax.legend(title="Ordering", loc="lower right", fontsize=18, title_fontsize=18)

    # Optionally tweak the y-axis range or other aesthetics
    ax.set_ylim([0.3, 1.0])
    
    plt.tight_layout()
    png_save_name = csv_path.replace(".csv", f"_{classifier_name}.png")
    # Add grid lines for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    # Save the plot
    plt.savefig(png_save_name)

    # Close fig
    plt.close(fig)

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", "-r", help="Root directory containing CSV files")
    args = parser.parse_args()

    result_root = args.result_root

    avail_csvs = os.listdir(result_root)
    avail_csvs = [csv for csv in avail_csvs if csv.endswith(".csv")]
    
    for csv_fname in avail_csvs:
        csv = os.path.join(result_root, csv_fname)
        print(f"Plotting {csv}")
        for classifier_name in ["Ridge", "SVM", "MLP", "KNN", "AdaBoost"]:
            plot_balanced_accuracy(csv, classifier_name=classifier_name)

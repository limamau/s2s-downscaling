import os
import matplotlib.pyplot as plt
import pandas as pd

from utils import create_folder


def main():
    # Experiments
    script_dir = os.path.dirname(os.path.realpath(__file__))
    experiment_names = ["debug", "diff_nolambda"]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10,5))
    plt.xlabel("Training iterations")
    plt.ylabel("Loss")
    cmap = plt.get_cmap("Dark2")
    
    # Training losses
    for color_idx, experiment_name in enumerate(experiment_names):
        print(experiment_name)
        experiment_dir = os.path.join(script_dir, "experiments", experiment_name)
        df = pd.read_csv(os.path.join(experiment_dir, "losses.csv"))
        ax.plot(df['iteration'], df['train_loss'], color=cmap(color_idx), label=experiment_name)
    
    plt.legend()
    figs_dir = os.path.join(script_dir, "figs")
    create_folder(figs_dir)
    fig.savefig(os.path.join(figs_dir, "training_losses.png"))
    

if __name__ == "__main__":
    main()

import os
import matplotlib.pyplot as plt
import pandas as pd

from utils import create_folder


def main():
    # Experiments
    script_dir = os.path.dirname(os.path.realpath(__file__))
    experiment_names = ["diffusers_jj"]
    variables = ["train_loss", "psd_distance", "cdf_distance", "pss"]
    
    # Plotting
    for var in variables:
        fig, ax = plt.subplots(figsize=(10,5))
        plt.xlabel("training iterations")
        plt.ylabel(var.split("_")[0] + " " + var.split("_")[-1])
        cmap = plt.get_cmap("Dark2")
        
        # Validation score
        for color_idx, experiment_name in enumerate(experiment_names):
            experiment_dir = os.path.join(script_dir, "experiments", experiment_name)
            df = pd.read_csv(os.path.join(experiment_dir, "output.csv"))
            ax.plot(df['iteration'], df[var], color=cmap(color_idx), label=experiment_name)
        
        plt.legend()
        figs_dir = os.path.join(script_dir, "figs")
        create_folder(figs_dir)
        fig.savefig(os.path.join(figs_dir, "{}.png".format(var)))
    

if __name__ == "__main__":
    main()

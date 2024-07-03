import os

from training.experiment import Experiment
from training.training import train

 
def main():
    experiment = Experiment(
        experiment_file=os.path.join(os.path.dirname(__file__), "experiment.yml"),
        dataset_file="/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/train_data/cpc_june-july-dry-filter.h5"
    )
    
    train(experiment)
    print("Done!")


if __name__ == "__main__":
    main()
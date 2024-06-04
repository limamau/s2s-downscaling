import os

from training.experiment import Experiment
from training.training import train
            
            
def main():
    experiment = Experiment(os.path.join(os.path.dirname(__file__), "experiment.yml"))
    train(experiment)
    print("Done!")


if __name__ == "__main__":
    main()
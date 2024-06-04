import os
import csv
import logging
import yaml
from jax import devices


class Writer:
    """"
    Class to write training information. This class does a bit more than only logging. It also 
    saves the configuration of the experiment in a .yml file and saves losses on .csv files.
    """
    def __init__(self, path):
        # .log
        self.log_file = os.path.join(path, "output.log")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(message)s')
        self.handler = logging.FileHandler(self.log_file)
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        self.logger.propagate = False
        
        # .csv
        self.csv_file = os.path.join(path, "losses.csv")
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['iteration', 'loss'])
            
        # .yml
        self.yaml_file = os.path.join(path, "experiment.yml")
        
        
    def _remove_identifier_suffix(self, s):
        if s.endswith("_identifier"):
            return s[:-11]  # remove the last 11 characters (_identifier)
        return s
        
        
    def log_and_save_config(self, experiment):
        config_dict = {}
        for key, value in vars(experiment).items():
            if isinstance(value, (float, int, str)) and key != 'experiment_dir':
                key = self._remove_identifier_suffix(key) 
                self.logger.info(f'{key}: {value}')
                config_dict[key] = value

        with open(self.yaml_file, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)
            
            
    def save_normalizations(self, dimensions, data_mean, data_std):
        info = {
            'dimensions': [dimensions[0], dimensions[1], dimensions[2]],
            'data_mean': float(data_mean),
            'data_std': float(data_std)
        }
        with open(self.yaml_file, 'a') as file:
            yaml.dump(info, file, default_flow_style=False)
            
            
    def log_device(self):
        for dev in devices():
            if dev.platform == 'gpu':
                self.logger.info("Running on GPU...")
                return 0
        self.logger.info("Running on CPU...")
            
    
    def log_loss(self, k, loss):
        self.logger.info(f'Iteration {k}, Loss: {loss:.4f}')
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([k, loss])


    def close(self):
        self.logger.info('End of experiment.')
        self.logger.removeHandler(self.handler)
        self.handler.close()
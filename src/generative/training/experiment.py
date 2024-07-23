import os, yaml, h5py
import numpy as np
from generative import networks


def _get_string(value):
    if value is None:
        return ValueError("String must be provided.")
    else:
        return str(value)
    

def _get_list(value):
    if value is None:
        return None
    else:
        return value
    
    
def _get_network(model_identifier, **kwargs):
    net_map = {
        'diffusers': networks.diffusers.Network,
        'homemade': networks.homemade.Network,
        'conditional': networks.conditional.Network,
        'heavy_diffusers': networks.heavy_diffusers.Network,
    }
    if model_identifier not in net_map:
        raise ValueError("Unsupported model: {}".format(model_identifier))
    
    # Filter out None values from kwargs
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    
    return net_map[model_identifier](**kwargs)


def _get_data(file, main_variables, conditional_variables, max_len):
    if file is not None:
        # Get actual length of the dataset (in time i.e. number of samples)
        dummy_var = main_variables[0]
        with h5py.File(file, 'r') as f:
            len_times = len(f[dummy_var][:,0,0])
            spatial_shape = f[dummy_var][0,:,:].shape
    
        # Reduce if necessary
        Nt = min(len_times, max_len)
    
        # Get data
        data = np.empty((Nt, *spatial_shape, len(main_variables)))
        with h5py.File(file, 'r') as f:
            for i, var in enumerate(main_variables):
                data[:,:,:,i] = f[var][:Nt,:,:]
        
            if conditional_variables is not None:
                conditions = np.empty((Nt, len(conditional_variables), 1))
                for i, var in enumerate(conditional_variables):
                    conditions[:,i,0] = f[var][:Nt]
            else:
                conditions = None
        
        return data, conditions, Nt


def _get_float(value, error_if_none=True):
    if value == None:
        if error_if_none:
            raise ValueError("Value cannot be None.")
        return None
    return float(value)


def _get_int(value):
    if value == None:
        return ValueError("Value cannot be None.")
    return int(value)


def _get_boolean(value):
    if value == None:
        return ValueError("Value cannot be None.")
    return bool(value)
    
def _get_dimensions(dimensions):
    if dimensions is None:
        return None
    else:
        return tuple(dimensions)
    

class Experiment:
    """
    Class to hold the experiment configuration.
    """
    def __init__(self, experiment_file, dataset_file=False):
        with open(experiment_file, 'r') as file:
            experiment = yaml.safe_load(file)
        
        self.experiment_name = _get_string(experiment.get('experiment_name'))
        self.max_len = _get_int(experiment.get('max_len'))
        self.batch_size = _get_int(experiment.get('batch_size'))
        self.epochs = _get_int(experiment.get('epochs'))
        self.main_variables = _get_list(experiment.get('main_variables'))
        self.conditional_variables = _get_list(experiment.get('conditional_variables'))
        self.validation_ratio = _get_float(experiment.get('validation_ratio'))
        self.data, self.conditions, self.Nt = _get_data(
            dataset_file,
            self.main_variables,
            self.conditional_variables,
            self.max_len,
        )
        self.training_iterations = self.epochs * self.Nt
        self.network_identifier = _get_string(experiment.get('network'))
        self.dropout = _get_float(experiment.get('dropout'), error_if_none=False)
        self.sigma_min = _get_float(experiment.get('sigma_min'))
        self.sigma_max = _get_float(experiment.get('sigma_max'))
        self.network = _get_network(self.network_identifier, 
            dropout_rate=self.dropout,
        )
        self.learning_rate = _get_float(experiment.get('learning_rate'))
        self.ema = _get_float(experiment.get('ema_decay_rate'), error_if_none=False)
        self.is_log_transforming = _get_boolean(experiment.get('log_transform'))
        self.norm_mean = _get_float(experiment.get('norm_mean'))
        self.norm_std = _get_float(experiment.get('norm_std'))
        self.log_each = _get_int(experiment.get('log_each'))
        self.ckpt_each = _get_int(experiment.get('checkpoint_each'))
        self.experiment_dir = os.path.join(os.path.dirname(experiment_file), "experiments", self.experiment_name)
        
        # in the case where one is creating an experiment the following will be none
        self.dimensions = _get_dimensions(experiment.get('dimensions'))
        self.dataset_mean = _get_float(experiment.get('dataset_mean'), error_if_none=False)
        self.dataset_std = _get_float(experiment.get('dataset_std'), error_if_none=False)

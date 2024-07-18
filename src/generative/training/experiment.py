import os, yaml, h5py
import optax
import numpy as np

from generative.networks import diffusers, homemade, conditional, heavy_diffusers
from .distances import *
from .schedules import *


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
        'diffusers': diffusers.Network,
        'homemade': homemade.Network,
        'conditional': conditional.Network,
        'heavy_diffusers': heavy_diffusers.Network,
    }
    if model_identifier not in net_map:
        raise ValueError("Unsupported model: {}".format(model_identifier))
    
    # Filter out None values from kwargs
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    
    return net_map[model_identifier](**kwargs)


def _get_distance(dist_identifier):
    dist_map = {
        'l1': l1,
        'l2': l2,
        'lpips': lpips,
    }
    if dist_identifier not in dist_map:
        raise ValueError("Unsupported distance: {}".format(dist_identifier))
    return dist_map[dist_identifier]


def _get_loss_weighting(is_loss_weighting):
    if is_loss_weighting:
        return lambda x,y: loss_weight(x,y)
    else:
        return lambda x,y: 1.0


def _get_optimizer(optimizer_identifier):
    optimizer_map = {
        'adam': optax.adam,
        'radam': optax.radam,
    }
    if optimizer_identifier not in optimizer_map:
        raise ValueError("Unsupported optimizer: {}".format(optimizer_identifier))
    return optimizer_map[optimizer_identifier]
        

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
    
    
def _get_N(N, s0, s1, training_iterations):
    if isinstance(N, int):
        return lambda k: N
    if s0 == None:
            raise ValueError("s0 cannot be None for N schedule.")
    if s1 == None:
        raise ValueError("s1 cannot be None for N schedule.")
    if N == 'schedule':
        return N_schedule(s0, s1, training_iterations)
    elif N == 'schedule_improved':
        return N_schedule_improved(s0, s1, training_iterations)
    
    raise ValueError("Unsupported N: {}".format(N))


def _get_mu(mu, s0, mu0, N):
    if isinstance(mu, float):
        return lambda k: mu
    elif mu == 'schedule':
        if s0 == None:
            raise ValueError("s0 cannot be None for EMA schedule.")
        elif mu0 == None:
            raise ValueError("mu0 cannot be None for EMA schedule.")
        else:
            return mu_schedule(s0, mu0, N)
    else:
        raise ValueError("Unsupported EMA decay rate: {}".format(mu))
    
    
def _get_noise_schedule(min_noise, max_noise):
    return noise_schedule(min_noise, max_noise)


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
        self.min_noise = _get_float(experiment.get('min_noise'))
        self.imin = _get_float(experiment.get('imin'))
        self.max_noise = _get_float(experiment.get('max_noise'))
        self.imax = _get_float(experiment.get('imax'))
        self.network = _get_network(self.network_identifier, 
            dropout_rate=self.dropout,
            imin=self.imin,
            imax=self.imax,
        )
        self.distance_identifier = _get_string(experiment.get('distance'))
        self.distance = _get_distance(self.distance_identifier)
        self.is_loss_weighting = _get_boolean(experiment.get('loss_weight'))
        self.loss_weighting = _get_loss_weighting(self.is_loss_weighting)
        self.learning_rate = _get_float(experiment.get('learning_rate'))
        self.ema = _get_float(experiment.get('EMA_decay_rate'), error_if_none=False)
        self.optimizer_identifier = _get_string(experiment.get('optimizer'))
        self.optimizer = _get_optimizer(self.optimizer_identifier)
        self.mu0 = _get_float(experiment.get('mu0'), error_if_none=False)
        self.s0 = _get_float(experiment.get('s0'), error_if_none=False)
        self.s1 = _get_float(experiment.get('s1'), error_if_none=False)
        self.discretization_steps_identifier = experiment.get('discretization_steps')
        self.discretization_steps = _get_N(self.discretization_steps_identifier, self.s0, self.s1, self.training_iterations)
        self.mu_identifier = experiment.get('mu')
        self.mu = _get_mu(self.mu_identifier, self.s0, self.mu0, self.discretization_steps)
        self.noise_schedule = _get_noise_schedule(self.min_noise, self.max_noise)
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

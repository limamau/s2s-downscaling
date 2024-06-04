import os, yaml
import optax

import xarray as xr

from models.networks.unet import UNet
from .distances import *
from .schedules import *


def _get_string(value):
    if value is None:
        return ValueError("String must be provided.")
    else:
        return str(value)
    
    
def _get_network(model_identifier):
    net_map = {
        'unet': UNet(),
    }
    if model_identifier not in net_map:
        raise ValueError("Unsupported model: {}".format(model_identifier))
    return net_map[model_identifier]


def _get_distance(dist_identifier):
    dist_map = {
        'l1': l1,
        'l2': l2,
    }
    if dist_identifier not in dist_map:
        raise ValueError("Unsupported distance: {}".format(dist_identifier))
    return dist_map[dist_identifier]


def _get_optimizer(optimizer_identifier, learning_rate, ema):
    optimizer_map = {
        'adam': optax.adam(learning_rate),
        'radam': optax.radam(learning_rate),
    }
    if optimizer_identifier not in optimizer_map:
        raise ValueError("Unsupported optimizer: {}".format(optimizer_identifier))
    return optax.chain(
        optimizer_map[optimizer_identifier],
        optax.ema(ema)
    )


def _get_data(data_dir):
    if data_dir is not None:
        cpc_file = os.path.join(data_dir, "cpc.h5")
        cpc_ds = xr.open_dataset(cpc_file, engine='h5netcdf')
        cpc_data = cpc_ds['precip'].values
        return cpc_data


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
    
    
def _get_N(N, s0, s1, training_iterations):
    if isinstance(N, int):
        return lambda k: N
    elif N == 'schedule':
        if s0 == None:
            raise ValueError("s0 cannot be None for N schedule.")
        elif s1 == None:
            raise ValueError("s1 cannot be None for N schedule.")
        else:
            return N_schedule(s0, s1, training_iterations)
    else:
        raise ValueError("Unsupported N: {}".format(N))


def _get_mu(mu, s0, mu0):
    if isinstance(mu, float):
        return lambda k: mu
    elif mu == 'schedule':
        if s0 == None:
            raise ValueError("s0 cannot be None for EMA schedule.")
        elif mu0 == None:
            raise ValueError("mu0 cannot be None for EMA schedule.")
        else:
            return mu_schedule(s0, mu0)
    else:
        raise ValueError("Unsupported EMA decay rate: {}".format(mu))
    
    
def _get_tn(tmin, tmax):
    return tn_schedule(tmin, tmax)


def _get_dimensions(dimensions):
    if dimensions is None:
        return None
    else:
        return tuple(dimensions)
    

class Experiment:
    """
    Class to hold the experiment configuration.
    """
    def __init__(self, experiment_file):
        with open(experiment_file, 'r') as file:
            experiment = yaml.safe_load(file)
        
        self.experiment_name = _get_string(experiment.get('experiment_name'))
        self.data = _get_data(experiment.get('data_dir'))
        self.batch_size = _get_int(experiment.get('batch_size'))
        self.network_identifier = _get_string(experiment.get('network'))
        self.network = _get_network(self.network_identifier)
        self.distance_identifier = _get_string(experiment.get('distance'))
        self.distance = _get_distance(self.distance_identifier)
        self.learning_rate = _get_float(experiment.get('learning_rate'))
        self.ema = _get_float(experiment.get('EMA_decay_rate'), error_if_none=False)
        self.optimizer_identifier = _get_string(experiment.get('optimizer'))
        self.optimizer = _get_optimizer(self.optimizer_identifier, self.learning_rate, self.ema)
        self.tmin = _get_float(experiment.get('tmin'))
        self.tmax = _get_float(experiment.get('tmax'))
        self.mu0 = _get_float(experiment.get('mu0'), error_if_none=False)
        self.s0 = _get_float(experiment.get('s0'), error_if_none=False)
        self.s1 = _get_float(experiment.get('s1'), error_if_none=False)
        self.training_iterations = _get_int(experiment.get('training_iterations'))
        self.N_identifier = experiment.get('N')
        self.N = _get_N(self.N_identifier, self.s0, self.s1, self.training_iterations)
        self.mu_identifier = experiment.get('mu')
        self.mu = _get_mu(self.mu_identifier, self.s0, self.mu0)
        self.tn = _get_tn(self.tmin, self.tmax)
        self.sigma_data = _get_float(experiment.get('sigma_data'))
        self.sigma_star = _get_float(experiment.get('sigma_star'))
        self.log_each = _get_int(experiment.get('log_each'))
        self.ckpt_each = _get_int(experiment.get('checkpoint_each'))
        self.experiment_dir = os.path.join(os.path.dirname(experiment_file), "experiments", self.experiment_name)
        
        # in the case where one is creating an experiment the following will be none
        self.dimensions = _get_dimensions(experiment.get('dimensions'))
        self.data_mean = _get_float(experiment.get('data_mean'), error_if_none=False)
        self.data_std = _get_float(experiment.get('data_std'), error_if_none=False)

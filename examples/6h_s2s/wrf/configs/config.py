import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.output_dir = "2018061100_analysis"
    config.cpc_file = "cpc.h5"
    config.wrf_simulations_dir = "wrf"
    
    return config

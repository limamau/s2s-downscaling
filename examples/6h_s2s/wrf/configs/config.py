import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.forecast_date = "2018-06-04"
    config.event_date = "2018-06-11"
    config.member_idx = 6
    config.output_dir = f"{config.event_date.replace('-', '')}00_{config.member_idx}_{config.forecast_date}_analysis"
    config.cpc_file = "cpc.h5"
    config.s2s_file = "ens_s2s_nearest.h5"
    config.lead_time_idx = 0
    config.wrf_simulations_dir = "wrf"

    return config

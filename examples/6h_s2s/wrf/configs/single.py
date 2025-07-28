import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.forecast_date = "2018-05-21"
    config.lead_time_idx = 0  # only used for analysis, not needed for processing
    config.event_date = "2018-06-11"
    config.member_idx = 14  # indexing on 0
    config.output_dir = f"{config.event_date.replace('-', '')}00_{config.member_idx + 1}_{config.forecast_date}"
    config.cpc_file = "cpc.h5"
    config.s2s_file = "ens_s2s_nearest.h5"
    config.single_wrf_simulations = "wrf/single"

    return config

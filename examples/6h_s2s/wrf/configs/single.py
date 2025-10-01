import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.event_date = "2021-06-28"
    config.forecast_date = "2021-06-07"
    config.lead_time_idx = 0  # only used for analysis, not needed for processing
    config.member_idx = 47  # starting the indexing on 0
    config.output_dir = f"{config.event_date.replace('-', '')}00_{config.member_idx + 1}_{config.forecast_date}"
    config.cpc_file = "cpc.h5"
    config.s2s_file = "ens_s2s_nearest.h5"
    config.single_wrf_simulations = "wrf/single"

    return config

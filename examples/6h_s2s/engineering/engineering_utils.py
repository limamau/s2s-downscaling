import numpy as np


def get_s2s_idxs_to_keep(times, bound_dates):
    idxs_to_keep = []
    start, end = bound_dates
    start = np.datetime64(start) + np.timedelta64(6, 'h')
    end = np.datetime64(end) + np.timedelta64(24, 'h')
    idxs_to_keep = [i for i, t in enumerate(times) if start <= t <= end]
    return idxs_to_keep

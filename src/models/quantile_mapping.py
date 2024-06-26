import numpy as np
from utils import get_cdf


class QuantileMapping:
    def __init__(self, n_quantiles=500, method='all'):
        if method == 'all':
            self.fit = self._fit_all
            self.predict = self._predict_all
        elif method == 'point_to_point':
            self.fit = self._fit_point_to_point
            self.predict = self._predict_point_to_point
        else:
            raise ValueError("Method should be either 'all' or 'point_to_point'.")
        
        self.n_quantiles = n_quantiles
        
        
    def _fit_all(self, obs, simh):
        global_min = min(np.nanmin(obs), np.nanmin(simh))
        global_max = max(np.nanmax(obs), np.nanmax(simh))

        wide = abs(global_max - global_min) / self.n_quantiles
        self.bins = np.arange(global_min, global_max + wide, wide)

        self.cdf_obs = get_cdf(obs, self.bins)
        self.cdf_simh = get_cdf(simh, self.bins)

        self.cdf_simh = np.interp(
            self.cdf_simh, 
            (self.cdf_simh.min(), self.cdf_simh.max()), 
            (self.cdf_obs.min(), self.cdf_obs.max())
        )


    def _fit_point_to_point(self, obs, simh):
        _, Ny, Nx = obs.shape
        self.bins = np.empty((Ny, Nx, self.n_quantiles+1))
        self.cdf_obs = np.empty((Ny, Nx, self.n_quantiles+1))
        self.cdf_simh = np.empty((Ny, Nx, self.n_quantiles+1))

        for j in range(Ny):
            for i in range(Nx):
                local_min = min(np.nanmin(obs[:, j, i]), np.nanmin(simh[:, j, i]))
                local_max = max(np.nanmax(obs[:, j, i]), np.nanmax(simh[:, j, i]))

                wide = abs(local_max - local_min) / self.n_quantiles
                bins = np.arange(local_min, local_max + wide, wide)
                if len(bins) > self.n_quantiles + 1:
                    bins = bins[:-1]
                self.bins[j, i, :] = bins

                self.cdf_obs[j, i, :] = get_cdf(obs[:, j, i], self.bins[j, i, :])
                self.cdf_simh[j, i, :] = get_cdf(simh[:, j, i], self.bins[j, i, :])

                self.cdf_simh[j, i, :] = np.interp(
                    self.cdf_simh[j, i, :], 
                    (self.cdf_simh[j, i, :].min(), self.cdf_simh[j, i, :].max()), 
                    (self.cdf_obs[j, i, :].min(), self.cdf_obs[j, i, :].max())
                )


    def _predict_all(self, simp):
        epsilon = np.interp(simp, self.bins, self.cdf_simh)
        corrected_simp = np.interp(epsilon, self.cdf_obs, self.bins)
        return corrected_simp


    def _predict_point_to_point(self, simp):
        Nt, Ny, Nx = simp.shape
        corrected_simp = np.empty((Nt, Ny, Nx))

        for y in range(Ny):
            for x in range(Nx):
                epsilon = np.interp(simp[:, y, x], self.bins[y, x, :], self.cdf_simh[y, x, :])
                corrected_simp[:, y, x] = np.interp(epsilon, self.cdf_obs[y, x, :], self.bins[y, x, :])

        return corrected_simp
    


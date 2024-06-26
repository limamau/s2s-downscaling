import numpy as np
from scipy.special import gamma, gammainc, gammaincinv
from scipy.optimize import minimize

class GeneralizedGamma:
    def __init__(self, method='all'):
        if method == 'all':
            self.fit = self._fit_all
            self.predict = self._predict_all
        elif method == 'point_to_point':
            self.fit = self._fit_point_to_point
            self.predict = self._predict_point_to_point
        else:
            raise ValueError("Method should be either 'all' or 'point_to_point'.")

        self.obs_params = None
        self.simh_params = None
        
        
    def _pdf(self, x, a, d, p):
        r = (p/a**d) * x**(d-1) * np.exp(-(x/a)**p) / gamma(d/p)
        return r


    def _cdf(self, x, a, d, p):
        return gammainc(d/p, (x/a)**p)


    def _ppf(self, q, a, d, p):
        return a * gammaincinv(d/p, q)**(1/p)

    
    def _minimize(self, data):
        def negative_likelihood(params, data):
            pdf_values = self._pdf(data, *params)
            r = -np.sum(np.nan_to_num(pdf_values))
            return r
        
        initial_params = np.array([10.0, 2.0, 1.0])
        bounds = [(1e-5, None), (1.0, None), (1e-5, None)]
        options = {'maxiter': 10}
        
        result = minimize(
            fun=negative_likelihood, 
            x0=initial_params,
            method='Powell',
            args=(data,), 
            bounds=bounds, 
            options=options
        )
        if result.success:
            return result.x
        else:
            print("No convergence until maxiter.")
            print("Current params: ", result.x)
            return result.x


    def _fit_all(self, obs, simh):
        self.obs_params = self._minimize(obs)
        self.simh_params = self._minimize(simh)
        
        
    def _fit_point_to_point(self, obs, simh):
        _, Ny, Nx = obs.shape
        self.obs_params = np.empty((Ny, Nx, 3))
        self.simh_params = np.empty((Ny, Nx, 3))
        
        for j in range(Ny):
            for i in range(Nx):
                self.obs_params[j, i, :] = np.array([10.0, 2.0, 1.0]) # self._minimize(obs[:, j, i])
                self.simh_params[j, i, :] = np.array([10.0, 2.0, 1.0]) # self._minimize(simh[:, j, i])


    def _predict_all(self, simp):
        cdf_simp = self._cdf(simp, *self.simh_params)
        return self._ppf(cdf_simp, *self.obs_params)
    
    
    def _predict_point_to_point(self, simp):
        Nt, Ny, Nx = simp.shape
        corrected_simp = np.empty((Nt, Ny, Nx))
        
        for y in range(Ny):
            for x in range(Nx):
                cdf_simp = self._cdf(simp[:, y, x], *self.simh_params[y, x, :])
                corrected_simp[:, y, x] = self._ppf(cdf_simp, *self.obs_params[y, x, :])
                
        return corrected_simp
        
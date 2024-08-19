import numpy as np

from debiasing.quantile_mapping import quantile_mapping

def test_quantile_mapping():
    matrix = np.full((5, 5), 2)
    matrix[0, :] = 1
    matrix[4, :] = 1
    matrix[:, 4] = 1
    matrix[:, 0] = 1
    matrix[2, 2] = 3
    
    obs = matrix * 2
    simh = matrix
    simp = matrix
    
    simp_corrected = quantile_mapping(obs, simh, simp)
    
    assert np.allclose(simp_corrected, obs, atol=1e-1)
    
    print("quantile_mapping.py passed!")

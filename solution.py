import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import t, expon

chat_id = 767458283 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    
    n_resamples = 1000
    alpha = 1 - p
    
    bootstrap_samples = np.random.choice(x, (n_resamples, len(x)), replace=True)
    bootstrap_accelerations = 2 * np.mean(bootstrap_samples, axis=1) / 17**2
    sorted_bootstrap_accelerations = np.sort(bootstrap_accelerations)
    
    lower_index = int(n_resamples * alpha / 2)
    upper_index = int(n_resamples * (1 - alpha / 2))
    
    return sorted_bootstrap_accelerations[lower_index], sorted_bootstrap_accelerations[upper_index]


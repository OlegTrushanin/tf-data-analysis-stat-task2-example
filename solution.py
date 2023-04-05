import pandas as pd
import numpy as np

from scipy.stats import norm


chat_id = 767458283 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    n = len(x)
    alpha = 1 - p
    t_val = t.ppf(1 - alpha / 2, n - 1)
    mean_x = np.mean(x)
    std_error = np.sqrt(1 / (2 * n))

    left_bound = mean_x - t_val * std_error
    right_bound = mean_x + t_val * std_error

    return left_bound, right_bound

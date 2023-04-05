import pandas as pd
import numpy as np

from scipy.stats import norm


chat_id = 767458283 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    n = len(x)
    alpha = 1 - p
    df = n - 1
    t_val = t.ppf(1 - alpha / 2, df)
    s = np.std(x, ddof=1)
    mean_x = np.mean(x)

    left_bound = mean_x - t_val * s / np.sqrt(n) * np.sqrt(2 / 3)
    right_bound = mean_x + t_val * s / np.sqrt(n) * np.sqrt(2 / 3)

    return [left_bound, right_bound]

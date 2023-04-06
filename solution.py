import pandas as pd
import numpy as np
from scipy.stats import t

from scipy.stats import norm


chat_id = 767458283 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    t_squared = 17 ** 2
    s = x
    n = len(x)
    y = s
    x = np.ones((n, 2))
    x[:, 1] = t_squared / 2
    beta_hat = np.linalg.inv(x.T @ x) @ x.T @ y
    a_hat = beta_hat[1]

    # Строим доверительный интервал для коэффициента ускорения
    alpha = 1 - p
    t_alpha_2 = t.ppf(1 - alpha / 2, n - 2)
    se = np.sqrt(np.sum((y - x @ beta_hat) ** 2) / (n - 2))
    ci_left = a_hat - t_alpha_2 * se / np.sqrt(np.sum((t_squared / 2 - np.mean(t_squared)) ** 2))
    ci_right = a_hat + t_alpha_2 * se / np.sqrt(np.sum((t_squared / 2 - np.mean(t_squared)) ** 2))

    return ci_left, ci_right

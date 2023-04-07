import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import t, expon

chat_id = 767458283 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
     # Количество измерений
    n = len(x)

    # Время эксперимента
    t_exp = 17

    # Определение ускорения и его стандартной ошибки
    a = 2 * x.mean() / (t_exp ** 2)
    error = expon(scale=1/2).std()
    se_a = 2 * error / (t_exp ** 2 * np.sqrt(n))

    # Вычисление границ доверительного интервала
    df = n - 1
    t_critical = t.ppf(1 - (1 - p) / 2, df)
    lower = a - t_critical * se_a
    upper = a + t_critical * se_a

    return lower, upper


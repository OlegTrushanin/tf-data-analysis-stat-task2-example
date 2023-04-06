import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import expon


chat_id = 767458283 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    time = 17.0

    def negative_log_likelihood(acceleration):
        predicted_distances = 0.5 * acceleration * time**2
        errors = x - predicted_distances
        return -np.sum(expon.logpdf(errors + 0.5, scale=1))

    # Оценка коэффициента ускорения с помощью метода максимального правдоподобия
    mle_result = minimize(negative_log_likelihood, 9.8)
    mle_acceleration = mle_result.x[0]

    # Построение профиля правдоподобия для оценки доверительного интервала
    alpha = 1 - p
    log_likelihood_mle = -mle_result.fun
    critical_value = expon.ppf(1 - alpha / 2)

    def find_confidence_interval_bound(initial_guess, target_log_likelihood):
        result = minimize(lambda acceleration: (negative_log_likelihood(acceleration) - target_log_likelihood)**2, initial_guess)
        return result.x[0]

    lower_bound = find_confidence_interval_bound(mle_acceleration - 1, log_likelihood_mle + critical_value)
    upper_bound = find_confidence_interval_bound(mle_acceleration + 1, log_likelihood_mle + critical_value)

    return lower_bound, upper_bound


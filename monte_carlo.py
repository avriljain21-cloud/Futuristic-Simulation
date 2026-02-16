import numpy as np

# -------------------- Monte Carlo Simulation --------------------
def simulate_price_paths(S0, mu, sigma, days, sims, shock_size=0.0, shock_day=None):
    """
    Simulate future stock prices using Geometric Brownian Motion (GBM),
    with optional sudden shock applied on a specific day.

    Parameters:
    - S0: float, current stock price
    - mu: float, mean daily log return
    - sigma: float, daily volatility (std dev of log returns)
    - days: int, number of days to simulate
    - sims: int, number of simulation paths
    - shock_size: float, optional sudden price jump (%) applied on shock_day
    - shock_day: int, optional day (1-indexed) to apply the shock

    Returns:
    - paths: 2D numpy array (days x sims) of simulated prices
    """
    S0 = float(S0)
    mu = float(mu)
    sigma = float(sigma)

    dt = 1  # daily steps
    paths = np.zeros((days, sims))
    paths[0] = S0

    for t in range(1, days):
        z = np.random.standard_normal(sims)
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

        # Apply sudden shock if specified
        if shock_day and t == shock_day - 1:
            paths[t] *= (1 + shock_size / 100)

    return paths

# -------------------- Statistics --------------------
def get_price_statistics(paths):
    """
    Calculate mean, median, and 5% / 95% percentiles of final prices.
    """
    final_prices = paths[-1]
    stats = {
        "mean": float(np.mean(final_prices)),
        "median": float(np.median(final_prices)),
        "p5": float(np.percentile(final_prices, 5)),
        "p95": float(np.percentile(final_prices, 95))
    }
    return stats

# -------------------- Time to Target --------------------
def time_to_target(paths, target):
    """
    Estimate how long it takes for each simulation path to reach a target price.

    Returns a list of days for each path that hits the target.
    """
    target = float(target)
    hit_days = []
    for sim in range(paths.shape[1]):
        hits = np.where(paths[:, sim] >= target)[0]
        if len(hits) > 0:
            hit_days.append(hits[0])
    return hit_days

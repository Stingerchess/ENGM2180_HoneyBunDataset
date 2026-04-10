import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from math import log, sqrt, exp, erf


# ==========================================
# 1. LOAD AND PREPARE HONBUN DATA
# ==========================================
def load_honbun_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    # data = pd.read_excel(file_path)

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Closing Price"] = pd.to_numeric(data["Closing Price"], errors="coerce")
    data = data.sort_values("Date").dropna(subset=["Date", "Closing Price"]).copy()

    data["LogReturn"] = np.log(data["Closing Price"] / data["Closing Price"].shift(1))
    data = data.dropna().reset_index(drop=True)

    return data


# ==========================================
# 2. ESTIMATE HISTORICAL PARAMETERS
# ==========================================
def estimate_parameters(data: pd.DataFrame):
    mu_daily = data["LogReturn"].mean()
    sigma_daily = data["LogReturn"].std()

    mu_annual = mu_daily * 252
    sigma_annual = sigma_daily * np.sqrt(252)

    return mu_daily, sigma_daily, mu_annual, sigma_annual


# ==========================================
# 3. PLOT HISTORICAL DATA
# ==========================================
def plot_historical_prices(data: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.plot(data["Date"], data["Closing Price"], label="HONBUN Closing Price")
    plt.title("Honey Bun (HONBUN) JSE Stock Price - Last 5 Years")
    plt.xlabel("Date")
    plt.ylabel("Price (JMD)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_log_returns(data: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.plot(data["Date"], data["LogReturn"], label="HONBUN Log Returns")
    plt.title("Honey Bun (HONBUN) Log Returns - Last 5 Years")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. GBM STOCK-PATH SIMULATION
# ==========================================
def simulate_gbm_paths(
    S0: float,
    mu_daily: float,
    sigma_daily: float,
    num_steps: int,
    num_paths: int,
    seed: int = 42
) -> np.ndarray:
    np.random.seed(seed)
    paths = np.zeros((num_steps, num_paths))
    paths[0, :] = S0

    for j in range(num_paths):
        Z = np.random.normal(0, 1, num_steps - 1)
        for i in range(1, num_steps):
            paths[i, j] = paths[i - 1, j] * np.exp(
                (mu_daily - 0.5 * sigma_daily**2) + sigma_daily * Z[i - 1]
            )

    return paths


def plot_gbm_paths(data: pd.DataFrame, paths: np.ndarray):
    plt.figure(figsize=(12, 6))
    for j in range(paths.shape[1]):
        plt.plot(data["Date"], paths[:, j], label=f"Path {j+1}")

    plt.title("GBM Simulation of HONBUN Stock Price (5 Years)")
    plt.xlabel("Year")
    plt.ylabel("Price (JMD)")

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================================
# 5. MONTE CARLO STOCK-PATH SIMULATION
# ==========================================
def simulate_monte_carlo_stock_paths(
    S0: float,
    mu_annual: float,
    sigma_annual: float,
    T: float = 1.0,
    N: int = 252,
    num_paths: int = 1000,
    seed: int = 42
) -> np.ndarray:
    np.random.seed(seed)
    dt = T / N

    paths = np.zeros((N + 1, num_paths))
    paths[0, :] = S0

    for t in range(1, N + 1):
        Z = np.random.normal(0, 1, num_paths)
        paths[t, :] = paths[t - 1, :] * np.exp(
            (mu_annual - 0.5 * sigma_annual**2) * dt
            + sigma_annual * np.sqrt(dt) * Z
        )

    return paths


def plot_monte_carlo_stock_paths(start_date: pd.Timestamp, paths: np.ndarray):
    future_dates = pd.bdate_range(start=start_date, periods=paths.shape[0])

    plt.figure(figsize=(12, 6))
    for j in range(min(50, paths.shape[1])):
        plt.plot(future_dates, paths[:, j], linewidth=0.8)

    plt.title("Monte Carlo Simulation of HONBUN Stock Price (1000 Paths, 1-Year Horizon)")
    plt.xlabel("Year")
    plt.ylabel("Price (JMD)")

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_terminal_distribution(terminal_prices: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.hist(terminal_prices, bins=40, edgecolor="black")
    plt.title("Distribution of Terminal HONBUN Prices After 1 Year")
    plt.xlabel("Terminal Stock Price (JMD)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_terminal_statistics(S0: float, terminal_prices: np.ndarray):
    print("\n=== Terminal Price Statistics ===")
    print(f"Initial stock price S0: {S0:.2f}")
    print(f"Mean terminal price: {terminal_prices.mean():.4f}")
    print(f"Median terminal price: {np.median(terminal_prices):.4f}")
    print(f"Standard deviation of terminal prices: {terminal_prices.std(ddof=1):.4f}")
    print(f"Minimum terminal price: {terminal_prices.min():.4f}")
    print(f"Maximum terminal price: {terminal_prices.max():.4f}")
    print(f"5th percentile: {np.percentile(terminal_prices, 5):.4f}")
    print(f"95th percentile: {np.percentile(terminal_prices, 95):.4f}")


def print_volatility_comparison(
    sigma_daily: float,
    sigma_annual: float,
    stock_paths: np.ndarray
):
    sim_log_returns = np.log(stock_paths[1:, :] / stock_paths[:-1, :])
    sim_daily_vol = sim_log_returns.std()
    sim_annual_vol = sim_daily_vol * np.sqrt(252)

    print("\n=== Volatility Comparison ===")
    print(f"Historical daily volatility: {sigma_daily:.6f}")
    print(f"Simulated daily volatility:  {sim_daily_vol:.6f}")
    print(f"Historical annual volatility: {sigma_annual:.6f}")
    print(f"Simulated annual volatility:  {sim_annual_vol:.6f}")


# ==========================================
# 6. OPTION PRICING FUNCTIONS
# ==========================================
def N(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def black_scholes_option(S0, K, r, sigma, T):
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    option_price = S0 * N(d1) - K * exp(-r * T) * N(d2)
    return option_price, d1, d2


def monte_carlo_option(S0, K, r, sigma, T, num_simulations=100000, seed=42):
    np.random.seed(seed)
    Z = np.random.normal(0, 1, num_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0)

    option_price = exp(-r * T) * np.mean(payoffs)
    std_error = exp(-r * T) * np.std(payoffs, ddof=1) / sqrt(num_simulations)

    return option_price, std_error


# ==========================================
# 7. OPTION SENSITIVITY ANALYSIS
# ==========================================
def sensitivity_analysis(S0, K, r, sigma, T):
    vol_values = [0.20, 0.40, sigma, 0.70, 0.90]
    rate_values = [0.01, 0.03, r, 0.08, 0.10]
    time_values = [0.25, 0.50, T, 1.50, 2.00]

    vol_results = []
    for vol in vol_values:
        price, _, _ = black_scholes_option(S0, K, r, vol, T)
        vol_results.append([vol, price])

    rate_results = []
    for rate in rate_values:
        price, _, _ = black_scholes_option(S0, K, rate, sigma, T)
        rate_results.append([rate, price])

    time_results = []
    for maturity in time_values:
        price, _, _ = black_scholes_option(S0, K, r, sigma, maturity)
        time_results.append([maturity, price])

    vol_df = pd.DataFrame(vol_results, columns=["Volatility", "Option Price (JMD)"])
    rate_df = pd.DataFrame(rate_results, columns=["Interest Rate", "Option Price (JMD)"])
    time_df = pd.DataFrame(time_results, columns=["Time to Maturity (Years)", "Option Price (JMD)"])

    return vol_df, rate_df, time_df


def plot_sensitivity(vol_df, rate_df, time_df):
    plt.figure(figsize=(8, 5))
    plt.plot(vol_df["Volatility"], vol_df["Option Price (JMD)"], marker="o")
    plt.title("Sensitivity of Option Price to Volatility")
    plt.xlabel("Volatility")
    plt.ylabel("Option Price (JMD)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(rate_df["Interest Rate"], rate_df["Option Price (JMD)"], marker="o")
    plt.title("Sensitivity of Option Price to Interest Rate")
    plt.xlabel("Risk-Free Interest Rate")
    plt.ylabel("Option Price (JMD)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(time_df["Time to Maturity (Years)"], time_df["Option Price (JMD)"], marker="o")
    plt.title("Sensitivity of Option Price to Time to Maturity")
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Option Price (JMD)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================================
# 8. MAIN SCRIPT
# ==========================================
data = load_honbun_data("honbun_price_history_full.csv")

mu_daily, sigma_daily, mu_annual, sigma_annual = estimate_parameters(data)

print("=== Historical Parameter Estimates ===")
print(f"Daily mean return: {mu_daily:.6f}")
print(f"Daily volatility: {sigma_daily:.6f}")
print(f"Annualized mean return: {mu_annual:.6f}")
print(f"Annualized volatility: {sigma_annual:.6f}")

plot_historical_prices(data)
plot_log_returns(data)

# GBM simulation
gbm_paths = simulate_gbm_paths(
    S0=float(data["Closing Price"].iloc[0]),
    mu_daily=mu_daily,
    sigma_daily=sigma_daily,
    num_steps=len(data),
    num_paths=5
)
plot_gbm_paths(data, gbm_paths)

# Monte Carlo stock simulation
stock_mc_paths = simulate_monte_carlo_stock_paths(
    S0=float(data["Closing Price"].iloc[-1]),
    mu_annual=mu_annual,
    sigma_annual=sigma_annual,
    T=1.0,
    N=252,
    num_paths=1000
)

plot_monte_carlo_stock_paths(data["Date"].iloc[-1], stock_mc_paths)

terminal_prices = stock_mc_paths[-1, :]
plot_terminal_distribution(terminal_prices)

print_terminal_statistics(float(data["Closing Price"].iloc[-1]), terminal_prices)
print_volatility_comparison(sigma_daily, sigma_annual, stock_mc_paths)

# ==========================================
# 9. OPTION PRICING INPUTS
# ==========================================
S0_option = float(data["Closing Price"].iloc[-1])   # current HONBUN price
K = 7.00                                             # strike price
r = 0.055                                            # risk-free rate
sigma = sigma_annual                                 # annualized volatility
T = 1.0                                              # time to maturity
num_simulations = 100000

bs_price, d1, d2 = black_scholes_option(S0_option, K, r, sigma, T)
mc_price, mc_error = monte_carlo_option(S0_option, K, r, sigma, T, num_simulations)

print("\n=== PART 4: EUROPEAN OPTION PRICING ===")
print(f"Current stock price S0: {S0_option:.2f}")
print(f"Strike price K: {K:.2f}")
print(f"Risk-free rate r: {r:.3f}")
print(f"Volatility sigma: {sigma:.6f}")
print(f"Time to maturity T: {T:.2f} year(s)\n")

print("=== Black-Scholes Results ===")
print(f"d1 = {d1:.6f}")
print(f"d2 = {d2:.6f}")
print(f"Black-Scholes Option Price = {bs_price:.4f} JMD\n")

print("=== Monte Carlo Results ===")
print(f"Monte Carlo Option Price = {mc_price:.4f} JMD")
print(f"Monte Carlo Standard Error = {mc_error:.4f}\n")

print("=== Comparison ===")
print(f"Absolute Difference = {abs(bs_price - mc_price):.4f} JMD\n")

# Sensitivity
vol_df, rate_df, time_df = sensitivity_analysis(S0_option, K, r, sigma, T)

print("=== Sensitivity to Volatility ===")
print(vol_df.to_string(index=False))
print()

print("=== Sensitivity to Interest Rate ===")
print(rate_df.to_string(index=False))
print()

print("=== Sensitivity to Time to Maturity ===")
print(time_df.to_string(index=False))
print()

plot_sensitivity(vol_df, rate_df, time_df)
from datetime import datetime, timedelta, timezone
import pytz
import sys
import numpy as np
import pandas as pd

# PyEMD for CEEMDAN
from PyEMD import CEEMDAN

# Scikit-learn
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_absolute_percentage_error


# Dummy implementation for IB API data retrieval.
# Replace this with your actual get_historical_data_from_ibapi function.
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import *
from ibapi.ticktype import *
from ibapi.utils import iswrapper
import time
import pandas as pd

"""
================================================================================
CEEMDAN–Bagging–HHO–SVR Forecasting Pipeline for Natural Gas Futures (NGM5)
================================================================================

OVERVIEW
--------
This script is an end-to-end machine learning pipeline for forecasting the 
Natural Gas Futures (NGM5) contract using a hybrid model that integrates:

- CEEMDAN: Complete Ensemble Empirical Mode Decomposition with Adaptive Noise,
  used for breaking the price series into interpretable intrinsic mode functions.
  
- BaggingRegressor: Ensemble learning technique to reduce overfitting, wrapping
  Support Vector Regression (SVR) as the base model.

- SVR (Support Vector Regression): Captures nonlinear trends in decomposed signals.

- HHO (Harris Hawks Optimization): Nature-inspired metaheuristic optimization algorithm 
  used to fine-tune SVR hyperparameters [C, epsilon, gamma].

- IB API: Pulls actual historical daily price bars for the Natural Gas Futures 
  contract (NGM5) via Interactive Brokers TWS or Gateway.

PIPELINE WORKFLOW
-----------------
1. Connects to Interactive Brokers via IBAPI and retrieves historical data 
   for contract symbol "NG" with expiry "20250528" from the NYMEX exchange.

2. Applies CEEMDAN to decompose the time series into multiple IMFs (intrinsic 
   mode functions) and a final residue, making the signal easier to model.

3. For each IMF:
   - Transforms the sequence into supervised format using a sliding window.
   - Optimizes SVR hyperparameters (C, ε, γ) using Harris Hawks Optimization (HHO).
   - Trains a Bagging ensemble of SVRs with those optimized hyperparameters.

4. Combines predictions across all components for multi-step time series forecasting.

5. Constructs 95% prediction intervals by running Monte Carlo simulations
   using randomly sampled base SVRs from each Bagging ensemble.

6. Prints the forecast, intervals, and logs the results (with hyperparameters)
   to a timestamped text file (`forecast_history_log.txt`).

USE CASE
--------
This model is designed for financial market forecasting—specifically,
for the Natural Gas Futures market (e.g., NGM5). It can be used as part 
of a systematic trading pipeline, signal generation engine, or for 
academic research into hybrid time series forecasting.

MODEL PREDICTABILITY
--------------------
This hybrid approach can become reasonably predictable over short horizons 
(1 to 7 days), given that:

- CEEMDAN helps reduce signal complexity and separate different frequency components.
- SVR captures nonlinear patterns effectively.
- Bagging helps reduce variance and overfitting.
- HHO ensures the SVRs are optimally tuned per component.

However, performance depends heavily on:

- The quality and amount of training data.
- Stationarity and noise characteristics of the time series.
- Structural breaks, macro shocks, or unexpected volatility events.

As with all models, this system does **not predict extreme events or black swan
scenarios**. It performs best during relatively stable or cyclical market regimes.

Accuracy can improve further with:
- More frequent retraining
- Feature engineering (e.g. adding fundamental drivers)
- Regime-switching models or volatility filters

LIMITATIONS
-----------
- This model is computationally expensive due to CEEMDAN and HHO.
- It does not handle missing data, gaps, or holidays.
- Requires TWS/Gateway to be running and connected to IB.

RECOMMENDATIONS
---------------
- Use for short-term directional bias or support/resistance modeling.
- Do not treat forecast as an exact price target, but as a probabilistic estimate.
- Use prediction intervals to quantify model confidence.
- Always complement with risk management and position sizing strategies.

================================================================================
"""
class IBDataApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.req_completed = False

    @iswrapper
    def historicalData(self, reqId, bar):
        self.data.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])

    @iswrapper
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        self.req_completed = True


def get_historical_data_from_ibapi(host='127.0.0.1', port=7496, client_id=8,
                                   contract_symbol="NG", exchange="NYMEX",
                                   currency="USD",
                                   endDateTime="",  # 'YYYYMMDD HH:MM:SS'
                                   durationStr="1 Y", barSizeSetting="1 day",
                                   whatToShow="TRADES", useRTH=0):
    """
    Connects to IB, requests historical data for the NG Futures contract (NGM5),
    and returns a pandas DataFrame.
    """
    import threading

    app = IBDataApp()
    app.connect(host, port, client_id)

    contract = Contract()
    contract.symbol = contract_symbol
    contract.secType = "FUT"
    contract.exchange = exchange
    contract.currency = currency
    contract.lastTradeDateOrContractMonth = "20250528"
    contract.localSymbol = "NGM5"
    contract.multiplier = "10000"

    # Start the IBAPI event loop in a background thread:
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    # Request historical data
    req_id = 1
    app.reqHistoricalData(
        reqId=req_id,
        contract=contract,
        endDateTime=endDateTime,
        durationStr=durationStr,
        barSizeSetting=barSizeSetting,
        whatToShow=whatToShow,
        useRTH=useRTH,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[]
    )

    # Wait until data is received
    timeout = time.time() + 30  # 30-second timeout
    while not app.req_completed and time.time() < timeout:
        time.sleep(1)

    app.disconnect()

    if not app.data:
        raise RuntimeError("No data received from IB.")

    df = pd.DataFrame(app.data, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df

def hho_optimize(objective_func, dim: int, lb: np.ndarray, ub: np.ndarray,
                 n_hawks: int = 20, max_iter: int = 30,
                 random_seed: int = 42):
    """
    Harris Hawks Optimization to minimize a given objective function.
    :param objective_func: Callable(params) -> float, returns an error measure to be minimized.
    :param dim: Dimension of the solution space (3 for [C, epsilon, gamma] in SVR).
    :param lb: Lower-bound array for each dimension.
    :param ub: Upper-bound array for each dimension.
    :param n_hawks: Population size (number of hawks).
    :param max_iter: Maximum number of iterations.
    :param random_seed: For reproducibility.
    :return: (best_position, best_fitness)
    """
    rng = np.random.default_rng(random_seed)

    # Initialize hawks positions randomly within bounds
    hawks = []
    for _ in range(n_hawks):
        hawk_pos = lb + rng.random(dim) * (ub - lb)
        hawks.append(hawk_pos)

    hawks = np.array(hawks)
    # Evaluate initial fitness
    fitness = np.array([objective_func(pos) for pos in hawks])

    # Identify initial best (rabbit)
    best_idx = np.argmin(fitness)
    rabbit_pos = hawks[best_idx].copy()
    rabbit_fit = fitness[best_idx]

    # Main optimization loop
    for t in range(1, max_iter + 1):
        # Compute rabbit's escaping energy E
        # E0 in [-1, 1]
        E0 = 2 * rng.random() - 1
        # linearly decreasing factor
        E = 2 * E0 * (1 - t / max_iter)

        # Average position (for some strategies)
        Xm = np.mean(hawks, axis=0)

        for i in range(n_hawks):
            X = hawks[i]
            f_curr = fitness[i]
            r1, r2, r3, r4 = rng.random(4)
            q = rng.random()

            if abs(E) >= 1:
                # Exploration phase
                if q < 0.5:
                    # Move towards a random hawk
                    rand_idx = rng.integers(n_hawks)
                    X_rand = hawks[rand_idx]
                    X_new = X_rand - r1 * np.abs(X_rand - 2 * r2 * X)
                else:
                    # Move considering rabbit, pop mean
                    X_new = (rabbit_pos - Xm) - r3 * (lb + r4 * (ub - lb))
            else:
                # Exploitation phase
                X_diff = rabbit_pos - X
                r = rng.random()
                J = 2 * (1 - rng.random())  # in [-2, 2]
                if r >= 0.5 and abs(E) >= 0.5:
                    # Soft besiege
                    X_new = X_diff - E * np.abs(J * rabbit_pos - X)
                elif r >= 0.5 and abs(E) < 0.5:
                    # Hard besiege
                    X_new = rabbit_pos - E * np.abs(X_diff)
                elif r < 0.5:
                    # Dive (progressive dives)
                    Y = rabbit_pos - E * np.abs(J * rabbit_pos - X)
                    Z = Y + rng.normal(size=dim) * 0.1 * (X - rabbit_pos)
                    fY = objective_func(Y)
                    fZ = objective_func(Z)
                    if fY < f_curr or fZ < f_curr:
                        if fY <= fZ:
                            X_new = Y
                            f_curr = fY
                        else:
                            X_new = Z
                            f_curr = fZ
                    else:
                        X_new = X
                else:
                    X_new = X

            # Bound check
            X_new = np.clip(X_new, lb, ub)

            # Evaluate
            f_new = objective_func(X_new)
            if f_new < f_curr:
                hawks[i] = X_new
                fitness[i] = f_new
                if f_new < rabbit_fit:
                    rabbit_pos = X_new.copy()
                    rabbit_fit = f_new

    return rabbit_pos, rabbit_fit


class CEEMDANBaggingHHOSVR:
    def __init__(self, n_estimators=15, population=20, iterations=30, lag=3,
                 seed=42):
        """
        Initialize the hybrid model.
        :param n_estimators: Number of SVR base learners for Bagging ensemble.
        :param population: Number of hawks (candidate solutions) in HHO.
        :param iterations: Max iterations for HHO.
        :param lag: Number of lag observations (window size).
        :param seed: Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.population = population
        self.iterations = iterations
        self.lag = lag
        self.seed = seed

        # Stored after training:
        self.component_models = []   # list of BaggingRegressor's for each IMF/residue
        self.component_series = []   # list of arrays, each is an IMF or residue
        self.component_params = []
        self._trained = False

    def decompose(self, series: np.ndarray):
        """
        Decompose input time series using CEEMDAN into IMFs plus final residue.
        Returns a list of components [IMF1, IMF2, ..., IMFk, Residue].
        """
        ceemdan = CEEMDAN(trials=100, epsilon=0.005)
        ceemdan.noise_seed(self.seed)  # for reproducibility

        imfs = ceemdan(series)
        # The PyEMD library also provides the separate method get_imfs_and_residue():
        imf_list, residue = ceemdan.get_imfs_and_residue()
        components = [imf for imf in imf_list]
        components.append(residue)
        return components

    def _prepare_supervised(self, arr: np.ndarray):
        """
        Make (X, y) from a 1D array, using self.lag for autoregression.
        """
        N = len(arr)
        p = self.lag
        if N <= p:
            raise ValueError("Series length must be > lag.")
        X, y = [], []
        for t in range(p, N):
            X.append(arr[t - p:t])
            y.append(arr[t])
        return np.array(X), np.array(y)

    def _optimize_svr_params(self, X_train, y_train):
        """
        Optimize [C, epsilon, gamma] for an SVR using HHO, with MAPE as the objective.
        We'll do a simple 80/20 train/val split inside this function.
        """
        split_idx = int(0.8 * len(X_train))
        if split_idx < 1:  # need at least 1 sample for sub-train
            raise ValueError("Not enough data to split subtrain/val for HHO parameter search.")

        X_subtrain, y_subtrain = X_train[:split_idx], y_train[:split_idx]
        X_val, y_val = X_train[split_idx:], y_train[split_idx:]

        # define objective
        def objective(params):
            # [C, epsilon, gamma]
            C_, eps_, gamma_ = params
            C_ = max(C_, 1e-6)
            eps_ = max(eps_, 1e-6)
            gamma_ = max(gamma_, 1e-6)

            svr = SVR(kernel='rbf', C=C_, epsilon=eps_, gamma=gamma_)
            svr.fit(X_subtrain, y_subtrain)
            preds = svr.predict(X_val)
            mape = mean_absolute_percentage_error(y_val, preds)
            return mape

        # bounds for [C, epsilon, gamma]
        lb = np.array([1.0, 0.001, 1e-4])  # lower bounds
        ub = np.array([1000.0, 0.1, 1.0])    # upper bounds

        best_params, best_fit = hho_optimize(
            objective_func=objective,
            dim=3,
            lb=lb,
            ub=ub,
            n_hawks=self.population,
            max_iter=self.iterations,
            random_seed=self.seed
        )
        C_best, eps_best, gamma_best = best_params
        # ensure positivity
        return {
            "C": float(max(C_best, 1e-6)),
            "epsilon": float(max(eps_best, 1e-6)),
            "gamma": float(max(gamma_best, 1e-6))
        }

    def _train_one_component(self, comp_array: np.ndarray):
        """
        Train a BaggingRegressor(SVR) on the decomposed component.
        """
        X, y = self._prepare_supervised(comp_array)
        if len(X) < 2:
            raise ValueError("Not enough data in component to train.")

        # optimize
        best_pars = self._optimize_svr_params(X, y)
        base_svr = SVR(kernel='rbf',
                       C=best_pars["C"],
                       epsilon=best_pars["epsilon"],
                       gamma=best_pars["gamma"])
        bagger = BaggingRegressor(estimator=base_svr,
                                  n_estimators=self.n_estimators,
                                  bootstrap=True,
                                  random_state=self.seed)
        bagger.fit(X, y)
        return bagger, best_pars

    def train(self, price_series: np.ndarray):
        """
        Full pipeline: CEEMDAN decompose => train bagging-svr on each component.
        """
        comps = self.decompose(price_series)
        self.component_series = comps
        self.component_models = []
        for c_arr in comps:
            model, params = self._train_one_component(c_arr)
            self.component_models.append(model)
            self.component_params.append(params)
        self._trained = True

    def predict(self, n_steps=1):
        """
        Iterative multi-step forecast. Return an array of length n_steps,
        each is the sum of the predicted next value from each decomposed component.
        """
        if not self._trained:
            raise RuntimeError("Model not trained yet.")

        predictions = []
        # For each component, store the last 'lag' data so we can move forward
        comp_histories = []
        for c_arr in self.component_series:
            comp_histories.append(list(c_arr[-self.lag:]))

        for step in range(n_steps):
            # predict next step for each component
            step_comp_preds = []
            for model, comp_vals in zip(self.component_models, comp_histories):
                x_input = np.array(comp_vals[-self.lag:]).reshape(1, -1)
                step_pred = model.predict(x_input)[0]
                step_comp_preds.append(step_pred)
            # sum across components => total price
            total_pred = sum(step_comp_preds)
            predictions.append(total_pred)
            # update each component's history
            for comp_vals, cpred in zip(comp_histories, step_comp_preds):
                comp_vals.append(cpred)

        return np.array(predictions)

    def predict_interval(self, n_steps=1, interval=0.95, n_simulations=1000):
        """
        Bootstrapped prediction intervals using random picks from base estimators
        across all components.
        :param n_steps: how many steps ahead to forecast
        :param interval: e.g. 0.95 for 95% interval
        :param n_simulations: how many Monte Carlo draws
        :return: (point_preds, lower_bounds, upper_bounds), each length n_steps
        """
        if not self._trained:
            raise RuntimeError("Model not trained yet.")
        alpha = 1 - interval
        q_lower = 100 * (alpha / 2)
        q_upper = 100 * (1 - alpha / 2)

        # shape (n_simulations, n_steps)
        sim_matrix = np.zeros((n_simulations, n_steps))

        # Original end-of-series for each component
        base_comp_hist = []
        for c_arr in self.component_series:
            base_comp_hist.append(list(c_arr[-self.lag:]))

        # Run simulations
        for sim_i in range(n_simulations):
            # copy the states
            sim_comp_hist = [h.copy() for h in base_comp_hist]

            for step in range(n_steps):
                step_pred_sum = 0.0
                for model, comp_vals in zip(self.component_models, sim_comp_hist):
                    # randomly pick a base estimator (an SVR)
                    base_est = np.random.choice(model.estimators_)
                    x_in = np.array(comp_vals[-self.lag:]).reshape(1, -1)
                    pred_val = base_est.predict(x_in)[0]
                    step_pred_sum += pred_val
                    # update
                    comp_vals.append(pred_val)
                sim_matrix[sim_i, step] = step_pred_sum

        # point forecast from the ensemble means
        point_preds = self.predict(n_steps=n_steps)

        lower_bounds = np.percentile(sim_matrix, q_lower, axis=0)
        upper_bounds = np.percentile(sim_matrix, q_upper, axis=0)

        return point_preds, lower_bounds, upper_bounds

def fetch_price_data_ibapi(
    contract_symbol="NG",
    exchange="NYMEX",
    currency="USD",
    endDateTime="",
    durationStr="1 Y",
    barSizeSetting="1 day",
    whatToShow="TRADES",
    useRTH=0
):
    return get_historical_data_from_ibapi(
        host='127.0.0.1',
        port=7496,
        client_id=8,
        contract_symbol=contract_symbol,
        exchange=exchange,
        currency=currency,
        endDateTime=endDateTime,
        durationStr=durationStr,
        barSizeSetting=barSizeSetting,
        whatToShow=whatToShow,
        useRTH=useRTH
    )

def main():
    # 1) Fetch data from IB
    print("Connecting to IB and fetching historical data...")
    df = get_historical_data_from_ibapi(
        contract_symbol="NG",
        exchange="NYMEX",
        currency="USD",
        endDateTime="",      # use current time
        durationStr="6 M",   # e.g. 6 months
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=0
    )
    df.sort_index(inplace=True)

    # 2) Summarize data
    print("\n✅ Historical Data Download Completed")
    print(f"Data retrieved. Rows: {len(df)}")
    print(f"Data range: {df.index[0]} --> {df.index[-1]}")

    # 3) Prepare the price series
    if 'Close' in df.columns:
        price_values = df['Close'].values
    elif 'close' in df.columns:
        price_values = df['close'].values
    else:
        raise KeyError("No 'Close' column found in data.")

    # 4) Initialize the model
    model = CEEMDANBaggingHHOSVR(
        n_estimators=15,
        population=20,
        iterations=30,
        lag=3,
        seed=42
    )

    # 5) Train the model
    print("\nTraining model...")
    model.train(price_values)

    # 6) Predict
    n_forecast_steps = 7
    point_preds = model.predict(n_steps=n_forecast_steps)

    # 7) Prediction interval
    pred_mean, pred_lower, pred_upper = model.predict_interval(
        n_steps=n_forecast_steps, interval=0.95, n_simulations=500
    )

    # 8) Print timestamp
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    nz_time = utc_now.astimezone(pytz.timezone("Pacific/Auckland"))
    print(f"\n📈 Prediction Completed: {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"📍 Local NZ Time: {nz_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    print("\n----------------------------------------")
    print(f"Forecast Horizon: {n_forecast_steps} Steps Ahead")
    print("----------------------------------------")
    print(" Date         |   Predicted Close ")
    print("--------------|---------------------")

    start_date = df.index[-1] + timedelta(days=1)
    for i in range(n_forecast_steps):
        forecast_date = start_date + timedelta(days=i)
        print(f" {forecast_date.date()} |   {point_preds[i]:.4f}")

    print("\n95% Prediction Intervals (Monte Carlo):")
    for i in range(n_forecast_steps):
        print(f"Day {i+1}: mean={pred_mean[i]:.4f}, "
              f"lower={pred_lower[i]:.4f}, upper={pred_upper[i]:.4f}")
    # 9) Log to history file
    log_path = "forecast_history_log.txt"
    with open(log_path, "a") as log_file:
        log_file.write(f"\n===== Forecast Run @ {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')} "
                       f"(Local: {nz_time.strftime('%Y-%m-%d %H:%M:%S %Z')}) =====\n")
        log_file.write(f"Data range: {df.index[0]} --> {df.index[-1]}\n")
        log_file.write(f"Forecast Horizon: {n_forecast_steps} Steps Ahead\n")
        log_file.write("Date       |  Predicted  |  Lower  |  Upper\n")
        log_file.write("-----------|-------------|---------|---------\n")
        for i in range(n_forecast_steps):
            forecast_date = start_date + timedelta(days=i)
            log_file.write(f"{forecast_date.date()} | {pred_mean[i]:9.4f} | "
                           f"{pred_lower[i]:7.4f} | {pred_upper[i]:7.4f}\n")
    
        # ✅ Hyperparameters go here — inside the same `with` block
        log_file.write("\nModel Hyperparameters per Component:\n")
        for idx, params in enumerate(model.component_params):
            log_file.write(f"Component {idx+1}: C={params['C']:.6f}, "
                           f"epsilon={params['epsilon']:.6f}, gamma={params['gamma']:.6f}\n")
    
        log_file.write("=====================================================\n")

if __name__ == "__main__":
    main()


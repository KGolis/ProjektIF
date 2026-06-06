from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RNG_SEED = 42

NOTIONAL_FX = 1_000_000          
K_FWD       = 4.4439             
NOTIONAL_IR = 10_000_000         
FIXED_RATE  = 0.0202             
T_YEARS     = 3.0
DT          = 1.0 / 12.0
N_STEPS     = int(round(T_YEARS / DT)) 
N_SIM       = 10_000
RECOVERY    = 0.40
LGD         = 1.0 - RECOVERY

def load_market_data(path: str = "Market_data.xlsx") -> dict:

    raw = pd.read_excel(path, sheet_name="Market data", header=None)

    cds_tickers = ["DB6MEUSM=R", "DB1YEUSM=R", "DB2YEUSM=R",
                   "DB3YEUSM=R", "DB4YEUSM=R", "DB5YEUSM=R"]
    cds_tenors  = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    cds_spreads = {}
    for r in range(3, 9):
        ticker = str(raw.iat[r, 1])
        val = float(raw.iat[r, 2])
        idx = cds_tickers.index(ticker)
        cds_spreads[cds_tenors[idx]] = val * 1e-4     

    S0 = float(raw.iat[10, 2])                        

    curve_start_row  = 4
    curve_end_row    = 65
    curve = raw.iloc[curve_start_row:curve_end_row, 5:11].copy()
    curve.columns = ["month", "EUR_Rate", "PLN_rate", "EUR_DF", "PLN_DF", "EUR_PLN_fwd"]
    curve = curve.astype(float).reset_index(drop=True)

    r_EUR = float(curve["EUR_Rate"].iloc[0])
    r_PLN = float(curve["PLN_rate"].iloc[0])

    return {
        "S0": S0,
        "r_EUR": r_EUR,
        "r_PLN": r_PLN,
        "curve": curve,
        "cds": cds_spreads,
    }


def calibrate_volatilities() -> dict:
    fx = pd.read_csv("eurpln_d.csv", parse_dates=["Data"])
    fx = fx[(fx["Data"] >= "2024-01-01") & (fx["Data"] <= "2025-12-31")] \
            .sort_values("Data").reset_index(drop=True)
    fx_log = np.log(fx["Zamkniecie"] / fx["Zamkniecie"].shift(1)).dropna()
    sigma_fx_annual = float(fx_log.std(ddof=1) * np.sqrt(252))

    ir = pd.read_csv("ECB Data Portal_20260417181527.csv")
    ir.columns = ["date", "time_period", "estr"]
    ir["date"] = pd.to_datetime(ir["date"])
    ir = ir[(ir["date"] >= "2024-01-01") & (ir["date"] <= "2025-12-31")] \
            .sort_values("date").reset_index(drop=True)
    ir["estr"] = ir["estr"].astype(float) / 100.0       # % -> dziesietnie
    delta_r = ir["estr"].diff().dropna()
    sigma_ir_annual = float(delta_r.std(ddof=1) * np.sqrt(252))

    fx_innov = pd.DataFrame({"date": fx["Data"].values[1:], "dlnS": fx_log.values})
    ir_innov = pd.DataFrame({"date": ir["date"].values[1:], "dr": delta_r.values})
    merged = pd.merge(fx_innov, ir_innov, on="date", how="inner")
    rho = float(np.corrcoef(merged["dlnS"], merged["dr"])[0, 1])

    return {
        "sigma_fx": sigma_fx_annual,
        "sigma_ir": sigma_ir_annual,
        "rho": rho,
        "n_obs": len(merged),
    }

def simulate_joint(S0: float, r0: float,
                   sigma_fx: float, sigma_ir: float, rho: float,
                   mu_fx_rn: float,
                   n_sim: int = N_SIM, n_steps: int = N_STEPS, dt: float = DT,
                   seed: int = RNG_SEED) -> tuple[np.ndarray, np.ndarray]:

    rho = float(np.clip(rho, -0.999, 0.999))
    L = np.linalg.cholesky([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(seed)

    fx = np.empty((n_steps + 1, n_sim));  fx[0] = S0
    ir = np.empty((n_steps + 1, n_sim));  ir[0] = r0
    for i in range(1, n_steps + 1):
        Z = L @ rng.standard_normal((2, n_sim))
        fx[i] = fx[i-1] * np.exp((mu_fx_rn - 0.5 * sigma_fx**2) * dt
                                 + sigma_fx * np.sqrt(dt) * Z[0])
        ir[i] = ir[i-1] + sigma_ir * np.sqrt(dt) * Z[1]      # ABM, mu=0 (RN)
    return fx, ir

def price_fx_forward(fx_paths: np.ndarray, r_EUR: float, r_PLN: float,
                     time_grid: np.ndarray) -> np.ndarray:

    T_fwd = T_YEARS
    mtm = np.zeros_like(fx_paths)
    for i, t in enumerate(time_grid):
        tau = T_fwd - t
        if tau < 0:
            continue
        if tau == 0:
            mtm[i] = NOTIONAL_FX * (fx_paths[i] - K_FWD)
            continue
        F_t = fx_paths[i] * np.exp((r_PLN - r_EUR) * tau)
        df_pln = np.exp(-r_PLN * tau)
        mtm[i] = NOTIONAL_FX * (F_t - K_FWD) * df_pln
    return mtm


def price_irs_receiver(ir_paths: np.ndarray, r_EUR: float,
                       time_grid: np.ndarray, dt: float = DT) -> np.ndarray:
    fixed_dates = np.array([1.0, 2.0, 3.0])
    reset_dates = np.array([k * 0.25 for k in range(13)])
    T_M = 3.0
    tau_fixed = 1.0
    tau_float = 0.25
    grid_idx = lambda t: int(round(t / dt))

    n_steps_plus, n_sim = ir_paths.shape
    mtm = np.zeros_like(ir_paths)
    for i, t_now in enumerate(time_grid):
        if t_now >= T_M:
            mtm[i] = 0.0
            continue
        if t_now == T_M:
            pv_fixed = FIXED_RATE * NOTIONAL_IR * tau_fixed
            k = int(np.searchsorted(reset_dates, t_now, side='right')) - 1
            T_prev = reset_dates[k]
            r_last = ir_paths[grid_idx(T_prev)]
            pv_float = NOTIONAL_IR * tau_float * r_last
            mtm[i] = pv_fixed - pv_float
            continue
        # --- Fixed leg (otrzymujemy) ---
        pv_fixed = np.zeros(n_sim)
        for T_n in fixed_dates:
            if T_n > t_now:
                pv_fixed += FIXED_RATE * NOTIONAL_IR * tau_fixed * \
                            np.exp(-r_EUR * (T_n - t_now))

        k = int(np.searchsorted(reset_dates, t_now, side='right')) - 1
        T_prev = reset_dates[k]           
        T_next = reset_dates[k + 1]       
        r_fixed_in_flight = ir_paths[grid_idx(T_prev)]

        df_next = np.exp(-r_EUR * (T_next - t_now))
        df_M    = np.exp(-r_EUR * (T_M - t_now))
        pv_float = NOTIONAL_IR * df_next * (1.0 + tau_float * r_fixed_in_flight) \
                   - NOTIONAL_IR * df_M

        mtm[i] = pv_fixed - pv_float
    return mtm


def build_survival_curve(cds_spreads: dict, df_func, time_grid: np.ndarray,
                         recovery: float = RECOVERY) -> np.ndarray:
    
    from scipy.interpolate import interp1d

    lgd = 1.0 - recovery

    tenor_m = [0.0] + [t * 12.0 for t in sorted(cds_spreads.keys())]
    spread_v = [0.0] + [cds_spreads[t] for t in sorted(cds_spreads.keys())]
    max_spread = max(spread_v)
    spread_interp = interp1d(
        tenor_m, spread_v, kind="linear",
        bounds_error=False, fill_value=(0.0, max_spread),
    )
    grid_m = time_grid * 12.0
    spreads = spread_interp(grid_m)

    n_pts = len(time_grid)
    PS = np.ones(n_pts)
    df = np.array([df_func(t) for t in time_grid])
    for n in range(1, n_pts):
        dt = time_grid[n] - time_grid[n-1]
        S_n = spreads[n]

        acc = 0.0
        for i in range(1, n):
            acc += df[i] * (lgd * (PS[i-1] - PS[i]) - S_n * dt * PS[i])
        numerator = lgd * df[n] * PS[n-1] + acc
        denominator = df[n] * (lgd + S_n * dt)
        PS[n] = numerator / denominator
    return PS

def compute_cva(ee: np.ndarray, time_grid: np.ndarray, df_curve,
                PS: np.ndarray, recovery: float = RECOVERY) -> float:
    """CVA = LGD * sum_i EE(t_i) * DF(0, t_i) * q(t_{i-1}, t_i).

    PS to tablica indeksowana po kroku siatki (z build_survival_curve)."""
    lgd = 1.0 - recovery
    total = 0.0
    for i in range(1, len(time_grid)):
        q = PS[i-1] - PS[i]
        df = df_curve(time_grid[i])
        total += ee[i] * df * q
    return lgd * total


def compute_cva_step(ee: np.ndarray, time_grid: np.ndarray, df_curve,
                     PS: np.ndarray, recovery: float = RECOVERY) -> np.ndarray:
    """CVA per-step: step[i] = LGD * EE(t_i) * DF(0,t_i) * q(t_{i-1},t_i),
    bez sumowania. step[0] = 0."""
    lgd = 1.0 - recovery
    step = np.zeros(len(time_grid))
    for i in range(1, len(time_grid)):
        q = PS[i-1] - PS[i]
        df = df_curve(time_grid[i])
        step[i] = lgd * ee[i] * df * q
    return step
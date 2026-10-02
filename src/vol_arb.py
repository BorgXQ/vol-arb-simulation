import numpy as np
import pandas as pd
from src.utils import implied_volatility_bs
from src.calc import (
    DEFAULT_FFT_N,
    DEFAULT_FFT_ALPHA,
    DEFAULT_FFT_ETA,
    validate_market_fft_config,
    interpolate_call_prices,
    put_from_call_parity,
    CM99_call_price_grid_fft,
    CM99_calibration_market
)


# ==============================================================================
# STATE DATAFRAME INITIALIZATION
# ==============================================================================

def make_option_id(option_type, strike):
    """
    Build a stable string identifier for an option contract.

    Parameters
    ----------
    option_type : str
        'C' for call, 'P' for put.
    strike : float or int
        Strike price.

    Returns
    -------
    str
        E.g. 'C_132' or 'P_168'.
    """
    strike_val = float(strike)
    if strike_val.is_integer():
        strike_str = str(int(strike_val))
    else:
        strike_str = str(strike_val)
    return f"{option_type}_{strike_str}"


def add_option_id_column(options_df):
    """
    Append a stable Option_ID column derived only from Type and Strike.

    Parameters
    ----------
    options_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Copy with an added 'Option_ID' column.
    """
    df = options_df.copy()
    df["Option_ID"] = [
        make_option_id(opt_type, strike)
        for opt_type, strike in zip(df["Type"], df["Strike"])
    ]
    return df


def initialize_strategy_state_df(options_df):
    """
    Allocate the strategy state DataFrame with all required columns.

    Static columns capture per-timestep scalars (price, model params, PnL).
    Dynamic columns are generated per option ID (weights, prices, IVs, Greeks).

    Parameters
    ----------
    options_df : pd.DataFrame
        Full options dataset; must contain 't_index', 'Type', and 'Strike'.

    Returns
    -------
    pd.DataFrame
        Zero/NaN-initialised state DataFrame indexed by time index.
    """
    df = add_option_id_column(options_df)

    time_indices = sorted(df["t_index"].unique())
    option_ids = sorted(df["Option_ID"].unique())

    static_cols = [
        "t_index", "S_t", "T",
        "target_option_id", "target_position",
        "kappa_trader", "theta_trader", "xi_trader", "rho_trader", "v0_trader",
        "net_delta", "net_gamma", "net_vega", "net_variance_sensitivity",
        "w_underlying",
        "pnl_incremental", "pnl_cumulative",
    ]

    dynamic_cols = []
    for oid in option_ids:
        dynamic_cols.extend([
            f"w_{oid}",
            f"mkt_price_{oid}",
            f"theo_price_{oid}",
            f"mkt_iv_{oid}",
            f"theo_iv_{oid}",
            f"iv_diff_{oid}",
            f"delta_{oid}",
            f"gamma_{oid}",
            f"vega_{oid}",
        ])

    state_df = pd.DataFrame(index=time_indices, columns=static_cols + dynamic_cols, dtype=object)
    state_df["t_index"] = time_indices

    # Numeric defaults
    for col in [
        "S_t", "T",
        "kappa_trader", "theta_trader", "xi_trader", "rho_trader", "v0_trader",
        "net_delta", "net_gamma", "net_vega", "net_variance_sensitivity",
        "w_underlying", "pnl_incremental", "pnl_cumulative"
    ]:
        state_df[col] = 0.0

    # String defaults
    state_df["target_option_id"] = None
    state_df["target_position"] = 0.0

    # Dynamic defaults
    for col in dynamic_cols:
        if col.startswith("w_"):
            state_df[col] = 0.0
        else:
            state_df[col] = np.nan

    return state_df


# ==============================================================================
# UNIVERSE SELECTION
# ==============================================================================

def select_otm_universe(options_slice, n_each_side=3):
    """
    Select a symmetric OTM universe around spot.

    Picks the nearest `n_each_side` OTM puts (strikes below spot) and
    OTM calls (strikes above spot).

    Parameters
    ----------
    options_slice : pd.DataFrame
        Single-timestep options data.
    n_each_side : int, optional
        Number of strikes to include on each side (default 3).

    Returns
    -------
    pd.DataFrame
        Combined OTM universe with Option_ID column added.
    """
    S_t = float(options_slice["S_t"].iloc[0])

    puts = (
        options_slice[(options_slice["Type"] == "P") & (options_slice["Strike"] < S_t)]
        .sort_values("Strike", ascending=False)
        .head(n_each_side)
    )

    calls = (
        options_slice[(options_slice["Type"] == "C") & (options_slice["Strike"] > S_t)]
        .sort_values("Strike", ascending=True)
        .head(n_each_side)
    )

    universe = pd.concat([puts, calls], ignore_index=True).copy()
    universe = add_option_id_column(universe)
    return universe


def ensure_target_in_universe(full_slice, universe_df, target_option_id):
    """
    Add the target contract back to the universe if it was excluded.

    Parameters
    ----------
    full_slice : pd.DataFrame
        Full options data for the current timestep.
    universe_df : pd.DataFrame
        Current tradable universe.
    target_option_id : str or None
        ID of the target contract to preserve.

    Returns
    -------
    pd.DataFrame
        Universe with the target contract included (if found in full_slice).
    """
    if target_option_id is None:
        return universe_df

    if target_option_id in set(universe_df["Option_ID"]):
        return universe_df

    target_row = full_slice[full_slice["Option_ID"] == target_option_id]
    if target_row.empty:
        return universe_df

    combined = pd.concat([universe_df, target_row], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Option_ID"]).reset_index(drop=True)
    return combined


# ==============================================================================
# THEORETICAL PRICING AND GREEKS
# ==============================================================================

def price_slice_with_heston_and_greeks(
    options_slice,
    S0,
    params,
    N=DEFAULT_FFT_N,
    alpha=DEFAULT_FFT_ALPHA,
    eta=DEFAULT_FFT_ETA,
    eps_S_rel=0.01,
    eps_v_rel=0.05,
):
    """Price options and validate Richardson Greeks against smaller bumps and 2N.

    VarianceSensitivity is dV/dv0, with other Heston parameters fixed.
    Vega is a compatibility alias, not Black–Scholes volatility vega.
    At the variance boundary use second-order forward differences.
    """
    validate_market_fft_config(options_slice, N, alpha, eta)
    kappa_v, theta_v, xi_v, rho, v0 = map(float, params)

    df = options_slice.copy()
    df = add_option_id_column(df)

    if not (np.isfinite(S0) and S0 > 0 and 0 < eps_S_rel < 0.5
            and np.isfinite(eps_v_rel) and eps_v_rel > 0):
        raise ValueError("Spot and Greek bumps must be finite and positive; spot bump < 0.5.")
    eps_S = eps_S_rel * S0
    eps_v = max(1e-5, eps_v_rel * max(v0, 1e-4))

    out_frames = []

    for (T, r), grp in df.groupby(["T", "r"], sort=False):
        T = float(T)
        r = float(r)
        strikes = grp["Strike"].to_numpy(dtype=float)
        types = grp["Type"].to_numpy()

        cache = {}

        def model_prices_for(S_bump, v0_bump, grid_N=N):
            key = (S_bump, v0_bump, grid_N)
            if key in cache:
                return cache[key]
            K_grid, call_grid = CM99_call_price_grid_fft(
                S0=S_bump,
                T=T,
                r=r,
                kappa_v=kappa_v,
                theta_v=theta_v,
                xi_v=xi_v,
                rho=rho,
                v0=v0_bump,
                N=grid_N,
                alpha=alpha,
                eta=eta,
            )
            call_vals = interpolate_call_prices(strikes, K_grid, call_grid)
            put_vals = put_from_call_parity(call_vals, S_bump, strikes, r, T)
            cache[key] = np.where(types == "C", call_vals, put_vals)
            return cache[key]

        def estimates(hs, hv, grid_N):
            f = lambda spot, variance: model_prices_for(spot, variance, grid_N)
            p = f(S0, v0)
            up, dn = f(S0 + hs, v0), f(S0 - hs, v0)
            delta = (up - dn) / (2 * hs)
            gamma = (up - 2 * p + dn) / hs**2
            if v0 >= eps_v:
                variance = (f(S0, v0 + hv) - f(S0, v0 - hv)) / (2 * hv)
            else:
                variance = (-3 * p + 4 * f(S0, v0 + hv) - f(S0, v0 + 2 * hv)) / (2 * hv)
            return np.array([delta, gamma, variance])

        def richardson(hs, hv, grid_N):
            # Keep the same variance stencil at both scales near the boundary.
            coarse = estimates(hs, hv, grid_N)
            fine = estimates(hs / 2, hv / 2, grid_N)
            return (4 * fine - coarse) / 3

        price_0 = model_prices_for(S0, v0)
        base = richardson(eps_S, eps_v, N)
        refined = richardson(eps_S / 2, eps_v / 2, N)
        reference = richardson(eps_S / 2, eps_v / 2, 2 * N)
        tolerance = np.array([2e-5, 2e-5, 2e-3])[:, None] + 0.005 * np.abs(reference)
        errors = np.maximum(np.abs(base - refined), np.abs(refined - reference))
        if not np.all(np.isfinite(reference)) or not np.all(errors <= tolerance):
            raise RuntimeError(
                "Heston Greeks did not converge across bump sizes and FFT resolution "
                f"(T={T:.6g}, N={N}); increase pricing resolution or review fitted parameters."
            )
        delta, gamma, vega = refined

        grp_out = grp.copy()
        grp_out["Theo_Price"] = price_0
        grp_out["Delta"] = delta
        grp_out["Gamma"] = gamma
        grp_out["VarianceSensitivity"] = vega
        grp_out["Vega"] = vega
        grp_out["Greek_Error_Ratio"] = np.max(errors / tolerance, axis=0)

        grp_out["Theo_IV"] = [
            implied_volatility_bs(
                price=p,
                S=S0,
                K=K,
                T=T,
                r=r,
                option_type=opt_type,
            )
            for p, K, opt_type in zip(price_0, strikes, types)
        ]

        out_frames.append(grp_out)

    priced_df = pd.concat(out_frames, ignore_index=True)
    priced_df["IV_Diff"] = priced_df["Theo_IV"] - priced_df["Market_IV"]
    priced_df["Abs_IV_Diff"] = priced_df["IV_Diff"].abs()

    return priced_df


# ==============================================================================
# TARGET SELECTION
# ==============================================================================

def select_target_contract(priced_universe_df):
    """
    Select the contract with the largest absolute IV gap as the target.

    Parameters
    ----------
    priced_universe_df : pd.DataFrame
        Must contain Theo_IV, Market_IV, Abs_IV_Diff, and Option_ID columns.

    Returns
    -------
    target_option_id : str
    target_position : float
        +1 if option is cheap (buy), -1 if expensive (sell).
    target_row : pd.Series
    """
    ranked = priced_universe_df.dropna(subset=["Theo_IV", "Market_IV"]).copy()
    if ranked.empty:
        raise ValueError("No valid IV comparison available for target selection.")

    target_row = ranked.loc[ranked["Abs_IV_Diff"].idxmax()].copy()
    target_option_id = target_row["Option_ID"]

    # Buy if theoretical IV > market IV (option is cheap); sell otherwise
    target_position = 1.0 if target_row["Theo_IV"] > target_row["Market_IV"] else -1.0

    return target_option_id, target_position, target_row


# ==============================================================================
# HEDGING LOGIC
# ==============================================================================

HEDGE_MODES = ("gamma_delta_variance", "gamma_delta")


def solve_option_hedge(priced_universe_df, target_option_id, target_position,
                       hedge_mode="gamma_delta_variance"):
    """Return weights, stock hedge and diagnostics; reject unstable/unhedged risk.

    Row scaling removes the units from the SVD. Singular values below 1e-3
    of the largest are discarded. Compatible rank-deficient systems are allowed.
    Policy limits: each hedge option <= 10, gross hedge options <= 20 per target.
    """
    if hedge_mode not in HEDGE_MODES:
        raise ValueError(f"hedge_mode must be one of {HEDGE_MODES}.")
    df = priced_universe_df.copy()
    if "VarianceSensitivity" not in df:
        df["VarianceSensitivity"] = df["Vega"]
    if df["Option_ID"].duplicated().any():
        raise ValueError("Hedge universe must contain unique option IDs.")
    if not np.isfinite(target_position) or target_position == 0:
        raise ValueError("Target position must be finite and nonzero.")
    if not np.isfinite(df[["Delta", "Gamma", "VarianceSensitivity"]].to_numpy(float)).all():
        raise ValueError("Hedge Greeks must be finite.")
    target = df.loc[df["Option_ID"] == target_option_id]
    if len(target) != 1:
        raise ValueError(f"Target {target_option_id} not found in priced universe.")
    target = target.iloc[0]
    hedgers = df.loc[df["Option_ID"] != target_option_id]
    risks = ["Gamma"] + (["VarianceSensitivity"] if hedge_mode == "gamma_delta_variance" else [])
    A = hedgers[risks].to_numpy(float).T
    b = -target_position * target[risks].to_numpy(float)
    scale = np.maximum(np.linalg.norm(A, axis=1), np.abs(b) / abs(target_position))
    scale = np.where(scale > 0, scale, 1.0)
    normalized = A / scale[:, None]
    rhs = b / scale
    hedge_w, _, rank, singular = np.linalg.lstsq(normalized, rhs, rcond=1e-3)
    residual = normalized @ hedge_w - rhs
    if np.max(np.abs(residual)) > 1e-6 * abs(target_position):
        raise RuntimeError("Cannot neutralize requested Greeks with a stable hedge (rank deficient or ill-conditioned universe).")
    gross = float(np.sum(np.abs(hedge_w))) / abs(target_position)
    if gross > 20 or np.any(np.abs(hedge_w) > 10 * abs(target_position)):
        raise RuntimeError("Hedge exceeds position limits: 10 per option or 20 gross per unit target.")
    weights = {target_option_id: float(target_position)}
    weights.update(zip(hedgers["Option_ID"], map(float, hedge_w)))
    stock = -(target_position * float(target["Delta"]) + hedge_w @ hedgers["Delta"].to_numpy(float))
    net = target_position * target[["Delta", "Gamma", "VarianceSensitivity"]].to_numpy(float)
    net += hedge_w @ hedgers[["Delta", "Gamma", "VarianceSensitivity"]].to_numpy(float)
    net[0] += stock
    condition = float(singular[0] / singular[-1]) if len(singular) == len(risks) and singular[-1] > 0 else np.inf
    diagnostics = dict(hedge_rank=int(rank), hedge_condition=condition,
                       hedge_gross_options=gross, hedge_residual=float(np.max(np.abs(residual))),
                       net_variance_sensitivity=float(net[2]))
    return weights, float(stock), diagnostics


def solve_gamma_vega_delta_hedge(priced_universe_df, target_option_id, target_position):
    """Compatibility wrapper for gamma–delta–variance hedging (Vega = dV/dv0)."""
    weights, stock, _ = solve_option_hedge(priced_universe_df, target_option_id, target_position)
    return weights, stock


# ==============================================================================
# STATE UPDATES
# ==============================================================================

def write_static_state(state_df, t_idx, options_slice):
    """Write spot price and time-to-maturity for the current timestep."""
    row = options_slice.iloc[0]
    state_df.loc[t_idx, "S_t"] = float(row["S_t"])
    state_df.loc[t_idx, "T"] = float(row["T"])


def write_model_params(state_df, t_idx, params):
    """Write calibrated Heston parameters for the current timestep."""
    kappa_v, theta_v, xi_v, rho, v0 = map(float, params)
    state_df.loc[t_idx, "kappa_trader"] = kappa_v
    state_df.loc[t_idx, "theta_trader"] = theta_v
    state_df.loc[t_idx, "xi_trader"] = xi_v
    state_df.loc[t_idx, "rho_trader"] = rho
    state_df.loc[t_idx, "v0_trader"] = v0


def zero_current_positions(state_df, t_idx):
    """Reset all option weights and net Greeks to zero for the current timestep."""
    for col in state_df.columns:
        if col.startswith("w_"):
            state_df.loc[t_idx, col] = 0.0
    state_df.loc[t_idx, "w_underlying"] = 0.0
    state_df.loc[t_idx, "net_delta"] = 0.0
    state_df.loc[t_idx, "net_gamma"] = 0.0
    state_df.loc[t_idx, "net_vega"] = 0.0
    state_df.loc[t_idx, "net_variance_sensitivity"] = 0.0


def write_row_metrics(state_df, t_idx, full_slice, priced_universe_df, option_weights, w_underlying,
                      target_option_id, target_position):
    """
    Populate all metrics for the current timestep in the state DataFrame.

    Fills:
    - Market prices and market IV for all options in the full slice.
    - Theoretical prices, IV, and Greeks for the priced universe.
    - Option and underlying weights.
    - Net portfolio Greeks.

    Parameters
    ----------
    state_df : pd.DataFrame
    t_idx : int
    full_slice : pd.DataFrame
    priced_universe_df : pd.DataFrame
    option_weights : dict
    w_underlying : float
    target_option_id : str
    target_position : float
    """
    zero_current_positions(state_df, t_idx)

    # Market values for all options at this timestep
    for _, row in full_slice.iterrows():
        oid = row["Option_ID"]
        state_df.loc[t_idx, f"mkt_price_{oid}"] = float(row["Market_Price"])
        state_df.loc[t_idx, f"mkt_iv_{oid}"] = float(row["Market_IV"]) if pd.notna(row["Market_IV"]) else np.nan

    # Theoretical values and Greeks for the priced universe
    for _, row in priced_universe_df.iterrows():
        oid = row["Option_ID"]
        state_df.loc[t_idx, f"theo_price_{oid}"] = float(row["Theo_Price"])
        state_df.loc[t_idx, f"theo_iv_{oid}"] = float(row["Theo_IV"]) if pd.notna(row["Theo_IV"]) else np.nan
        state_df.loc[t_idx, f"iv_diff_{oid}"] = float(row["IV_Diff"]) if pd.notna(row["IV_Diff"]) else np.nan
        state_df.loc[t_idx, f"delta_{oid}"] = float(row["Delta"])
        state_df.loc[t_idx, f"gamma_{oid}"] = float(row["Gamma"])
        state_df.loc[t_idx, f"vega_{oid}"] = float(row["Vega"])

    # Weights
    for oid, w in option_weights.items():
        state_df.loc[t_idx, f"w_{oid}"] = float(w)
    state_df.loc[t_idx, "w_underlying"] = float(w_underlying)

    # Net Greeks
    net_delta = 0.0
    net_gamma = 0.0
    net_vega = 0.0

    for oid, w in option_weights.items():
        row = priced_universe_df[priced_universe_df["Option_ID"] == oid]
        if row.empty:
            continue
        row = row.iloc[0]
        net_delta += float(w) * float(row["Delta"])
        net_gamma += float(w) * float(row["Gamma"])
        net_vega += float(w) * float(row["Vega"])

    net_delta += float(w_underlying)

    state_df.loc[t_idx, "net_delta"] = net_delta
    state_df.loc[t_idx, "net_gamma"] = net_gamma
    state_df.loc[t_idx, "net_vega"] = net_vega
    state_df.loc[t_idx, "net_variance_sensitivity"] = net_vega

    state_df.loc[t_idx, "target_option_id"] = target_option_id
    state_df.loc[t_idx, "target_position"] = float(target_position)


def compute_initial_gross_exposure(priced_universe_df, option_weights, w_underlying, S_t):
    """
    Compute gross exposure at strategy inception.

    Defined as: sum_i |w_i| * market_price_i + |w_underlying| * S_t

    Parameters
    ----------
    priced_universe_df : pd.DataFrame
    option_weights : dict
    w_underlying : float
    S_t : float

    Returns
    -------
    float
    """
    price_map = priced_universe_df.set_index("Option_ID")["Market_Price"].to_dict()

    gross_options = 0.0
    for oid, w in option_weights.items():
        if oid in price_map:
            gross_options += abs(float(w)) * float(price_map[oid])

    gross_underlying = abs(float(w_underlying)) * float(S_t)
    return gross_options + gross_underlying


ACCOUNT_COLUMNS = (
    "cash_balance", "holdings_value", "equity", "trade_cashflow",
    "financing_incremental", "financing_cumulative", "trading_pnl_incremental",
)


def settle_cash_account(state_df, options_curr, t_idx, options_prev=None, prev_t_idx=None):
    """Settle current weights at current quotes, after accruing prior cash.

    Zero initial capital; no external deposits, transaction costs or dividends.
    Both borrowing and lending use the previous slice's continuously compounded
    r over the decrease in T (the simulation's year convention). Entry and
    rebalancing exchange cash for holdings without creating equity. After exit,
    the account is frozen; proceeds are no longer invested by this strategy.
    """
    def positions(index):
        if index is None:
            return {}, 0.0
        weights = {col[2:]: float(state_df.loc[index, col]) for col in state_df.columns
                   if col.startswith("w_") and col != "w_underlying"}
        stock = float(state_df.loc[index, "w_underlying"])
        if not np.isfinite([*weights.values(), stock]).all():
            raise ValueError("Accounting requires finite positions.")
        return {oid: w for oid, w in weights.items() if w != 0}, stock

    def value(weights, stock, quotes, spot):
        prices = quotes.set_index("Option_ID")["Market_Price"]
        if prices.index.has_duplicates:
            raise ValueError("Accounting requires unique option quotes per timestep.")
        missing = set(weights) - set(prices.index)
        if missing:
            raise ValueError(f"Missing market quotes for held contracts: {sorted(missing)}")
        marks = prices.reindex(list(weights)).to_numpy(float)
        if not np.isfinite(marks).all() or not np.isfinite(spot) or spot <= 0:
            raise ValueError("Accounting requires finite marks and a positive spot.")
        return float(np.dot(list(weights.values()), marks) + stock * spot)

    old_weights, old_stock = positions(prev_t_idx)
    new_weights, new_stock = positions(t_idx)
    spot = float(state_df.loc[t_idx, "S_t"])
    holdings = value(new_weights, new_stock, options_curr, spot)
    old_marked = value(old_weights, old_stock, options_curr, spot)
    previous_cash = previous_equity = previous_holdings = interest = financing_total = 0.0
    if prev_t_idx is not None:
        previous_cash = float(state_df.loc[prev_t_idx, "cash_balance"])
        previous_equity = float(state_df.loc[prev_t_idx, "equity"])
        financing_total = float(state_df.loc[prev_t_idx, "financing_cumulative"])
        previous_holdings = value(old_weights, old_stock, options_prev,
                                  float(state_df.loc[prev_t_idx, "S_t"]))
        elapsed = float(state_df.loc[prev_t_idx, "T"] - state_df.loc[t_idx, "T"])
        if not np.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Accounting requires strictly decreasing maturities.")
        if state_df.loc[prev_t_idx, "account_status"] == "active":
            rates = options_prev["r"].to_numpy(float)
            if not np.isfinite(rates).all() or not np.all(rates == rates[0]):
                raise ValueError("Accounting requires a single finite funding rate per timestep.")
            interest = previous_cash * np.expm1(float(rates[0]) * elapsed)

    # Selling old holdings and buying new ones is equivalent to trading changes.
    trade_cashflow = old_marked - holdings
    cash = previous_cash + interest + trade_cashflow
    equity = cash + holdings
    trading_pnl = old_marked - previous_holdings
    if not np.isfinite([cash, equity, interest, trading_pnl]).all():
        raise ValueError("Nonfinite cash-account result.")
    values = dict(cash_balance=cash, holdings_value=holdings, equity=equity,
                  trade_cashflow=trade_cashflow, financing_incremental=interest,
                  financing_cumulative=financing_total + interest,
                  trading_pnl_incremental=trading_pnl,
                  pnl_incremental=equity - previous_equity, pnl_cumulative=equity)
    for name, amount in values.items():
        state_df.loc[t_idx, name] = amount


# ==============================================================================
# MAIN STRATEGY LOOP
# ==============================================================================

def run_vol_arb_strategy(
    options_market_df,
    calibration_N=None,
    pricing_N=DEFAULT_FFT_N,
    alpha=DEFAULT_FFT_ALPHA,
    eta=DEFAULT_FFT_ETA,
    n_each_side=3,
    dt=1/252,
    exit_days_before_expiry=10,
    hedge_mode="gamma_delta_variance",
):
    """
    Run the full volatility arbitrage strategy over all timesteps.

    Strategy rules
    --------------
    - t=0 : calibrate model, price universe, select target once, hedge.
    - t>=1 : target is fixed; universe is rebuilt each period; recalibrate,
             reprice, and rehedge.
    - Cash starts at zero; trades are self-financing and cash accrues at prior r.
    - Exit: liquidate at current marks when T <= exit_days_before_expiry * dt;
            include that holding period's P&L and financing, then freeze cash.

    Parameters
    ----------
    options_market_df : pd.DataFrame
        Must contain Market_IV and all standard option columns.
    calibration_N : int or None, optional
        Legacy alias: if provided, must equal pricing_N. By default calibration
        uses pricing_N, so fitting and valuation cannot silently diverge.
    pricing_N : int, optional
        Shared FFT grid size for calibration and valuation (default 16384).
    alpha : float, optional
        Carr-Madan damping parameter (default 1.5).
    eta : float, optional
        Frequency grid spacing (default 0.25).
    n_each_side : int, optional
        Number of OTM strikes per side in the tradable universe (default 3).
    dt : float, optional
        Time step in years (default 1/252).
    exit_days_before_expiry : int, optional
        Days before expiry at which positions are zeroed (default 10).
    hedge_mode : str, optional
        "gamma_delta_variance" (default) or "gamma_delta". The latter leaves
        variance sensitivity unconstrained. Unstable hedges raise RuntimeError.

    Returns
    -------
    state_df : pd.DataFrame
        Full strategy state across all timesteps.
    initial_gross_exposure : float
        Gross notional exposure at inception.
    """
    if hedge_mode not in HEDGE_MODES:
        raise ValueError(f"hedge_mode must be one of {HEDGE_MODES}.")
    if calibration_N is not None and calibration_N != pricing_N:
        raise ValueError("calibration_N must equal pricing_N; use one FFT configuration.")
    validate_market_fft_config(options_market_df, pricing_N, alpha, eta)

    df = add_option_id_column(options_market_df).copy()

    if "Market_IV" not in df.columns:
        raise ValueError("options_market_df must already contain Market_IV.")

    time_indices = sorted(df["t_index"].unique())
    state_df = initialize_strategy_state_df(df).copy()
    state_df["hedge_mode"] = hedge_mode
    state_df["account_status"] = "pending"
    for name in ACCOUNT_COLUMNS:
        state_df[name] = 0.0
    for name in ("hedge_rank", "hedge_condition", "hedge_gross_options", "hedge_residual", "greek_error_ratio"):
        state_df[name] = np.nan

    target_option_id = None
    target_position = None
    initial_gross_exposure = None

    for i, t_idx in enumerate(time_indices):
        full_slice = df[df["t_index"] == t_idx].copy().reset_index(drop=True)
        write_static_state(state_df, t_idx, full_slice)

        prev_t = time_indices[i - 1] if i > 0 else None
        prev_slice = df[df["t_index"] == prev_t].copy() if i > 0 else None
        # Record current marks on exit as well as on active trading days.
        for _, quote in full_slice.iterrows():
            oid = quote["Option_ID"]
            state_df.loc[t_idx, f"mkt_price_{oid}"] = float(quote["Market_Price"])
            state_df.loc[t_idx, f"mkt_iv_{oid}"] = float(quote["Market_IV"])

        # Close at current marks, including the final holding-period P&L and interest.
        T_t = float(full_slice["T"].iloc[0])
        if T_t <= exit_days_before_expiry * dt:
            zero_current_positions(state_df, t_idx)
            state_df.loc[t_idx, "target_option_id"] = target_option_id
            state_df.loc[t_idx, "target_position"] = 0.0
            was_active = prev_t is not None and state_df.loc[prev_t, "account_status"] == "active"
            state_df.loc[t_idx, "account_status"] = "exited" if was_active else "closed"
            settle_cash_account(state_df, full_slice, t_idx, prev_slice, prev_t)
            continue

        state_df.loc[t_idx, "account_status"] = "active"

        # Build current tradable universe
        universe_df = select_otm_universe(full_slice, n_each_side=n_each_side)
        universe_df = ensure_target_in_universe(full_slice, universe_df, target_option_id)

        S_t = float(full_slice["S_t"].iloc[0])

        # 1) Calibration
        try:
            trader_params, _, _, _ = CM99_calibration_market(
                universe_df,
                S0=S_t,
                N=pricing_N,
                alpha=alpha,
                eta=eta,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Calibration at time index {t_idx} "
                f"({T_t / dt:.1f} trading days to expiry): {exc}"
            ) from exc
        write_model_params(state_df, t_idx, trader_params)

        # 2) Theoretical pricing, Greeks, and implied volatility
        try:
            priced_universe_df = price_slice_with_heston_and_greeks(
                universe_df,
                S0=S_t,
                params=trader_params,
                N=pricing_N,
                alpha=alpha,
                eta=eta,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Greeks at time index {t_idx}: {exc}") from exc

        # 3) Target selection — performed only once at inception
        if target_option_id is None:
            target_option_id, target_position, _ = select_target_contract(priced_universe_df)

        # 4) Explicit risk constraints, with numerical stability and size checks.
        try:
            option_weights, w_underlying, diagnostics = solve_option_hedge(
                priced_universe_df, target_option_id, target_position, hedge_mode=hedge_mode,
            )
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Hedge at time index {t_idx}: {exc}") from exc
        for name, value in diagnostics.items():
            state_df.loc[t_idx, name] = value
        state_df.loc[t_idx, "greek_error_ratio"] = priced_universe_df["Greek_Error_Ratio"].max()

        # 5) Gross exposure — recorded only once at inception
        if initial_gross_exposure is None:
            initial_gross_exposure = compute_initial_gross_exposure(
                priced_universe_df=priced_universe_df,
                option_weights=option_weights,
                w_underlying=w_underlying,
                S_t=S_t,
            )

        # 6) Write all metrics for the current timestep
        write_row_metrics(
            state_df=state_df,
            t_idx=t_idx,
            full_slice=full_slice,
            priced_universe_df=priced_universe_df,
            option_weights=option_weights,
            w_underlying=w_underlying,
            target_option_id=target_option_id,
            target_position=target_position,
        )

        settle_cash_account(state_df, full_slice, t_idx, prev_slice, prev_t)

    return state_df, initial_gross_exposure

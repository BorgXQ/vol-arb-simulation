import numpy as np
import pandas as pd
from src.utils import implied_volatility_bs
from src.calc import (
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
        "net_delta", "net_gamma", "net_vega",
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
        "net_delta", "net_gamma", "net_vega",
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
    N=4096,
    alpha=1.5,
    eta=0.25,
    eps_S_rel=0.01,
    eps_v_rel=0.05,
):
    """
    Price the current options slice and compute Delta, Gamma, and Vega via
    central finite differences.

    Parameters
    ----------
    options_slice : pd.DataFrame
        Options to price; must contain Strike, Type, T, r.
    S0 : float
        Current underlying price.
    params : array-like
        Heston parameters (kappa_v, theta_v, xi_v, rho, v0).
    N : int, optional
        FFT grid size (default 4096).
    alpha : float, optional
        Carr-Madan damping parameter (default 1.5).
    eta : float, optional
        Frequency grid spacing (default 0.25).
    eps_S_rel : float, optional
        Relative spot bump for finite-difference Greeks (default 0.01).
    eps_v_rel : float, optional
        Relative variance bump for finite-difference Greeks (default 0.05).

    Returns
    -------
    pd.DataFrame
        Input slice augmented with Theo_Price, Delta, Gamma, Vega,
        Theo_IV, IV_Diff, and Abs_IV_Diff columns.
    """
    kappa_v, theta_v, xi_v, rho, v0 = map(float, params)

    df = options_slice.copy()
    df = add_option_id_column(df)

    eps_S = max(1e-4, eps_S_rel * S0)
    eps_v = max(1e-5, eps_v_rel * max(v0, 1e-4))

    out_frames = []

    for (T, r), grp in df.groupby(["T", "r"], sort=False):
        T = float(T)
        r = float(r)
        strikes = grp["Strike"].to_numpy(dtype=float)
        types = grp["Type"].to_numpy()

        def model_prices_for(S_bump, v0_bump):
            K_grid, call_grid = CM99_call_price_grid_fft(
                S0=S_bump,
                T=T,
                r=r,
                kappa_v=kappa_v,
                theta_v=theta_v,
                xi_v=xi_v,
                rho=rho,
                v0=v0_bump,
                N=N,
                alpha=alpha,
                eta=eta,
            )
            call_vals = interpolate_call_prices(strikes, K_grid, call_grid)
            put_vals = put_from_call_parity(call_vals, S_bump, strikes, r, T)
            return np.where(types == "C", call_vals, put_vals)

        price_0 = model_prices_for(S0, v0)
        price_up = model_prices_for(S0 + eps_S, v0)
        price_dn = model_prices_for(max(S0 - eps_S, 1e-8), v0)
        price_vu = model_prices_for(S0, v0 + eps_v)
        price_vd = model_prices_for(S0, max(v0 - eps_v, 1e-8))

        delta = (price_up - price_dn) / (2.0 * eps_S)
        gamma = (price_up - 2.0 * price_0 + price_dn) / (eps_S ** 2)
        vega = (price_vu - price_vd) / (2.0 * eps_v)

        grp_out = grp.copy()
        grp_out["Theo_Price"] = price_0
        grp_out["Delta"] = delta
        grp_out["Gamma"] = gamma
        grp_out["Vega"] = vega

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

def solve_gamma_vega_delta_hedge(priced_universe_df, target_option_id, target_position):
    """
    Hedge the target's gamma and vega with non-target options, then
    delta-hedge the residual exposure with the underlying.

    Parameters
    ----------
    priced_universe_df : pd.DataFrame
        Must contain Option_ID, Delta, Gamma, and Vega columns.
    target_option_id : str
        ID of the target contract.
    target_position : float
        Signed position in the target (+1 long, -1 short).

    Returns
    -------
    option_weights : dict
        Mapping of Option_ID to signed weight.
    w_underlying : float
        Delta-hedge weight in the underlying.
    """
    df = priced_universe_df.copy()

    target = df[df["Option_ID"] == target_option_id]
    if target.empty:
        raise ValueError(f"Target {target_option_id} not found in priced universe.")

    target = target.iloc[0]
    hedgers = df[df["Option_ID"] != target_option_id].copy()

    option_weights = {target_option_id: float(target_position)}

    if hedgers.empty:
        w_underlying = -target_position * float(target["Delta"])
        return option_weights, float(w_underlying)

    A = np.vstack([
        hedgers["Gamma"].to_numpy(dtype=float),
        hedgers["Vega"].to_numpy(dtype=float),
    ])
    b = -target_position * np.array([
        float(target["Gamma"]),
        float(target["Vega"]),
    ])

    # Minimum-norm solution
    hedge_w = np.linalg.lstsq(A, b, rcond=None)[0]

    hedger_ids = hedgers["Option_ID"].tolist()
    for oid, w in zip(hedger_ids, hedge_w):
        option_weights[oid] = float(w)

    total_delta = target_position * float(target["Delta"])
    total_delta += float(np.dot(hedge_w, hedgers["Delta"].to_numpy(dtype=float)))

    w_underlying = -total_delta
    return option_weights, float(w_underlying)


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


def update_pnl_from_previous_row(state_df, options_prev, options_curr, t_idx, prev_t_idx):
    """
    Compute incremental and cumulative PnL from the previous timestep's positions.

    PnL is the mark-to-market change in value of holdings carried from prev_t_idx.

    Parameters
    ----------
    state_df : pd.DataFrame
    options_prev : pd.DataFrame
        Options data at the previous timestep.
    options_curr : pd.DataFrame
        Options data at the current timestep.
    t_idx : int
        Current time index.
    prev_t_idx : int
        Previous time index.
    """
    prev_prices = options_prev.set_index("Option_ID")["Market_Price"]
    curr_prices = options_curr.set_index("Option_ID")["Market_Price"]
    common_ids = prev_prices.index.intersection(curr_prices.index)

    pnl = 0.0
    for oid in common_ids:
        w_prev = float(state_df.loc[prev_t_idx, f"w_{oid}"])
        pnl += w_prev * (float(curr_prices.loc[oid]) - float(prev_prices.loc[oid]))

    S_prev = float(state_df.loc[prev_t_idx, "S_t"])
    S_curr = float(state_df.loc[t_idx, "S_t"])
    w_u_prev = float(state_df.loc[prev_t_idx, "w_underlying"])

    pnl += w_u_prev * (S_curr - S_prev)

    state_df.loc[t_idx, "pnl_incremental"] = pnl
    state_df.loc[t_idx, "pnl_cumulative"] = float(state_df.loc[prev_t_idx, "pnl_cumulative"]) + pnl


# ==============================================================================
# MAIN STRATEGY LOOP
# ==============================================================================

def run_vol_arb_strategy(
    options_market_df,
    calibration_N=1024,
    pricing_N=4096,
    alpha=1.5,
    eta=0.25,
    n_each_side=3,
    dt=1/252,
    exit_days_before_expiry=10,
):
    """
    Run the full volatility arbitrage strategy over all timesteps.

    Strategy rules
    --------------
    - t=0 : calibrate model, price universe, select target once, hedge.
    - t>=1 : target is fixed; universe is rebuilt each period; recalibrate,
             reprice, and rehedge.
    - Exit  : zero positions when T <= exit_days_before_expiry * dt.

    Parameters
    ----------
    options_market_df : pd.DataFrame
        Must contain Market_IV and all standard option columns.
    calibration_N : int, optional
        FFT grid size used during calibration (default 1024).
    pricing_N : int, optional
        FFT grid size used for theoretical pricing (default 4096).
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

    Returns
    -------
    state_df : pd.DataFrame
        Full strategy state across all timesteps.
    initial_gross_exposure : float
        Gross notional exposure at inception.
    """
    df = add_option_id_column(options_market_df).copy()

    if "Market_IV" not in df.columns:
        raise ValueError("options_market_df must already contain Market_IV.")

    time_indices = sorted(df["t_index"].unique())
    state_df = initialize_strategy_state_df(df)

    target_option_id = None
    target_position = None
    initial_gross_exposure = None

    for i, t_idx in enumerate(time_indices):
        full_slice = df[df["t_index"] == t_idx].copy().reset_index(drop=True)
        write_static_state(state_df, t_idx, full_slice)

        # PnL from previous holdings into current time
        if i > 0:
            prev_t = time_indices[i - 1]
            prev_slice = df[df["t_index"] == prev_t].copy()
            update_pnl_from_previous_row(state_df, prev_slice, full_slice, t_idx, prev_t)

        # Exit rule near expiry
        T_t = float(full_slice["T"].iloc[0])
        if T_t <= exit_days_before_expiry * dt:
            zero_current_positions(state_df, t_idx)
            state_df.loc[t_idx, "target_option_id"] = target_option_id
            state_df.loc[t_idx, "target_position"] = float(target_position) if target_position is not None else 0.0
            continue

        # Build current tradable universe
        universe_df = select_otm_universe(full_slice, n_each_side=n_each_side)
        universe_df = ensure_target_in_universe(full_slice, universe_df, target_option_id)

        S_t = float(full_slice["S_t"].iloc[0])

        # 1) Calibration
        trader_params, _, _, _ = CM99_calibration_market(
            universe_df,
            S0=S_t,
            N=calibration_N,
            alpha=alpha,
            eta=eta,
        )
        write_model_params(state_df, t_idx, trader_params)

        # 2) Theoretical pricing, Greeks, and implied volatility
        priced_universe_df = price_slice_with_heston_and_greeks(
            universe_df,
            S0=S_t,
            params=trader_params,
            N=pricing_N,
            alpha=alpha,
            eta=eta,
        )

        # 3) Target selection — performed only once at inception
        if target_option_id is None:
            target_option_id, target_position, _ = select_target_contract(priced_universe_df)

        # 4) Gamma/vega/delta hedge around the fixed target
        option_weights, w_underlying = solve_gamma_vega_delta_hedge(
            priced_universe_df,
            target_option_id=target_option_id,
            target_position=target_position,
        )

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

    return state_df, initial_gross_exposure
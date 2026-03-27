import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq
from src.calc import (
    CM99_call_price_grid_jd_fft,
    interpolate_call_prices,
    put_from_call_parity
)


# ==============================================================================
# STOCHASTIC VOLATILITY + JUMP DIFFUSION PATH SIMULATION
# ==============================================================================

def plot_stochastic_volatility_jump_path(
        S0=100.0,            # initial price
        mu=0.03,             # drift (3% annual)
        r=0.02,              # risk-free rate (for reference)
        v0=0.03514,          # initial variance (20% vol)
        kappa=11.31,         # mean reversion speed
        theta=0.05167,       # long-term variance (5% vol)
        xi=0.2459,           # volatility of volatility (25%)
        rho=-0.6833,         # strong negative correlation for leverage effect
        jump_intensity=0.7,  # average 0.7 jumps per year
        jump_mean=-0.02,     # average jump size (-2% for downward bias)
        jump_std=0.04,       # jump size standard deviation (4%)
        T=1.0,
        use_last_n=60,
        n_steps=252,
        seed=42,
):
    """
    Simulate a price path using a Heston stochastic volatility model with
    Merton log-normal jumps.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    mu : float
        Physical drift.
    r : float
        Risk-free rate (reference only).
    v0 : float
        Initial variance.
    kappa : float
        Variance mean-reversion speed.
    theta : float
        Long-run variance.
    xi : float
        Volatility of variance.
    rho : float
        Price/variance Brownian correlation.
    jump_intensity : float
        Poisson jump arrival rate (per year).
    jump_mean : float
        Mean log-jump size.
    jump_std : float
        Standard deviation of log-jump size.
    T : float
        Simulation horizon in years (default 1.0).
    use_last_n : int
        Number of trailing timesteps to use as the display window.
    n_steps : int
        Total number of simulation steps (default 252).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    S : ndarray, shape (n_steps + 1,)
        Simulated price path.
    v : ndarray, shape (n_steps + 1,)
        Simulated variance path.
    time_grid : ndarray, shape (n_steps + 1,)
        Time grid in years.
    """
    
    # Set random seed
    np.random.seed(seed)
    dt = T / n_steps     # time step
    
    # ========== PRE-ALLOCATE ARRAYS ==========
    S = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    S[0] = S0
    v[0] = v0
    
    # ========== SIMULATION ==========
    for i in range(n_steps):
        # Generate correlated random numbers
        Z1 = np.random.normal(0, 1)
        Z2 = np.random.normal(0, 1)
        Z_vol = Z1
        Z_price = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        # Ensure variance stays non-negative (full truncation scheme)
        v_plus = max(v[i], 0)
        
        # Heston dynamics
        S[i+1] = S[i] * np.exp((mu - 0.5 * v_plus) * dt + np.sqrt(v_plus * dt) * Z_price)
        v[i+1] = v[i] + kappa * (theta - v_plus) * dt + xi * np.sqrt(v_plus * dt) * Z_vol
        
        # ========== ADD JUMPS (Poisson process) ==========
        # Number of jumps in this interval
        n_jumps = np.random.poisson(jump_intensity * dt)
        
        if n_jumps > 0:
            jump_sizes = np.random.normal(jump_mean, jump_std, n_jumps)
            S[i+1] = S[i+1] * np.exp(np.sum(jump_sizes))
    
    # Ensure variance stays positive (final check)
    v = np.maximum(v, 1e-6)
    volatility = np.sqrt(v)
    time_grid = np.linspace(0, T, n_steps + 1)
    
    # # ========== PLOTTING ==========
    # fig = go.Figure()

    # # Price (left axis)
    # fig.add_trace(go.Scatter(
    #     x=time_grid,
    #     y=S,
    #     name="Price",
    #     line=dict(width=2),
    #     yaxis="y1",
    #     hovertemplate="Price: %{y:.6f}<br>t=%{x}<extra></extra>"
    # ))

    # # Volatility (right axis)
    # fig.add_trace(go.Scatter(
    #     x=time_grid,
    #     y=volatility,
    #     name="Volatility",
    #     line=dict(width=2),
    #     yaxis="y2",
    #     hovertemplate="Volatility: %{y:.6f}<br>t=%{x}<extra></extra>"
    # ))

    # if use_last_n < n_steps:
    #     fig.add_vline(
    #         x=T - use_last_n * dt,
    #         line=dict(color="gray", width=1, dash="dash"),
    #         annotation_text="Display Window Start",
    #         annotation_position="top"
    #     )

    # fig.update_layout(
    #     title="Asset Price & Volatility (Stochastic Volatility + Jumps)",
    #     xaxis=dict(title="Time (Years)"),
    #     yaxis=dict(title="Price ($)"),
    #     yaxis2=dict(
    #         title="Volatility",
    #         overlaying="y",
    #         side="right"
    #     ),
    #     legend=dict(x=1.05, y=1),
    #     template="plotly_white",
    #     hovermode="x unified"
    # )

    # fig.show()
    
    # # ========== PRINT STATISTICS ==========
    # print("=" * 60)
    # print("SIMULATION STATISTICS")
    # print("=" * 60)
    # print(f"Initial price: ${S0:.2f}")
    # print(f"Final price: ${S[-1]:.2f}")
    # print(f"Total return: {(S[-1]/S0 - 1):.2%}")
    # print(f"Max price: ${np.max(S):.2f}")
    # print(f"Min price: ${np.min(S):.2f}")
    # print(f"Annualized realized volatility: {np.std(np.diff(np.log(S)) / np.sqrt(dt)):.2%}")
    # print(f"Average volatility: {np.mean(volatility):.2%}")
    # print(f"Volatility range: [{np.min(volatility):.2%}, {np.max(volatility):.2%}]")
    # print("=" * 60)
    
    return S, v, time_grid


# ==============================================================================
# SYNTHETIC MARKET OPTION PRICE GENERATION
# ==============================================================================

def generate_market_option_prices_across_time(
    S_path,
    v_path,
    v0_m,
    kappa_m,
    theta_m,
    xi_m,
    rho_m,
    jump_intensity_m,
    jump_mean_m,
    jump_std_m,
    r=0.02,
    dt=1/252,
    N=1024,
    alpha=1.5,
    eta=0.25,
    use_last_n=60,
    noise_scale=0.005,
):
    """
    Generate synthetic market call and put prices across time for a fixed
    strike universe.

    The market model is a perturbed version of the true parameters; at each
    timestep the current state is:

        S_t       = S_window[t]
        v_t_market = v_window[t] * 1.05

    Parameters
    ----------
    S_path : array-like
        Simulated price path.
    v_path : array-like
        Simulated variance path.
    v0_m, kappa_m, theta_m, xi_m, rho_m : float
        Market Heston parameters (perturbed from true params).
    jump_intensity_m, jump_mean_m, jump_std_m : float
        Market jump parameters.
    r : float, optional
        Risk-free rate (default 0.02).
    dt : float, optional
        Time step in years (default 1/252).
    N : int, optional
        FFT grid size (default 1024).
    alpha : float, optional
        Carr-Madan damping parameter (default 1.5).
    eta : float, optional
        Frequency grid spacing (default 0.25).
    use_last_n : int, optional
        Number of trailing timesteps to use (default 60).
    noise_scale : float, optional
        Proportional noise scale applied to prices (default 0.005).

    Returns
    -------
    options_df : pd.DataFrame
        Long-format DataFrame with one row per contract per timestep.
    """

    # Window selection
    S_window = np.asarray(S_path[-use_last_n:], dtype=float)
    v_window = np.asarray(v_path[-use_last_n:], dtype=float)

    n_steps = len(S_window)

    # Strike universe based on the price range in the window
    S_min = np.min(S_window)
    S_max = np.max(S_window)

    K_min = int(np.floor(S_min - 5.0))
    K_max = int(np.ceil(S_max + 5.0))
    strikes = np.arange(K_min, K_max + 1, 2, dtype=float)

    # Generate prices over time
    rows = []

    for t_idx in range(n_steps):
        S_t = S_window[t_idx]

        # Time to expiry measured from current time to final time in this window
        tau_steps = (n_steps - 1) - t_idx
        T_t = max(tau_steps * dt, 1e-10)

        # Use current variance state; scaled to reflect market mismatch
        v_t_market = max(v_window[t_idx] * 1.05, 1e-10)

        # Price all call strikes at once
        K_grid, call_grid = CM99_call_price_grid_jd_fft(
            S0=S_t,
            T=T_t,
            r=r,
            v0=v_t_market,
            kappa_v=kappa_m,
            theta_v=theta_m,
            xi_v=xi_m,
            rho=rho_m,
            lambda_j=jump_intensity_m,
            mu_j=jump_mean_m,
            sigma_j=jump_std_m,
            N=N,
            alpha=alpha,
            eta=eta,
        )

        call_prices = interpolate_call_prices(strikes, K_grid, call_grid)
        put_prices = put_from_call_parity(
            call_prices=call_prices,
            S0=S_t,
            strikes=strikes,
            r=r,
            T=T_t,
        )

        # Add realistic market noise
        noise_scale = noise_scale

        call_noise = np.random.normal(0, noise_scale, size=len(call_prices))
        put_noise  = np.random.normal(0, noise_scale, size=len(put_prices))

        call_prices = call_prices * (1 + call_noise)
        put_prices  = put_prices  * (1 + put_noise)

        # Enforce no-arbitrage bounds
        intrinsic_call = np.maximum(S_t - strikes, 0.0)
        intrinsic_put  = np.maximum(strikes - S_t, 0.0)

        call_prices = np.maximum(call_prices, intrinsic_call)
        put_prices  = np.maximum(put_prices, intrinsic_put)

        # Store calls
        for K, price in zip(strikes, call_prices):
            rows.append({
                "t_index": t_idx,
                "S_t": S_t,
                "v_t_market": v_t_market,
                "Strike": K,
                "Type": "C",
                "T": T_t,
                "r": r,
                "Market_Price": float(price),
                "Contract": f"C_{int(K)}_{t_idx}",
            })

        # Store puts
        for K, price in zip(strikes, put_prices):
            rows.append({
                "t_index": t_idx,
                "S_t": S_t,
                "v_t_market": v_t_market,
                "Strike": K,
                "Type": "P",
                "T": T_t,
                "r": r,
                "Market_Price": float(price),
                "Contract": f"P_{int(K)}_{t_idx}",
            })

    options_df = pd.DataFrame(rows)

    return options_df


# ==============================================================================
# BLACK-SCHOLES PRICING
# ==============================================================================

def bs_price(S, K, T, r, sigma, option_type):
    """
    Black-Scholes price for a European call or put.

    Parameters
    ----------
    S : float
        Underlying price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    sigma : float
        Implied volatility.
    option_type : str
        'C' for call, 'P' for put.

    Returns
    -------
    float
    """
    if T <= 0:
        if option_type == "C":
            return max(S - K, 0.0)
        elif option_type == "P":
            return max(K - S, 0.0)
        raise ValueError("option_type must be 'C' or 'P'")

    sigma = max(float(sigma), 1e-12)
    sqrtT = np.sqrt(T)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if option_type == "C":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "P":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'C' or 'P'")
    

# ==============================================================================
# IMPLIED VOLATILITY INVERSION
# ==============================================================================

def implied_volatility_bs(price, S, K, T, r, option_type,
                          sigma_lower=1e-8, sigma_upper=5.0):
    """
    Compute implied volatility from a Black-Scholes price via 1D root-finding.

    Returns np.nan if no valid IV exists within the search bracket.

    Parameters
    ----------
    price : float
        Observed option price.
    S : float
        Underlying price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    option_type : str
        'C' for call, 'P' for put.
    sigma_lower : float, optional
        Lower bound of the volatility search (default 1e-8).
    sigma_upper : float, optional
        Upper bound of the volatility search (default 5.0).

    Returns
    -------
    float
        Implied volatility, or np.nan if not found.
    """
    if any(x <= 0 for x in [S, K]):
        return np.nan

    # Handle expiry separately
    if T <= 0:
        return np.nan

    # No-arbitrage bounds
    disc_K = K * np.exp(-r * T)

    if option_type == "C":
        lower_bound = max(S - disc_K, 0.0)
        upper_bound = S
    elif option_type == "P":
        lower_bound = max(disc_K - S, 0.0)
        upper_bound = disc_K
    else:
        return np.nan

    if not (lower_bound <= price <= upper_bound):
        return np.nan

    def objective(sig):
        return bs_price(S, K, T, r, sig, option_type) - price

    try:
        return brentq(objective, sigma_lower, sigma_upper, maxiter=200)
    except ValueError:
        return np.nan


# ==============================================================================
# IMPLIED VOLATILITY DATAFRAME WRAPPER
# ==============================================================================

def append_market_iv(options_df):
    """
    Compute and append market implied volatility to an options DataFrame.

    Parameters
    ----------
    options_df : pd.DataFrame
        Must contain columns: S_t, Strike, T, r, Type, Market_Price.

    Returns
    -------
    pd.DataFrame
        Copy with an added 'Market_IV' column.
    """
    required_cols = {"S_t", "Strike", "T", "r", "Type", "Market_Price"}
    missing = required_cols - set(options_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = options_df.copy()

    market_iv = [
        implied_volatility_bs(
            price=price,
            S=S,
            K=K,
            T=T,
            r=r,
            option_type=opt_type,
        )
        for S, K, T, r, opt_type, price in zip(
            df["S_t"].to_numpy(dtype=float),
            df["Strike"].to_numpy(dtype=float),
            df["T"].to_numpy(dtype=float),
            df["r"].to_numpy(dtype=float),
            df["Type"].to_numpy(),
            df["Market_Price"].to_numpy(dtype=float),
        )
    ]

    df["Market_IV"] = market_iv
    return df


# ==============================================================================
# STATE DATAFRAME DIAGNOSTICS
# ==============================================================================

def strip_state_df(state_df):
    """
    Reduce the full strategy state DataFrame to a compact diagnostic view.

    Extracts target-option IV/price columns, selects core columns and weight
    columns, and rounds all numeric values to 6 decimal places.

    Parameters
    ----------
    state_df : pd.DataFrame
        Full strategy state as returned by run_vol_arb_strategy.

    Returns
    -------
    pd.DataFrame
        Reduced DataFrame with target metrics, weights, and core columns.
    """
    df = state_df.copy()

    # Extract target IV columns
    mkt_iv_target = []
    theo_iv_target = []
    iv_diff_target = []
    mkt_price_target = []
    theo_price_target = []

    for _, row in df.iterrows():
        target = row["target_option_id"]

        if pd.isna(target):
            mkt_iv_target.append(np.nan)
            theo_iv_target.append(np.nan)
            iv_diff_target.append(np.nan)
            mkt_price_target.append(np.nan)
            theo_price_target.append(np.nan)
            continue

        mkt_iv_target.append(row.get(f"mkt_iv_{target}", np.nan))
        theo_iv_target.append(row.get(f"theo_iv_{target}", np.nan))
        iv_diff_target.append(row.get(f"iv_diff_{target}", np.nan))
        mkt_price_target.append(row.get(f"mkt_price_{target}", np.nan))
        theo_price_target.append(row.get(f"theo_price_{target}", np.nan))


    target_iv_df = pd.DataFrame({
        "mkt_iv_target": mkt_iv_target,
        "theo_iv_target": theo_iv_target,
        "iv_diff_target": iv_diff_target,
        "mkt_price_target": mkt_price_target,
        "theo_price_target": theo_price_target,
    }, index=df.index)

    # Select columns
    weight_cols = [
        col for col in df.columns
        if col.startswith("w_") and not col.startswith(("mkt_iv_", "theo_iv_", "iv_diff_"))
    ]

    core_cols = [
        "S_t",
        "T",
        "target_option_id",
        "target_position",
        "pnl_incremental",
        "pnl_cumulative",
    ]

    reduced_df = pd.concat(
        [df[core_cols], target_iv_df, df[weight_cols]],
        axis=1
    ).copy()

    # Round numeric columns to 6 decimal places
    exclude_cols = {"target_option_id", "target_position"}

    numeric_cols = [
        col for col in reduced_df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(reduced_df[col])
    ]

    reduced_df[numeric_cols] = reduced_df[numeric_cols].round(6)

    return reduced_df


def plot_strategy_dashboard_plotly(state_df_reduced, initial_gross_exposure, periods_per_year=252):
    """
    Render a 4-panel Plotly dashboard for strategy diagnostics.

    Panels
    ------
    1. Hedge weights over time.
    2. Target market price vs theoretical price.
    3. Target market IV vs theoretical IV.
    4. Cumulative returns with Sharpe, Sortino, Max Drawdown, and Calmar.

    Parameters
    ----------
    state_df_reduced : pd.DataFrame
        Compact state DataFrame as returned by strip_state_df.
    initial_gross_exposure : float
        Gross notional at inception; used to normalise PnL into returns.
    periods_per_year : int, optional
        Annualisation factor (default 252).

    Notes
    -----
    Risk metrics are computed from pnl_incremental. If true portfolio returns
    are available, substitute them for more standard ratio calculations.
    """

    df = state_df_reduced.copy().reset_index(drop=True)

    # Identify target weight column and non-target weight columns
    target_option_id = df["target_option_id"].dropna().iloc[0] if df["target_option_id"].notna().any() else None
    target_weight_col = f"w_{target_option_id}" if target_option_id is not None else None

    weight_cols = [
        col for col in df.columns
        if col.startswith("w_") and col != "w_underlying" and col != target_weight_col
    ]

    # Risk metrics from pnl_incremental
    capital = initial_gross_exposure

    returns = pd.to_numeric(df["pnl_incremental"], errors="coerce").fillna(0.0) / capital
    returns_cum = returns.cumsum()

    mean_ret = returns.mean()
    std_ret = returns.std(ddof=1)
    downside_std = returns[returns < 0].std(ddof=1)

    sharpe = np.nan
    if std_ret > 0:
        sharpe = np.sqrt(periods_per_year) * mean_ret / std_ret

    sortino = np.nan
    if pd.notna(downside_std) and downside_std > 0:
        sortino = np.sqrt(periods_per_year) * mean_ret / downside_std

    running_max = returns_cum.cummax()
    drawdown = returns_cum - running_max
    max_drawdown = drawdown.min()

    annualized_return = mean_ret * periods_per_year

    calmar = np.nan
    if max_drawdown < 0:
        calmar = annualized_return / abs(max_drawdown)

    metrics_text = (
        f"Sharpe: {sharpe:.3f}<br>"
        f"Sortino: {sortino:.3f}<br>"
        f"Max Drawdown: {max_drawdown:.6f}<br>"
        f"Calmar: {calmar:.3f}"
    )

    # Build figure
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Hedge Weights",
            "Target Market Price vs Theoretical Price",
            "Target Market IV vs Theoretical IV",
            "Cumulative Returns"
        )
    )

    # Panel 1: Hedge weights
    weight_plot_cols = weight_cols.copy()

    if "w_underlying" in df.columns:
        weight_plot_cols.append("w_underlying")

    if target_weight_col is not None and target_weight_col in df.columns:
        weight_plot_cols.append(target_weight_col)

    # Plot actual lines without hover on them
    for col in weight_cols:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col,
                showlegend=False,
                hoverinfo="skip"
            ),
            row=1,
            col=1
        )

    if "w_underlying" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["w_underlying"],
                mode="lines",
                name="w_underlying",
                line=dict(width=2),
                hoverinfo="skip"
            ),
            row=1,
            col=1
        )

    if target_weight_col is not None and target_weight_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[target_weight_col],
                mode="lines",
                name=target_weight_col,
                line=dict(width=3),
                hoverinfo="skip"
            ),
            row=1,
            col=1
        )

    # Build hover text showing only non-zero weights
    hover_text_weights = []
    tol = 1e-12

    for i in df.index:
        entries = [f"t={i}"]
        for col in weight_plot_cols:
            val = df.loc[i, col]
            if pd.notna(val) and abs(val) > tol:
                entries.append(f"{col}: {val:.6f}")
        hover_text_weights.append("<br>".join(entries))

    # Invisible helper trace that carries the hover tooltip
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[0.0] * len(df),
            mode="lines",
            line=dict(width=20, color="rgba(0,0,0,0)"),
            name="weights_hover",
            showlegend=False,
            hovertemplate="%{text}<extra></extra>",
            text=hover_text_weights
        ),
        row=1,
        col=1
    )

    # Panel 2: Price comparison
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["mkt_price_target"],
            mode="lines",
            name="Market Price",
            line=dict(width=2),
            hovertemplate="Market Price: %{y:.6f}<br>t=%{x}<extra></extra>"
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["theo_price_target"],
            mode="lines",
            name="Theoretical Price",
            line=dict(width=2, dash="dash"),
            hovertemplate="Theoretical Price: %{y:.6f}<br>t=%{x}<extra></extra>"
        ),
        row=2,
        col=1
    )

    # Panel 3: IV comparison
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["mkt_iv_target"],
            mode="lines",
            name="Market IV",
            line=dict(width=2),
            hovertemplate="Market IV: %{y:.6f}<br>t=%{x}<extra></extra>"
        ),
        row=3,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["theo_iv_target"],
            mode="lines",
            name="Theoretical IV",
            line=dict(width=2, dash="dash"),
            hovertemplate="Theoretical IV: %{y:.6f}<br>t=%{x}<extra></extra>"
        ),
        row=3,
        col=1
    )

    # Panel 4: Cumulative returns
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=returns_cum,
            mode="lines",
            name="Cumulative Returns",
            line=dict(width=3),
            hovertemplate="Cumulative Returns: %{y:.6f}<br>t=%{x}<extra></extra>"
        ),
        row=4,
        col=1
    )

    # Metrics annotation in the PnL panel
    fig.add_annotation(
        x=0.99,
        y=0.02,
        xref="paper",
        yref="paper",
        text=metrics_text,
        showarrow=False,
        align="left",
        bordercolor="black",
        borderwidth=1,
        bgcolor="white",
        opacity=0.9
    )

    # Layout
    fig.update_layout(
        height=1100,
        width=1200,
        title="Volatility Arbitrage Dashboard",
        template="plotly_white",
        hovermode="x unified"
    )

    fig.update_yaxes(title_text="Weight", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    fig.update_yaxes(title_text="Implied Volatility", row=3, col=1)
    fig.update_yaxes(title_text="Cumulative Returns", row=4, col=1)

    fig.update_xaxes(title_text="Time Index", row=4, col=1)

    fig.show()
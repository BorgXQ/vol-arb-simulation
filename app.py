import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Ensure local package imports work when app is launched from this directory.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.utils import (  # noqa: E402
    plot_stochastic_volatility_jump_path,
    generate_market_option_prices_across_time,
    append_market_iv,
    strip_state_df,
)
from src.vol_arb import (  # noqa: E402
    add_option_id_column,
    ensure_target_in_universe,
    price_slice_with_heston_and_greeks,
    run_vol_arb_strategy,
    select_otm_universe,
    select_target_contract,
)
from src.calc import CM99_calibration_market  # noqa: E402


DEFAULTS = {
    "seed": 1,
    "use_last_n": 60,
    "kappa": 6.4,
    "theta": 0.077,
    "xi": 0.24,
    "rho": -0.7,
    "jump_on": True,
    "lambdaj": 1.62,
    "muj": -0.165,
    "sigmaj": 0.135,
    "noise_scale": 0.005,
    "exit_days_before_expiry": 10,
    "calibration_N": 1024,
    "pricing_N": 4096,
    "alpha": 1.5,
    "eta": 0.25,
    "n_each_side": 3,
    "dt": 1 / 252,
    "r": 0.02,
}


# -----------------------------
# Data pipeline
# -----------------------------
@st.cache_data(show_spinner=False)
def run_analysis_cached(
    seed: int,
    use_last_n: int,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    jump_on: bool,
    lambdaj: float,
    muj: float,
    sigmaj: float,
    noise_scale: float,
    exit_days_before_expiry: int,
    calibration_N: int,
    pricing_N: int,
    alpha: float,
    eta: float,
    n_each_side: int,
    dt: float,
    r: float,
):
    jump_intensity = lambdaj if jump_on else 0.0
    jump_mean = muj if jump_on else 0.0
    jump_std = sigmaj if jump_on else 0.0

    S_path, v_path, t_grid = plot_stochastic_volatility_jump_path(
        seed=seed,
        use_last_n=use_last_n,
    )

    options_market_df = generate_market_option_prices_across_time(
        S_path=S_path,
        v_path=v_path,
        kappa_m=kappa,
        theta_m=theta,
        xi_m=xi,
        rho_m=rho,
        v0_m=0.042,
        jump_intensity_m=jump_intensity,
        jump_mean_m=jump_mean,
        jump_std_m=jump_std,
        r=r,
        dt=dt,
        N=4096,
        alpha=alpha,
        eta=eta,
        use_last_n=use_last_n,
        noise_scale=noise_scale,
    )
    options_market_df = append_market_iv(options_market_df)

    state_df, initial_gross_exposure = run_vol_arb_strategy(
        options_market_df=options_market_df,
        calibration_N=calibration_N,
        pricing_N=pricing_N,
        alpha=alpha,
        eta=eta,
        n_each_side=n_each_side,
        dt=dt,
        exit_days_before_expiry=exit_days_before_expiry,
    )

    state_df_reduced = strip_state_df(state_df).iloc[: use_last_n - (exit_days_before_expiry + 1)].copy()

    # Build t=0 diagnostics for row 2.
    full_slice_t0 = add_option_id_column(
        options_market_df.loc[options_market_df["t_index"] == 0].copy().reset_index(drop=True)
    )
    universe_t0 = select_otm_universe(full_slice_t0, n_each_side=n_each_side)

    S_t0 = float(full_slice_t0["S_t"].iloc[0])
    trader_params_t0, _, _, _ = CM99_calibration_market(
        universe_t0,
        S0=S_t0,
        N=calibration_N,
        alpha=alpha,
        eta=eta,
    )
    priced_universe_t0 = price_slice_with_heston_and_greeks(
        universe_t0,
        S0=S_t0,
        params=trader_params_t0,
        N=pricing_N,
        alpha=alpha,
        eta=eta,
    )
    target_option_id_t0, target_position_t0, target_row_t0 = select_target_contract(priced_universe_t0)
    priced_universe_t0 = ensure_target_in_universe(full_slice_t0, priced_universe_t0, target_option_id_t0)

    return {
        "S_path": S_path,
        "v_path": v_path,
        "t_grid": t_grid,
        "options_market_df": options_market_df,
        "state_df_reduced": state_df_reduced,
        "initial_gross_exposure": float(initial_gross_exposure),
        "full_slice_t0": full_slice_t0,
        "priced_universe_t0": priced_universe_t0,
        "target_option_id_t0": target_option_id_t0,
        "target_position_t0": float(target_position_t0),
        "target_row_t0": target_row_t0,
        "trader_params_t0": trader_params_t0,
    }


# -----------------------------
# Plot helpers
# -----------------------------
def make_path_figure(S_path: np.ndarray, v_path: np.ndarray, use_last_n: int) -> go.Figure:
    idx = np.arange(len(S_path))
    vol = np.sqrt(np.maximum(v_path, 1e-12))
    window_start = max(0, len(S_path) - use_last_n)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=S_path,
            name="Price",
            line=dict(width=2),
            yaxis="y1",
            hovertemplate="Index: %{x}<br>Price: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=vol,
            name="Volatility",
            line=dict(width=2),
            yaxis="y2",
            hovertemplate="Index: %{x}<br>Volatility: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=window_start,
        line_width=1,
        line_dash="dash",
        line_color="white",
        annotation_text="Window start",
        annotation=dict(
            xanchor="right",  # Anchor the annotation to the right side of the text
            xshift=-12,       # Shift left by 20 pixels
            yshift=8         # Optional: shift up/down
        )
    )
    fig.update_layout(
        title="Underlying Price Path and Instantaneous Volatility",
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(title="Time Index"),
        yaxis=dict(title="Price ($)"),
        yaxis2=dict(title="Volatility", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
    )
    return fig


def make_iv_smile_figure(window_start: int, priced_universe_t0: pd.DataFrame) -> go.Figure:
    df = priced_universe_t0.copy().sort_values(["Type", "Strike"])
    fig = go.Figure()

    for opt_type, label in [("P", "Put"), ("C", "Call")]:
        sub = df[df["Type"] == opt_type].sort_values("Strike")
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["Strike"],
                y=sub["Market_IV"],
                mode="lines+markers",
                name=f"Market {label} IV",
                hovertemplate="Strike: %{x}<br>Market IV: %{y:.6f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sub["Strike"],
                y=sub["Theo_IV"],
                mode="markers",
                marker=dict(size=10, symbol="diamond"),
                name=f"Theo {label} IV",
                hovertemplate="Strike: %{x}<br>Theo IV: %{y:.6f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"t = {window_start} IV Smile in Calibration Universe",
        template="plotly_white",
        xaxis_title="Strike",
        yaxis_title="Implied Volatility",
        hovermode="closest",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_target_iv_diff_figure(window_start: int, priced_universe_t0: pd.DataFrame, target_option_id_t0: str) -> go.Figure:
    df = (
        priced_universe_t0.copy()
        .assign(Type_Order=lambda x: x["Type"].map({"P": 0, "C": 1}))
        .sort_values(["Type_Order", "Strike"])
        .drop(columns="Type_Order")
        .reset_index(drop=True)
    )
    colors = ["crimson" if oid == target_option_id_t0 else "steelblue" for oid in df["Option_ID"]]

    fig = go.Figure(
        go.Bar(
            x=df["Option_ID"],
            y=df["IV_Diff"],
            marker_color=colors,
            hovertemplate="Contract: %{x}<br>IV Diff: %{y:.6f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line_width=1, line_color="gray")
    fig.update_layout(
        title=f"t = {window_start} IV Difference in Calibration Universe",
        template="plotly_white",
        xaxis_title="Contract",
        yaxis_title="IV Difference",
        height=420,
        showlegend=False,
    )
    return fig


def _format_metric(val: float) -> str:
    return "N/A" if pd.isna(val) or np.isinf(val) else f"{val:.3f}"


def compute_return_metrics(df: pd.DataFrame, initial_gross_exposure: float, periods_per_year: int = 252):
    capital = float(initial_gross_exposure)
    returns = pd.to_numeric(df["pnl_incremental"], errors="coerce").fillna(0.0) / capital
    cumulative_returns = returns.cumsum()

    mean_ret = returns.mean()
    std_ret = returns.std(ddof=1)
    downside_std = returns[returns < 0].std(ddof=1)

    sharpe = np.nan if std_ret <= 0 or pd.isna(std_ret) else np.sqrt(periods_per_year) * mean_ret / std_ret
    sortino = np.nan if pd.isna(downside_std) or downside_std <= 0 else np.sqrt(periods_per_year) * mean_ret / downside_std

    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns - running_max
    max_drawdown = drawdown.min()

    annualized_return = mean_ret * periods_per_year
    calmar = np.nan if pd.isna(max_drawdown) or max_drawdown >= 0 else annualized_return / abs(max_drawdown)

    return {
        "returns": returns,
        "cumulative_returns": cumulative_returns,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


def make_strategy_dashboard_figure(state_df_reduced: pd.DataFrame, initial_gross_exposure: float):
    df = state_df_reduced.copy().reset_index(drop=True)
    metrics = compute_return_metrics(df, initial_gross_exposure)
    returns_cum = metrics["cumulative_returns"]

    target_option_id = df["target_option_id"].dropna().iloc[0] if df["target_option_id"].notna().any() else None
    target_weight_col = f"w_{target_option_id}" if target_option_id is not None else None
    weight_cols = [
        col for col in df.columns
        if col.startswith("w_") and col != "w_underlying" and col != target_weight_col
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Hedge Weights",
            "Target Market vs Theoretical Price",
            "Target Market vs Theoretical IV",
            "Cumulative Return",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )

    # Weights panel.
    weight_plot_cols = weight_cols.copy()
    if "w_underlying" in df.columns:
        weight_plot_cols.append("w_underlying")
    if target_weight_col is not None and target_weight_col in df.columns:
        weight_plot_cols.append(target_weight_col)

    for col in weight_cols:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], mode="lines", showlegend=False, hoverinfo="skip", name=col),
            row=1,
            col=1,
        )
    if "w_underlying" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["w_underlying"], mode="lines", line=dict(width=2), hoverinfo="skip", name="Δ Hedge"),
            row=1,
            col=1,
        )
    if target_weight_col is not None and target_weight_col in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[target_weight_col], mode="lines", line=dict(width=3), hoverinfo="skip", name=target_weight_col),
            row=1,
            col=1,
        )

    hover_text_weights = []
    for i in df.index:
        entries = [f"t={i}"]
        for col in weight_plot_cols:
            val = df.loc[i, col]
            if pd.notna(val) and abs(val) > 1e-12:
                entries.append(f"{col}: {float(val):.6f}")
        hover_text_weights.append("<br>".join(entries))

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[0.0] * len(df),
            mode="lines",
            line=dict(width=18, color="rgba(0,0,0,0)"),
            showlegend=False,
            hovertemplate="%{text}<extra></extra>",
            text=hover_text_weights,
            name="weights_hover",
        ),
        row=1,
        col=1,
    )

    # Price panel.
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["mkt_price_target"],
            mode="lines",
            name="Mkt Price",
            hovertemplate="t=%{x}<br>Market Price: %{y:.6f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["theo_price_target"],
            mode="lines",
            line=dict(dash="dash"),
            name="Theo Price",
            hovertemplate="t=%{x}<br>Theo Price: %{y:.6f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # IV panel.
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["mkt_iv_target"],
            mode="lines",
            name="Mkt IV",
            hovertemplate="t=%{x}<br>Market IV: %{y:.6f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["theo_iv_target"],
            mode="lines",
            line=dict(dash="dash"),
            name="Theo IV",
            hovertemplate="t=%{x}<br>Theo IV: %{y:.6f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Cumulative return panel.
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=returns_cum,
            mode="lines",
            name="Cum Ret",
            hovertemplate="t=%{x}<br>Cumulative Return: %{y:.6%}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=900,
        title="Strategy Dashboard",
        legend=dict(
            orientation="h", yanchor="middle", y=0.5, xanchor="center", x=0.5,
        ),
    )

    fig.update_xaxes(title_text="Time Index", row=2, col=1)
    fig.update_xaxes(title_text="Time Index", row=2, col=2)
    fig.update_yaxes(title_text="Weight", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=2)
    fig.update_yaxes(title_text="Implied Volatility", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=2, col=2, tickformat=".1%")
    return fig, metrics


# -----------------------------
# Streamlit UI
# -----------------------------
def reset_defaults():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.pop("analysis_result", None)


def main():
    st.set_page_config(page_title="Volatility Arbitrage Simulator", layout="wide")
    st.title("Volatility Arbitrage Research Dashboard")
    st.caption("Bates market generation, Heston trader calibration, hedged target-contract diagnostics.")

    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)

    with st.sidebar:
        st.header("Controls")
        st.number_input("Input seed", min_value=0, step=1, key="seed")
        st.caption("Seed controls the randomness of the simulated market. Change for new paths and market scenarios.")
        st.subheader("Market Model Parameters")
        st.slider("κ (mean reversion speed)", min_value=1.0, max_value=15.0, step=0.1, key="kappa")
        st.slider("θ (long-term variance)", min_value=0.01, max_value=0.15, step=0.001, key="theta", format="%.3f")
        st.slider("ξ (volatility of volatility)", min_value=0.05, max_value=0.80, step=0.005, key="xi", format="%.3f")
        st.slider("ρ (correlation)", min_value=-0.95, max_value=0.0, step=0.01, key="rho")
        st.slider("σ_ε (market price noise)", min_value=0.0, max_value=0.03, step=0.001, key="noise_scale", format="%.3f")
        st.toggle("Jump diffusion", key="jump_on")
        st.slider("λ (jump intensity)", min_value=0.0, max_value=3.0, step=0.01, key="lambdaj", disabled=not st.session_state["jump_on"])
        st.slider("μⱼ (jump mean)", min_value=-0.30, max_value=0.0, step=0.005, key="muj", format="%.3f", disabled=not st.session_state["jump_on"])
        st.slider("σⱼ (jump std)", min_value=0.0, max_value=0.30, step=0.005, key="sigmaj", format="%.3f", disabled=not st.session_state["jump_on"])
        st.subheader("Strategy Window")
        st.slider("Time to expiry (trading days)", min_value=20, max_value=60, step=1, key="use_last_n")
        st.caption("CAUTION: Analysis on a 60-day window takes up to ~11 minutes. It is recommended to clone the repository and run locally for faster execution.")
        st.button("Reset to defaults", use_container_width=True, on_click=reset_defaults)
        run_clicked = st.button("Run analysis", type="primary", use_container_width=True)

    if not run_clicked and "analysis_result" not in st.session_state:
        st.info("Set parameters in the sidebar, then click **Run analysis**.")
        return

    if run_clicked:
        with st.spinner("Running full pricing, calibration, and hedging analysis..."):
            st.session_state["analysis_result"] = run_analysis_cached(
                seed=st.session_state["seed"],
                use_last_n=st.session_state["use_last_n"],
                kappa=st.session_state["kappa"],
                theta=st.session_state["theta"],
                xi=st.session_state["xi"],
                rho=st.session_state["rho"],
                jump_on=st.session_state["jump_on"],
                lambdaj=st.session_state["lambdaj"],
                muj=st.session_state["muj"],
                sigmaj=st.session_state["sigmaj"],
                noise_scale=st.session_state["noise_scale"],
                exit_days_before_expiry=st.session_state["exit_days_before_expiry"],
                calibration_N=st.session_state["calibration_N"],
                pricing_N=st.session_state["pricing_N"],
                alpha=st.session_state["alpha"],
                eta=st.session_state["eta"],
                n_each_side=st.session_state["n_each_side"],
                dt=st.session_state["dt"],
                r=st.session_state["r"],
            )

    result = st.session_state.get("analysis_result")
    if result is None:
        return

    S_path = result["S_path"]
    v_path = result["v_path"]
    priced_universe_t0 = result["priced_universe_t0"]
    state_df_reduced = result["state_df_reduced"]
    initial_gross_exposure = result["initial_gross_exposure"]
    target_row_t0 = result["target_row_t0"]
    target_option_id_t0 = result["target_option_id_t0"]

    # Summary metrics.
    window_start = max(0, len(S_path) - st.session_state["use_last_n"])
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    summary_col1.metric("Target Contract", str(target_option_id_t0))
    summary_col2.metric("Target Position", "Long" if result["target_position_t0"] > 0 else "Short")
    summary_col3.metric("Initial Gross Exposure", f"${initial_gross_exposure:.2f}")
    summary_col4.metric(f"t={window_start} IV Diff", f"{float(target_row_t0['IV_Diff']):.6f}")

    # Row 1
    st.plotly_chart(make_path_figure(S_path, v_path, st.session_state["use_last_n"]), use_container_width=True)

    # Row 2
    row2_left, row2_right = st.columns([2, 1])
    with row2_left:
        st.plotly_chart(make_iv_smile_figure(window_start, priced_universe_t0), use_container_width=True)
    with row2_right:
        st.plotly_chart(make_target_iv_diff_figure(window_start, priced_universe_t0, target_option_id_t0), use_container_width=True)

    # Row 3
    strategy_fig, strategy_metrics = make_strategy_dashboard_figure(state_df_reduced, initial_gross_exposure)
    st.plotly_chart(strategy_fig, use_container_width=True)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Sharpe Ratio", _format_metric(strategy_metrics["sharpe"]))
    metric_col2.metric("Max Drawdown", "N/A" if pd.isna(strategy_metrics["max_drawdown"]) else f"{strategy_metrics['max_drawdown']:.3%}")
    metric_col3.metric("Sortino Ratio", _format_metric(strategy_metrics["sortino"]))
    metric_col4.metric("Calmar Ratio", _format_metric(strategy_metrics["calmar"]))

    with st.expander(f"Show t = {window_start} calibration-universe data"):
        display_cols = [
            "Option_ID", "Type", "Strike", "Market_Price", "Market_IV", "Theo_Price", "Theo_IV", "IV_Diff"
        ]
        st.dataframe(
            priced_universe_t0[display_cols].sort_values(["Type", "Strike"]).reset_index(drop=True),
            use_container_width=True,
        )

    with st.expander("Show reduced state dataframe"):
        st.dataframe(state_df_reduced, use_container_width=True)


if __name__ == "__main__":
    main()

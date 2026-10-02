"""Performance of a zero-capital strategy, normalized by initial gross exposure."""
import numpy as np
import pandas as pd


def compute_performance_metrics(df, initial_gross_exposure, periods_per_year=252):
    """Use post-inception P&L periods and a zero benchmark/Sortino target.

    Normalized P&L is additive, not a compounded return on invested capital.
    Downside deviation is sqrt(mean(min(x, 0)**2)) over *all* holding periods.
    Drawdown is peak-to-trough normalized P&L with an initial zero baseline;
    max_drawdown is reported as a positive loss magnitude. Annualized statistics
    are unavailable for irregular intervals, when T is supplied, or insufficient
    observations/zero denominators. Missing P&L is an error, never a zero return.
    """
    if initial_gross_exposure is None or not np.isfinite(initial_gross_exposure) or initial_gross_exposure <= 0:
        raise ValueError("Initial gross exposure must be finite and positive.")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive.")
    pnl = pd.to_numeric(df['pnl_incremental'], errors='raise').astype(float)
    if not np.isfinite(pnl).all():
        raise ValueError("Performance metrics require finite P&L observations.")
    if len(pnl) and pnl.iloc[0] != 0:
        raise ValueError("Performance metrics require the zero-P&L inception row.")
    normalized = pnl / float(initial_gross_exposure)
    cumulative = normalized.cumsum()
    periods = normalized.iloc[1:]
    drawdown = cumulative - cumulative.cummax().clip(lower=0)
    max_drawdown = max(0.0, float(-drawdown.min())) if len(drawdown) else np.nan
    regular = True
    if 'T' in df:
        maturities = pd.to_numeric(df['T'], errors='raise').to_numpy(float)
        if not np.isfinite(maturities).all() or np.any(np.diff(maturities) >= 0):
            raise ValueError("Performance metrics require finite, strictly decreasing maturities.")
        regular = bool(np.allclose(-np.diff(maturities), 1 / periods_per_year, rtol=1e-7, atol=1e-12))
    mean = float(periods.mean()) if len(periods) else np.nan
    variable = len(periods) >= 2 and bool((periods != periods.iloc[0]).any())
    std = float(periods.std(ddof=1)) if variable else np.nan
    downside = float(np.sqrt(np.mean(np.minimum(periods.to_numpy(), 0)**2))) if len(periods) else np.nan
    annualized = mean * periods_per_year if regular else np.nan
    sharpe = np.sqrt(periods_per_year) * mean / std if regular and std > 0 else np.nan
    sortino = np.sqrt(periods_per_year) * mean / downside if regular and downside > 0 else np.nan
    ratio = annualized / max_drawdown if max_drawdown > 0 else np.nan
    return dict(
        normalized_pnl=normalized, cumulative_normalized_pnl=cumulative,
        period_count=len(periods), regular_intervals=regular,
        mean_normalized_pnl=mean, downside_deviation=downside,
        annualized_normalized_pnl=annualized, sharpe=sharpe, sortino=sortino,
        drawdown=drawdown, max_drawdown=max_drawdown,
        annualized_pnl_to_drawdown=ratio,
    )

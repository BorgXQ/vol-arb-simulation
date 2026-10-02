# Volatility Arbitrage Simulation Dashboard

A Python and Streamlit simulation dashboard for studying option-model disagreement and dynamic hedging. A hidden simulated Heston-plus-jumps process drives the underlying, a Bates model generates market quotes, and a calibrated Heston trader takes positions based on implied-volatility differences. Profitability is an experimental outcome, not a promised consequence of disagreement.

Check out the **[demo](https://vol-arb-simulation-borgxq.streamlit.app/)**

(NOTE: Long windows require many calibrations. The higher-accuracy pricing settings increase runtime; running locally is recommended).

## Features

- **Underlying Market Simulation**: Bates model (Heston + jump diffusion) for asset price path generation
- **Synthetic Options Market**: Variable Bates model; Black-Scholes inversion for market implied volatility (IV)
- **Trader Model**: Heston model calibration (via Carr-Madan FFT) for theoretical option pricing and IV
- **Volatility Arbitrage Strategy**: Selectable gamma–delta or gamma–delta–variance hedging for the selected target option
- **Diagnostics**: PnL & hedge dynamics tracking; exposure-normalized P&L, zero-benchmark Sharpe/Sortino, drawdown, and annualized P&L-to-drawdown ratio

## Key Concepts Demonstrated

- Volatility vs price-based trading
- Model risk (Bates vs Heston mismatch)
- Implied volatility surfaces & skew
- Greeks-based hedging:
  - Delta
  - Gamma
  - Variance sensitivity (`dV/dv₀`)
- Calibration vs direct simulation
- Path dependency in PnL

## 📁 Project Structure

```
📈 vol-arb-simulation/
├── 🖥️ .streamlit/
│   └── 📄 config.toml        # Streamlit configuration
├── 📂 src/
│   ├── 📚 calc.py            # Core pricing & IV calculations
│   ├── 📊 metrics.py         # Shared performance calculations
│   ├── 🛠️ utils.py           # Simulation + plotting utilities
│   └── 🧠 vol_arb.py         # Strategy logic (calibration, hedging, PnL)
├── 📂 tests/                # Numerical and behavioral regression checks
├── 📄 CHANGES.md            # Rationale and equations for fixes 1–7
├── 🚀 app.py                 # Streamlit application
├── 📝 test.ipynb             # Development notebook
├── 📦 requirements.txt       # Python dependencies
└── 📖 README.md              # This file
```

## Installation

This project was developed and tested with **Python 3.11.9**.

```bash
git clone https://github.com/BorgXQ/vol-arb-simulation.git
cd vol-arb-simulation

pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Models, controls, and time conventions

| Layer | What it does | What controls it |
| --- | --- | --- |
| Underlying path | Simulates a physical Heston volatility process with lognormal jumps | Seed; the generator uses its separate fixed parameter defaults |
| Option market | Prices non-dividend European options using Bates dynamics | Sidebar κ, θ, ξ, ρ and market jump parameters; current variance is `max(1.05 * simulated variance, 1e-10)` |
| Trader | Calibrates a Heston model to current quotes and hedges a fixed target | Calibration determines its parameters; the hedge-mode selector determines the neutralized risks |

The path generator defaults are `S0=100`, `mu=0.03`, `v0=0.03514`, `kappa=11.31`, `theta=0.05167`, `xi=0.2459`, `rho=-0.6833`, and jump intensity/mean/std `0.7/-0.02/0.04`. These differ from the quote-model defaults shown in the sidebar. Turning off **Jumps in option-market pricing** removes jumps from the quote model, not from the underlying simulation. The trader observes current quotes and spot, not future path values or the true generating parameters. Quote noise is disabled. The generator argument `v0_m` is a compatibility placeholder and does not override the market variance derived from `v_path`.

The app's **Initial time to expiry** slider now means actual trading days: 11–30, default 11. Internally, `use_last_n = initial_tte_days + 1` counts observations, including inception and expiry. Thus 30 days uses 31 observations. The fixed exit threshold is 10 trading days remaining: 11-day expiry gives one holding interval, while 30 gives 20. Time uses `dt=1/252`. A single holding interval is insufficient to estimate Sharpe, so that metric is `N/A` at the minimum window. The 11-day minimum permits a trade before exit; it is not a guarantee that every parameter choice passes numerical or hedge checks. Longer windows require more recalibrations.

The lower-level API retains the observation-count argument `use_last_n`; the notebook derives it explicitly from `initial_tte_days`. Stored app results include their original settings. Editing sidebar controls shows a pending-change notice; plots, time labels, and displayed run details stay tied to the completed run until **Run analysis** is clicked again. Reset clears the displayed run and restores defaults.

Variance sensitivity means `dV/dv₀`, not Black–Scholes vega. The legacy internal `Vega` fields remain compatibility aliases. The three-model design intentionally preserves model disagreement; quote consistency, local hedge neutrality, and passing numerical checks do not imply a correctly identified fair value or guaranteed realized profits.

## Numerical Pricing and Calibration

Market generation, Heston calibration, and trader valuation share the defaults in `src/calc.py`: `N=16384`, `alpha=1.5`, and `eta=0.25`, with cubic strike interpolation. The app and notebook use the same settings throughout. Synthetic quotes retain their FFT configuration in DataFrame metadata; calibration and valuation reject a different configuration when that metadata is present. For externally supplied quotes without metadata, the caller must ensure numerical consistency. The legacy `calibration_N` strategy argument, if supplied, must equal `pricing_N`.

Calibration enforces positive mean reversion, long-run variance and volatility of variance, correlation strictly between -1 and 1, and nonnegative initial variance. The Feller condition is not imposed: it is sufficient for strictly positive variance, but its violation does not invalidate Heston pricing. If bounded Nelder–Mead fails to converge, a scaled, bounded least-squares solver retries from its best finite iterate, using the same price-error objective and FFT settings. Invalid or nonfinite results are rejected; if both solvers fail, the error includes the strategy time index and remaining maturity. Convergence does not imply a good fit to a different model's quotes or unique parameter recovery.

Run the regression checks with:

```bash
python -B -m unittest discover -s tests -v
```

The pricing tests compare 32 Heston/Bates scenarios with direct adaptive integration and a doubled FFT grid. They cover 1, 11, 30 and 60 trading days, strikes from 70% to 130% of spot, and low-variance/high-volatility-of-variance stress cases. The absolute price tolerance is `5e-6 * spot / 100`. A noiseless Heston calibration test uses finer-grid quotes and requires repriced options to agree within `$0.0001` and IVs within `0.0001`. These are regression tolerances for the tested cases, not guarantees for arbitrary parameters or expiry. Recheck convergence when extending the domain or changing numerical settings. Previously saved notebook outputs should be regenerated.

## Synthetic Market Quotes

Quote noise is disabled in the app and notebook to isolate market/trader model disagreement. The generator defaults to `noise_scale=0` and explicitly rejects nonzero values. Calls must satisfy `max(S - K*exp(-r*T), 0) <= C <= S`; only violations within the numerical pricing tolerance are corrected. Puts are derived from the final calls using `P = C - S + K*exp(-r*T)`, without separate clipping. This also enforces the corresponding European put bounds.

Each strike slice is checked for monotonicity, vertical-spread bounds, and convexity, allowing only the stated numerical tolerance. Material inconsistencies raise errors identifying the time index. Expiry is represented as `T=0` with exact intrinsic payoffs; implied volatility is undefined there. Quote regression tests cover discounted bounds, parity, cross-strike checks, positive/zero/negative rates, and market generation for seeds 1, 42, and 67 over 20-, 30-, and 60-observation windows.

## Greek and Hedge Validation

The dashboard explicitly selects **gamma–delta–variance** (the existing default) or **gamma–delta** hedging. The latter leaves variance exposure unconstrained and reports it. The API and notebook use `hedge_mode="gamma_delta_variance"` or `"gamma_delta"`. Variance sensitivity means `dV/dv₀` with other Heston parameters fixed; the `Vega`, `vega_*`, and `net_vega` fields remain compatibility aliases and are not Black–Scholes volatility vega.

Greeks use second-order finite differences with Richardson extrapolation. Near zero variance, a forward stencil replaces the invalid clipped central difference. Each slice compares extrapolated Greeks with halved bumps and a doubled FFT grid. The accepted difference is at most `0.005 * abs(reference)` plus an absolute tolerance of `2e-5` for delta/gamma or `2e-3` for variance sensitivity. Returned prices and Greeks use the configured grid; the doubled grid is a validation reference. A failed check stops the run with the time index. These extra valuations add work per rebalance; calibration remains unchanged.

The hedge solver scales the constraint rows before least squares and discards singular values below `1e-3` of the largest. Rank-deficient systems are accepted only if the requested exposures can still be neutralized. It explicitly checks scaled residuals against `1e-6` per unit target and rejects any hedge option exceeding 10 units or gross hedge options exceeding 20 units per unit target. These are conservative simulation policy limits, not estimated optimal limits. An unstable hedge stops the run rather than silently changing the selected constraints.

The dashboard's **Hedge validation** table includes remaining exposures, matrix rank and condition number, gross hedge positions, residuals, and Greek convergence errors. Tests compare Greeks with adaptive price integration at 11, 30, and 60 trading days, including zero initial variance; verify hedge exposures with finer repricing; and exercise ill-conditioned, oversized, missing, and invalid hedge inputs. Passing these checks establishes local model-risk neutralization, not protection against jumps, discrete rebalancing, or model disagreement.

## Cash Account and P&L

The strategy now uses a self-financing cash account with **zero initial capital**. At entry, net purchases create a negative cash balance (borrowing); net sales create positive cash. At each subsequent observation, the previous cash balance earns or pays the previous observation's continuously compounded pricing rate `r`, over `T_previous - T_current`. The simulation uses years of 252 trading days. Borrowing and lending use the same rate; cash and short-sale proceeds are unrestricted, with no margin, stock-borrow fees, dividends, transaction costs, or slippage.

After interest, trades settle at the current market quotes. For option positions `q` and underlying position `u`:

```text
interest_t = cash_previous * (exp(r_previous * elapsed_years) - 1)
cash_t = cash_previous + interest_t - sum((q_t - q_previous) * option_price_t)
         - (u_t - u_previous) * spot_t
equity_t = cash_t + sum(q_t * option_price_t) + u_t * spot_t
incremental_PnL_t = equity_t - equity_previous
cumulative_PnL_t = equity_t  # initial equity is zero
```

Entry and rebalancing exchange cash for holdings without creating equity. On the first observation meeting the exit threshold, all positions close at that observation's quotes. Its holding-period price P&L and cash interest are included. The closed account then freezes: subsequent idle observations add no strategy interest or P&L. The app and notebook report through this liquidation row, including its market prices and zero ending positions. Missing quotes for held contracts raise an error rather than silently omitting their P&L.

The dashboard's **Cash account and financing** table exposes cash, holdings value, equity, trade cash flows, price P&L, financing, and total P&L from the full-precision state. Initial gross exposure remains a normalization measure, not deposited capital; the displayed normalized performance is not return on equity.

Regression checks use a hand-calculated entry, rebalance, and exit example with both borrowing and lending, changing rates, unequal time intervals, and frozen post-exit cash. Zero-rate checks recover holdings-only P&L. Strategy-level checks verify exit quotes, position liquidation, and inclusion of the exit row in the reporting window.

## Performance Metrics

The app and notebook share `src/metrics.py`. The reduced state retains full numerical precision; rounding happens only when displaying values. Previously saved notebook outputs must be regenerated.

For each holding period, `x = incremental P&L / initial gross exposure`, including financing. The inception row stays in plots as the zero baseline but is excluded from period statistics; the exit-day observation is included. Cumulative normalized P&L is the **sum** of these observations, not a compounded investment return.

- **Sharpe (zero benchmark):** `sqrt(252) * mean(x) / sample_std(x)`.
- **Sortino (zero target):** `sqrt(252) * mean(x) / sqrt(mean(min(x, 0)^2))`. The downside mean uses all holding periods, including zeros on non-losing periods. Repeated identical losses therefore have nonzero downside deviation.
- **Max drawdown / initial exposure:** the largest decline in cumulative normalized P&L from its running peak, including the initial zero baseline. Displayed as a positive loss magnitude; it is not a percentage decline from peak equity.
- **Annualized P&L / max drawdown:** `252 * mean(x) / max_drawdown`. This arithmetic annualization replaces the misleading Calmar label; it does not use compounded annual growth.

The zero benchmark/target is explicit for this zero-initial-capital, self-financing experiment; no additional risk-free return is subtracted from the already financed P&L. Zero denominators and insufficient observations produce `N/A`, not infinite ratios. Invalid exposure or missing/nonfinite P&L raises an error instead of silently adding zero observations. When supplied maturities have irregular intervals, annualized statistics are unavailable rather than assuming each observation is one trading day. Tests cover hand-calculated ratios, first-period drawdown, repeated losses, degenerate cases, full-precision small P&L, and agreement between both plotting paths.

## Notes

- This project is **educational and demonstrative**, not a production trading system
- No transaction costs or slippage are modeled
- The option universe is fixed at the start of the selected trading window: positive strikes spaced $2 apart, spanning at least ±30% of the initial spot (with a minimum half-width of $6), rounded outward to the grid. Future prices do not affect contract availability. Large subsequent moves can leave fewer OTM hedge contracts on one side.
- Annualized statistics from short simulated paths are unstable and depend on the idealized trading and financing assumptions

## Future Improvements

- Introduce transaction costs and bid-ask spreads
- Multi-path simulation (Monte Carlo backtesting)
- More robust calibration techniques
- Alternative models (SABR, local volatility)
- Parallelization for faster execution

## License

MIT

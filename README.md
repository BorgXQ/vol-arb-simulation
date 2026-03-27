# Volatility Arbitrage Simulation Dashboard

An end-to-end quantitative finance project that simulates and visualizes a **volatility arbitrage strategy** using stochastic volatility and jump diffusion models.

Built with **Python + Streamlit**, this project demonstrates how a trader can exploit discrepancies between **market implied volatility** and **model-implied volatility** through dynamic hedging.

---

## Overview

This project simulates the full pipeline of a volatility trading workflow:

1. **Underlying Market Simulation**

   * Generates asset price paths using a **Bates model** (Heston + jump diffusion)

2. **Synthetic Options Market**

   * Prices options across strikes and time using a (possibly different) market model
   * Computes **market implied volatility (IV)** via Black-Scholes inversion

3. **Trader Model**

   * Calibrates a **Heston model** to a subset of market options
   * Produces **theoretical prices and implied volatility**

4. **Volatility Arbitrage Strategy**

   * Identifies the most mispriced option (IV difference)
   * Takes a position (long/short)
   * Applies:

     * **Gamma-Vega hedging (options)**
     * **Delta hedging (underlying)**

5. **Backtesting & Diagnostics**

   * Tracks PnL and hedge dynamics over time
   * Computes performance metrics (Sharpe, Sortino, MDD, Calmar)

---

## Streamlit Dashboard

The interactive app allows users to explore how model assumptions impact trading outcomes.

### Sidebar Controls

* Random seed (market scenario generator)
* Market model parameters:

  * Heston: $\kappa$, $\theta$, $\xi$, $\rho$
  * Jump diffusion: $\lambda$, $\mu_j$, $\sigma_j$
* Market noise level
* Time to expiry
* Run / Reset controls

---

### Main Dashboard

#### **Row 1 — Market Simulation**

* Dual-axis plot:

  * Asset price
  * Instantaneous volatility
* Visualizes the simulated “true” market path

#### **Row 2 — Volatility Surface Diagnostics**

* **Left:** Market IV smile vs trader model IV (t = 0)
* **Right:** IV mispricing across contracts

  * Highlights selected **target option**

#### **Row 3 — Strategy Performance**

* Hedge weights over time
* Market vs theoretical price (target)
* Market vs theoretical IV (target)
* Cumulative returns with metrics:

  * Sharpe Ratio
  * Sortino Ratio
  * Maximum Drawdown
  * Calmar Ratio

## Key Concepts Demonstrated

* Volatility vs price-based trading
* Model risk (Bates vs Heston mismatch)
* Implied volatility surfaces & skew
* Greeks-based hedging:

  * Delta
  * Gamma
  * Vega
* Calibration vs direct simulation
* Path dependency in PnL

---

## 📁 Project Structure

```
📈 vol-arb-simulation/
├── 🖥️ .streamlit/
│   └── 📄 config.toml        # Streamlit configuration
├── 📂 src/
│   ├── 📚 calc.py            # Core pricing & IV calculations
│   ├── 🛠️ utils.py           # Simulation + plotting utilities
│   └── 🧠 vol_arb.py         # Strategy logic (calibration, hedging, PnL)
├── 🚀 app.py                 # Streamlit application
├── 📝 test.ipynb             # Development notebook
├── 📦 requirements.txt       # Python dependencies
└── 📖 README.md              # This file
```

---

## Installation

```bash
git clone https://github.com/your-username/vol-arb-simulation.git
cd volatility-arbitrage-dashboard

pip install -r requirements.txt
```

---

## Run the App

```bash
streamlit run app.py
```

---

## Notes

* This project is **educational and demonstrative**, not a production trading system
* No transaction costs or slippage are modeled
* Performance metrics may appear inflated due to:

  * small return magnitudes
  * idealized hedging assumptions

---

## Future Improvements

* Introduce transaction costs and bid-ask spreads
* Multi-path simulation (Monte Carlo backtesting)
* More robust calibration techniques
* Alternative models (SABR, local volatility)
* Parallelization for faster execution

---

## License

MIT
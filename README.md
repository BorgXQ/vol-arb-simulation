# Volatility Arbitrage Simulation Dashboard

An end-to-end quantitative finance project that simulates and visualizes a volatility arbitrage strategy using stochastic volatility and jump diffusion models. Built with Python + Streamlit, this project demonstrates how a trader can exploit discrepancies between market implied volatility and model-implied volatility through dynamic hedging.

Check out the **[demo](https://vol-arb-simulation-borgxq.streamlit.app/)**

(NOTE: The analysis for a 60-day window takes ~11 minutes to run on Streamlit; it is highly recommended to clone the repository and run it locally instead).

## Features

- **Underlying Market Simulation**: Bates model (Heston + jump diffusion) for asset price path generation
- **Synthetic Options Market**: Variable Bates model; Black-Scholes inversion for market implied volatility (IV)
- **Trader Model**: Heston model calibration (via Carr-Madan FFT) for theoretical option pricing and IV
- **Volatility Arbitrage Strategy**: Gamma-Vega hedging (options) and Delta hedging (underlying) for most mispriced option
- **Diagnostics**: PnL & hedge dynamics tracking; performance metrics (Sharpe, Sortino, MDD, Calmar) computation

## Key Concepts Demonstrated

- Volatility vs price-based trading
- Model risk (Bates vs Heston mismatch)
- Implied volatility surfaces & skew
- Greeks-based hedging:
  - Delta
  - Gamma
  - Vega
- Calibration vs direct simulation
- Path dependency in PnL

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

## Installation

This project was developed and tested with **Python 3.11.9**.

```bash
git clone https://github.com/your-username/vol-arb-simulation.git
cd volatility-arbitrage-dashboard

pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```
## Notes

- This project is **educational and demonstrative**, not a production trading system
- No transaction costs or slippage are modeled
- Performance metrics may appear inflated due to small return magnitudes and idealized hedging assumptions

## Future Improvements

- Introduce transaction costs and bid-ask spreads
- Multi-path simulation (Monte Carlo backtesting)
- More robust calibration techniques
- Alternative models (SABR, local volatility)
- Parallelization for faster execution

## License

MIT
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brute, least_squares, minimize


# Shared by market generation, calibration, valuation, and finite differences.
# Validated against direct integration in tests/test_pricing_consistency.py.
DEFAULT_FFT_N = 16384
DEFAULT_FFT_ALPHA = 1.5
DEFAULT_FFT_ETA = 0.25


def validate_fft_config(N, alpha, eta):
    """Validate and return a comparable FFT configuration tuple."""
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)):
        raise ValueError("FFT N must be an integer power of two, at least 64.")
    if N < 64 or N & (N - 1):
        raise ValueError("FFT N must be an integer power of two, at least 64.")
    if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(eta) or eta <= 0:
        raise ValueError("FFT alpha and eta must be finite and positive.")
    return int(N), float(alpha), float(eta)


def validate_market_fft_config(options, N, alpha, eta):
    """Reject inconsistent settings when quotes retain generation metadata."""
    config = validate_fft_config(N, alpha, eta)
    market_config = options.attrs.get("fft_config")
    if market_config is not None and tuple(market_config) != config:
        raise ValueError("Market generation, calibration, and valuation must use the same FFT configuration.")


def _valid_heston_params(params):
    """The Feller condition is sufficient, but not required for Heston pricing."""
    params = np.asarray(params, dtype=float)
    if params.shape != (5,) or not np.all(np.isfinite(params)):
        return False
    kappa, theta, xi, rho, v0 = params
    return kappa > 0 and theta > 0 and xi > 0 and -1 < rho < 1 and v0 >= 0


def H93_char_func_cm(u, S0, v0, kappa_v, theta_v, xi_v, rho, r, T):
    """
    Heston (1993) characteristic function.

    Parameters
    ----------
    u : complex or ndarray
        Frequency argument.
    S0 : float
        Current underlying price.
    v0 : float
        Initial variance.
    kappa_v : float
        Mean-reversion speed of variance.
    theta_v : float
        Long-run variance.
    xi_v : float
        Volatility of variance.
    rho : float
        Correlation between price and variance Brownian motions.
    r : float
        Risk-free rate.
    T : float
        Time to maturity.

    Returns
    -------
    ndarray of complex
    """
    d = np.sqrt((rho * xi_v * 1j * u - kappa_v) ** 2 +
                xi_v ** 2 * (1j * u + u ** 2))
    g = ((kappa_v - rho * xi_v * 1j * u - d) /
         (kappa_v - rho * xi_v * 1j * u + d))

    C = (
        r * 1j * u * T
        + (kappa_v * theta_v / xi_v ** 2)
        * (
            (kappa_v - rho * xi_v * 1j * u - d) * T
            - 2.0 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
        )
    )

    D = (
        (kappa_v - rho * xi_v * 1j * u - d) / xi_v ** 2
        * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    )

    return np.exp(C + D * v0 + 1j * u * np.log(S0))


def Heston_jump_char_func(u, S0, v0, kappa_v, theta_v, xi_v, rho, r, T,
                         lambda_j, mu_j, sigma_j):
    """
    Characteristic function for the Heston model with Merton log-normal jumps.

    Parameters
    ----------
    u : complex or ndarray
        Frequency argument.
    S0 : float
        Current underlying price.
    v0 : float
        Initial variance.
    kappa_v : float
        Mean-reversion speed of variance.
    theta_v : float
        Long-run variance.
    xi_v : float
        Volatility of variance.
    rho : float
        Correlation between price and variance Brownian motions.
    r : float
        Risk-free rate.
    T : float
        Time to maturity.
    lambda_j : float
        Jump arrival intensity.
    mu_j : float
        Mean log-jump size.
    sigma_j : float
        Standard deviation of log-jump size.

    Returns
    -------
    ndarray of complex
    """
    # Jump compensator (CRITICAL)
    kappa_J = lambda_j * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)

    # Heston part
    d = np.sqrt((rho * xi_v * 1j * u - kappa_v) ** 2 +
                xi_v ** 2 * (1j * u + u ** 2))

    g = (kappa_v - rho * xi_v * 1j * u - d) / \
        (kappa_v - rho * xi_v * 1j * u + d)

    # FIX: r → r - kappa_J
    C = (r - kappa_J) * 1j * u * T + (kappa_v * theta_v) / xi_v ** 2 * (
        (kappa_v - rho * xi_v * 1j * u - d) * T
        - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    )

    D = ((kappa_v - rho * xi_v * 1j * u - d) / xi_v ** 2 *
         (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))

    # Jump component
    jump_cf = np.exp(
        lambda_j * T * (
            np.exp(1j * u * mu_j - 0.5 * sigma_j**2 * u**2) - 1
        )
    )

    return np.exp(C + D * v0 + 1j * u * np.log(S0)) * jump_cf


def CM99_call_price_grid_jd_fft(
    S0,
    T,
    r,
    kappa_v,
    theta_v,
    xi_v,
    rho,
    v0,
    lambda_j,
    mu_j,
    sigma_j,
    N=DEFAULT_FFT_N,
    alpha=DEFAULT_FFT_ALPHA,
    eta=DEFAULT_FFT_ETA
):
    """
    Price a full strike grid in a single FFT call using the Heston + jump
    characteristic function (Carr-Madan 1999).

    Parameters
    ----------
    S0 : float
        Current underlying price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    kappa_v, theta_v, xi_v, rho, v0 : float
        Heston variance process parameters.
    lambda_j, mu_j, sigma_j : float
        Merton jump parameters (intensity, mean log-jump, std log-jump).
    N : int, optional
        FFT grid size (default 16384).
    alpha : float, optional
        Carr-Madan damping parameter (default 1.5).
    eta : float, optional
        Frequency grid spacing (default 0.25).

    Returns
    -------
    K_grid : ndarray
        Strike grid.
    call_prices : ndarray
        Call prices on the strike grid.
    """
    validate_fft_config(N, alpha, eta)
    if not _valid_heston_params((kappa_v, theta_v, xi_v, rho, v0)):
        raise ValueError("Invalid Heston parameters: require positive kappa, theta, xi, |rho| < 1, and v0 >= 0.")
    if not np.all(np.isfinite([S0, T, r])) or S0 <= 0 or T <= 0:
        raise ValueError("Pricing requires finite S0, T, r and positive S0 and T.")

    lambda_val = 2 * np.pi / (N * eta)
    b = 0.5 * N * lambda_val

    k_grid = np.arange(N) * lambda_val - b      # log-strike grid
    v_grid = np.arange(N) * eta                 # frequency grid

    phi = Heston_jump_char_func(
        u=v_grid - (alpha + 1) * 1j,
        S0=S0,
        v0=v0,
        kappa_v=kappa_v,
        theta_v=theta_v,
        xi_v=xi_v,
        rho=rho,
        lambda_j=lambda_j,
        mu_j=mu_j,
        sigma_j=sigma_j,
        r=r,
        T=T
    )

    denom = alpha**2 + alpha - v_grid**2 + 1j * (2 * alpha + 1) * v_grid
    psi = np.exp(-r * T) * phi / denom

    weights = np.ones(N)
    weights[0] = 0.5
    weights[-1] = 0.5

    fft_input = np.exp(1j * b * v_grid) * psi * eta * weights
    y = np.fft.fft(fft_input)

    call_prices = np.exp(-alpha * k_grid) / np.pi * np.real(y)
    if not np.all(np.isfinite(call_prices)):
        raise FloatingPointError("Nonfinite FFT prices; check parameters and damping.")
    call_prices = np.maximum(call_prices, 0.0)

    K_grid = np.exp(k_grid)
    return K_grid, call_prices


def CM99_call_price_grid_fft(
    S0,
    T,
    r,
    kappa_v,
    theta_v,
    xi_v,
    rho,
    v0,
    N=DEFAULT_FFT_N,
    alpha=DEFAULT_FFT_ALPHA,
    eta=DEFAULT_FFT_ETA,
):
    """
    Price a full strike grid in a single FFT call using the pure Heston
    characteristic function (Carr-Madan 1999).

    Parameters
    ----------
    S0 : float
        Current underlying price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    kappa_v, theta_v, xi_v, rho, v0 : float
        Heston variance process parameters.
    N : int, optional
        FFT grid size (default 16384).
    alpha : float, optional
        Carr-Madan damping parameter (default 1.5).
    eta : float, optional
        Frequency grid spacing (default 0.25).

    Returns
    -------
    K_grid : ndarray
        Strike grid.
    call_prices : ndarray
        Call prices on the strike grid.
    """
    validate_fft_config(N, alpha, eta)
    if not _valid_heston_params((kappa_v, theta_v, xi_v, rho, v0)):
        raise ValueError("Invalid Heston parameters: require positive kappa, theta, xi, |rho| < 1, and v0 >= 0.")
    if not np.all(np.isfinite([S0, T, r])) or S0 <= 0 or T <= 0:
        raise ValueError("Pricing requires finite S0, T, r and positive S0 and T.")

    lambda_val = 2 * np.pi / (N * eta)
    b = 0.5 * N * lambda_val

    k_grid = np.arange(N) * lambda_val - b      # log-strike grid
    v_grid = np.arange(N) * eta                 # frequency grid

    phi = H93_char_func_cm(
        u=v_grid - (alpha + 1) * 1j,
        S0=S0,
        v0=v0,
        kappa_v=kappa_v,
        theta_v=theta_v,
        xi_v=xi_v,
        rho=rho,
        r=r,
        T=T,
    )

    denom = alpha**2 + alpha - v_grid**2 + 1j * (2 * alpha + 1) * v_grid
    psi = np.exp(-r * T) * phi / denom

    weights = np.ones(N)
    weights[0] = 0.5
    weights[-1] = 0.5

    fft_input = np.exp(1j * b * v_grid) * psi * eta * weights
    y = np.fft.fft(fft_input)

    call_prices = np.exp(-alpha * k_grid) / np.pi * np.real(y)
    if not np.all(np.isfinite(call_prices)):
        raise FloatingPointError("Nonfinite FFT prices; check parameters and damping.")
    call_prices = np.maximum(call_prices, 0.0)

    K_grid = np.exp(k_grid)
    return K_grid, call_prices


def interpolate_call_prices(target_strikes, K_grid, call_grid):
    """
    Interpolate call prices from the FFT strike grid to a set of target strikes.

    Uses cubic interpolation to avoid linear-interpolation pricing bias. Strikes
    outside the grid are rejected rather than assigned an endpoint price.

    Parameters
    ----------
    target_strikes : ndarray
        Strikes at which prices are required.
    K_grid : ndarray
        FFT-produced strike grid.
    call_grid : ndarray
        Call prices on K_grid.

    Returns
    -------
    ndarray
        Interpolated call prices at target_strikes.
    """
    target_strikes = np.asarray(target_strikes, dtype=float)
    if not np.all(np.isfinite(target_strikes)) or np.any(target_strikes <= 0):
        raise ValueError("Target strikes must be finite and positive.")
    if np.any(target_strikes < K_grid[0]) or np.any(target_strikes > K_grid[-1]):
        raise ValueError("Target strikes fall outside the FFT grid; adjust eta.")
    prices = CubicSpline(K_grid, call_grid, extrapolate=False)(target_strikes)
    if not np.all(np.isfinite(prices)):
        raise FloatingPointError("Nonfinite interpolated option prices.")
    return np.maximum(prices, 0.0)


def put_from_call_parity(call_prices, S0, strikes, r, T):
    """
    Derive put prices from call prices via put-call parity.

    Parameters
    ----------
    call_prices : ndarray
    S0 : float
    strikes : ndarray
    r : float
    T : float

    Returns
    -------
    ndarray
    """
    return np.maximum(call_prices - S0 + strikes * np.exp(-r * T), 0.0)


def _heston_price_residuals(p0, options, S0, N, alpha, eta):
    """Signed per-option pricing errors shared by both local solvers."""
    if not _valid_heston_params(p0):
        return np.full(len(options), np.inf)
    kappa_v, theta_v, xi_v, rho, v0 = p0

    # Group by maturity and rate so each group uses one FFT
    residuals = []

    grouped = options.groupby(["T", "r"], sort=False)

    for (T, r), group in grouped:
        try:
            strikes = group["Strike"].to_numpy(dtype=float)
            market_prices = group["Market_Price"].to_numpy(dtype=float)
            types = group["Type"].to_numpy()

            K_grid, call_grid = CM99_call_price_grid_fft(
                S0=S0,
                T=float(T),
                r=float(r),
                kappa_v=kappa_v,
                theta_v=theta_v,
                xi_v=xi_v,
                rho=rho,
                v0=v0,
                N=N,
                alpha=alpha,
                eta=eta,
            )

            model_calls = interpolate_call_prices(strikes, K_grid, call_grid)
            model_puts = put_from_call_parity(
                call_prices=model_calls,
                S0=S0,
                strikes=strikes,
                r=float(r),
                T=float(T),
            )

            model_prices = np.where(types == "C", model_calls, model_puts)
            if not np.all(np.isfinite(model_prices)):
                return np.full(len(options), np.inf)
            residuals.append(model_prices - market_prices)

        except (ValueError, FloatingPointError, OverflowError):
            return np.full(len(options), np.inf)

    return np.concatenate(residuals) if residuals else np.array([np.inf])


def CM99_error_function_vectorized(
    p0,
    options,
    S0,
    N=DEFAULT_FFT_N,
    alpha=DEFAULT_FFT_ALPHA,
    eta=DEFAULT_FFT_ETA,
    _state=None,
):
    """
    Calibration error function: mean squared error between model and market prices.

    Groups options by maturity/rate so each group requires only one FFT call.

    Parameters
    ----------
    p0 : array-like
        Parameter vector (kappa_v, theta_v, xi_v, rho, v0).
    options : pd.DataFrame
        Must contain columns: Strike, Type, T, r, Market_Price.
    S0 : float
        Current underlying price.
    N : int, optional
        FFT grid size (default 16384).
    alpha : float, optional
        Carr-Madan damping parameter (default 1.5).
    eta : float, optional
        Frequency grid spacing (default 0.25).
    _state : dict or None, optional
        Mutable dict for tracking iteration history and progress bar.

    Returns
    -------
    float
        MSE, or infinity for invalid parameters or nonfinite model prices.
    """
    residuals = _heston_price_residuals(p0, options, S0, N, alpha, eta)
    mse = float(np.mean(residuals**2))
    if not np.isfinite(mse):
        return np.inf

    if _state is not None:
        _state["min_MSE"] = min(_state["min_MSE"], mse)
        _state["MSE_history"].append(mse)
        _state["iteration_history"].append(_state["i"])

        pbar = _state.get("pbar")
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                MSE=f"{mse:.6f}",
                best=f"{_state['min_MSE']:.6f}",
                refresh=False,
            )

        _state["i"] += 1

    return mse


def CM99_calibration_market(
    options, S0, N=DEFAULT_FFT_N, alpha=DEFAULT_FFT_ALPHA, eta=DEFAULT_FFT_ETA
):
    """
    Calibrate Heston parameters to market option prices via a two-stage
    brute-force grid search followed by bounded Nelder-Mead local refinement.
    If the simplex does not converge, retry with scaled, bounded nonlinear
    least squares from its best finite iterate (or the grid seed).

    Invalid inputs raise ValueError. Failed optimization or nonfinite final
    prices raise RuntimeError; unsuccessful fits are never silently returned.

    Parameters
    ----------
    options : pd.DataFrame
        Must contain columns: Strike, Type, T, r, Market_Price.
    S0 : float
        Current underlying price.
    N : int, optional
        FFT grid size (default 16384).
    alpha : float, optional
        Carr-Madan damping parameter (default 1.5).
    eta : float, optional
        Frequency grid spacing (default 0.25).

    Returns
    -------
    opt : ndarray
        Calibrated parameters (kappa_v, theta_v, xi_v, rho, v0).
    stage1_end_iter : int
        Number of iterations completed by the brute-force stage.
    MSE_history : list of float
        MSE value at every iteration.
    iteration_history : list of int
        Iteration index corresponding to each MSE entry.
    """
    validate_market_fft_config(options, N, alpha, eta)
    required = {"Strike", "Type", "T", "r", "Market_Price"}
    if not required.issubset(options.columns) or options.empty:
        raise ValueError("Calibration needs nonempty quotes with Strike, Type, T, r, Market_Price.")
    numeric = options[["Strike", "T", "r", "Market_Price"]].to_numpy(dtype=float)
    if not np.isfinite(S0) or S0 <= 0 or not np.all(np.isfinite(numeric)):
        raise ValueError("Calibration requires finite quotes and a positive finite spot.")
    if (options[["Strike", "T"]] <= 0).any().any() or (options["Market_Price"] < 0).any():
        raise ValueError("Calibration requires positive strikes/maturities and nonnegative prices.")
    if not options["Type"].isin(["C", "P"]).all():
        raise ValueError("Calibration option types must be C or P.")

    state = {
        "i": 0,
        "min_MSE": np.inf,
        "MSE_history": [],
        "iteration_history": [],
    }

    def error_func(p0):
        return CM99_error_function_vectorized(
            p0,
            options=options,
            S0=S0,
            N=N,
            alpha=alpha,
            eta=eta,
            _state=state,
        )

    param_grid = (
        (5.0, 15.0, 1.0),    # kappa_v
        (0.01, 0.11, 0.02),  # theta_v
        (0.1, 0.4, 0.05),    # xi_v
        (-0.9, 0.1, 0.2),    # rho
        (0.01, 0.1, 0.02),   # v0
    )

    p0 = brute(error_func, param_grid, finish=None)
    stage1_end_iter = len(state["iteration_history"])
    if not np.isfinite(error_func(p0)):
        raise RuntimeError("Calibration grid search found no finite pricing candidate.")

    lower_bounds = np.array([1e-6, 1e-8, 1e-6, -0.999999, 0.0])
    upper_bounds = np.array([np.inf, np.inf, np.inf, 0.999999, np.inf])
    result = minimize(
        fun=error_func,
        x0=p0,
        method="Nelder-Mead",
        bounds=list(zip(lower_bounds, upper_bounds)),
        options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 4000, "maxfev": 6000},
    )
    if not result.success:
        simplex_message = str(result.message)
        restart = p0
        if _valid_heston_params(result.x) and np.isfinite(error_func(result.x)):
            restart = np.clip(result.x, lower_bounds, upper_bounds)

        def residual_func(params):
            residuals = _heston_price_residuals(params, options, S0, N, alpha, eta)
            mse = float(np.mean(residuals**2))
            if np.isfinite(mse):
                state["min_MSE"] = min(state["min_MSE"], mse)
                state["MSE_history"].append(mse)
                state["iteration_history"].append(state["i"])
                state["i"] += 1
            return residuals

        # Scaling handles disparate parameter units and weakly identified fits.
        # A larger derivative step avoids differencing FFT roundoff. Retain the
        # same squared-price-error objective and the full pricing resolution.
        try:
            retry = least_squares(
                residual_func,
                restart,
                bounds=(lower_bounds, upper_bounds),
                method="trf",
                x_scale="jac",
                diff_step=1e-4,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                max_nfev=1000,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(
                f"Heston calibration failed. Nelder-Mead: {simplex_message}; "
                f"least-squares retry could not run: {exc}"
            ) from exc
        if not retry.success:
            raise RuntimeError(
                f"Heston calibration failed. Nelder-Mead: {simplex_message}; "
                f"least-squares retry: {retry.message}"
            )
        result = retry
    if (
        not _valid_heston_params(result.x)
        or not np.all(np.isfinite(result.fun))
        or not np.isfinite(error_func(result.x))
    ):
        raise RuntimeError("Heston calibration returned invalid parameters or nonfinite prices.")

    return result.x, stage1_end_iter, state["MSE_history"], state["iteration_history"]

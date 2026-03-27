import numpy as np
from scipy.optimize import brute, fmin


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
    N=4096,
    alpha=1.5,
    eta=0.25
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
        FFT grid size (default 4096).
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
    N=4096,
    alpha=1.5,
    eta=0.25,
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
        FFT grid size (default 4096).
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
    call_prices = np.maximum(call_prices, 0.0)

    K_grid = np.exp(k_grid)
    return K_grid, call_prices


def interpolate_call_prices(target_strikes, K_grid, call_grid):
    """
    Interpolate call prices from the FFT strike grid to a set of target strikes.

    Uses linear interpolation; faster and more stable than cubic for this use case.

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
    return np.interp(
        target_strikes,
        K_grid,
        call_grid,
        left=call_grid[0],
        right=call_grid[-1],
    )


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


def CM99_error_function_vectorized(
    p0,
    options,
    S0,
    N=4096,
    alpha=1.5,
    eta=0.25,
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
        FFT grid size (default 4096).
    alpha : float, optional
        Carr-Madan damping parameter (default 1.5).
    eta : float, optional
        Frequency grid spacing (default 0.25).
    _state : dict or None, optional
        Mutable dict for tracking iteration history and progress bar.

    Returns
    -------
    float
        MSE (plus any boundary penalties).
    """
    kappa_v, theta_v, xi_v, rho, v0 = p0

    # Hard constraints
    if kappa_v < 0.0 or theta_v < 0.005 or xi_v < 0.05 or not (-1.0 < rho < 1.0):
        return 500.0
    if 2.0 * kappa_v * theta_v < xi_v**2:
        return 500.0

    # Soft penalties
    boundary_penalty = 0.0
    if abs(rho + 1.0) < 0.01 or abs(rho - 1.0) < 0.01:
        boundary_penalty += 10.0
    if xi_v < 0.08:
        boundary_penalty += 20.0
    if v0 < 0.005:
        boundary_penalty += 5.0

    # Group by maturity and rate so each group uses one FFT
    se_all = []

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
            se_all.append((model_prices - market_prices) ** 2)

        except Exception:
            se_all.append(np.full(len(group), 100.0))

    if se_all:
        se_concat = np.concatenate(se_all)
        mse = float(np.mean(se_concat)) + boundary_penalty
    else:
        mse = 500.0 + boundary_penalty

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


def CM99_calibration_market(options, S0, N=4096, alpha=1.5, eta=0.25):
    """
    Calibrate Heston parameters to market option prices via a two-stage
    brute-force grid search followed by Nelder-Mead local refinement.

    Parameters
    ----------
    options : pd.DataFrame
        Must contain columns: Strike, Type, T, r, Market_Price.
    S0 : float
        Current underlying price.
    N : int, optional
        FFT grid size (default 4096).
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
    state = {
        "i": 0,
        "min_MSE": 500.0,
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

    opt = fmin(
        func=error_func,
        x0=p0,
        xtol=1e-6,
        ftol=1e-6,
        maxiter=750,
        maxfun=900,
        disp=False,
    )

    return opt, stage1_end_iter, state["MSE_history"], state["iteration_history"]
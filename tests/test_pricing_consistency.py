import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.integrate import quad_vec
from scipy.optimize import OptimizeResult, least_squares

from src.calc import (
    DEFAULT_FFT_N,
    DEFAULT_FFT_ALPHA,
    DEFAULT_FFT_ETA,
    CM99_call_price_grid_fft,
    CM99_call_price_grid_jd_fft,
    CM99_calibration_market,
    CM99_error_function_vectorized,
    H93_char_func_cm,
    Heston_jump_char_func,
    interpolate_call_prices,
    put_from_call_parity,
)
from src.utils import append_market_iv, bs_price, generate_market_option_prices_across_time
from src.vol_arb import price_slice_with_heston_and_greeks, run_vol_arb_strategy


PARAMS = np.array([6.4, 0.077, 0.24, -0.7, 0.04])


def model_kwargs(params=PARAMS, spot=100.0, days=30):
    return dict(
        zip(["kappa_v", "theta_v", "xi_v", "rho", "v0"], params),
        S0=spot, T=days / 252, r=0.02,
    )


def direct_call_prices(strikes, params):
    """Independent adaptive integration: no FFT or strike interpolation.

    Carr & Madan (1999), equations (5)-(6):
    https://engineering.nyu.edu/sites/default/files/2018-08/CarrMadan2_0.pdf
    Integrate to infinity, with a different damping value from production.
    """
    damping = 1.0
    log_k = np.log(strikes)

    def integrand(u):
        phi = Heston_jump_char_func(u=u - (damping + 1) * 1j, **params)
        denominator = damping**2 + damping - u**2 + 1j * (2 * damping + 1) * u
        return (
            np.exp(-damping * log_k - params["r"] * params["T"]) / np.pi
            * np.real(np.exp(-1j * u * log_k) * phi / denominator)
        )

    values, error = quad_vec(integrand, 0, np.inf, epsabs=1e-9, epsrel=1e-9)
    if error > 1e-7:
        raise AssertionError(f"Reference integration did not converge: error={error}")
    return values


def noiseless_quotes():
    strikes = np.array([85.0, 90.0, 94.0, 98.0, 102.0, 106.0, 110.0, 115.0])
    params = model_kwargs()
    # Higher-resolution data prevents a same-grid fit from hiding pricing bias.
    grid, prices = CM99_call_price_grid_fft(**params, N=2 * DEFAULT_FFT_N)
    calls = interpolate_call_prices(strikes, grid, prices)
    types = np.where(strikes < params["S0"], "P", "C")
    puts = put_from_call_parity(calls, params["S0"], strikes, params["r"], params["T"])
    return append_market_iv(pd.DataFrame({
        "S_t": params["S0"], "Strike": strikes, "Type": types,
        "T": params["T"], "r": params["r"],
        "Market_Price": np.where(types == "C", calls, puts),
    }))


class PricingAccuracyTests(unittest.TestCase):
    def test_fft_agrees_with_direct_integration_and_finer_grid(self):
        # Includes low variance, strong skew, and a Feller-violating stress case.
        scenarios = [
            (PARAMS, 100.0),
            ([1.0, 0.01, 0.8, -0.95, 0.005], 100.0),
            ([15.0, 0.15, 0.8, 0.0, 0.15], 150.0),
            ([2.0, 0.01, 0.05, -0.3, 0.01], 85.43),
        ]
        for parameters, spot in scenarios:
            strikes = spot * np.array([0.70, 0.85, 0.94, 0.98, 1.0, 1.02, 1.06, 1.15, 1.30])
            tolerance = 5e-6 * spot / 100.0
            for days in [1, 11, 30, 60]:
                for jumps in [(0.0, 0.0, 0.0), (1.62, -0.165, 0.135)]:
                    with self.subTest(params=parameters, spot=spot, days=days, jumps=jumps):
                        params = model_kwargs(parameters, spot, days)
                        params.update(zip(["lambda_j", "mu_j", "sigma_j"], jumps))
                        reference = direct_call_prices(strikes, params)
                        grid, prices = CM99_call_price_grid_jd_fft(**params)
                        actual = interpolate_call_prices(strikes, grid, prices)
                        finer_grid, finer_prices = CM99_call_price_grid_jd_fft(
                            **params, N=2 * DEFAULT_FFT_N
                        )
                        finer = interpolate_call_prices(strikes, finer_grid, finer_prices)
                        np.testing.assert_allclose(actual, reference, atol=tolerance, rtol=0)
                        np.testing.assert_allclose(actual, finer, atol=tolerance, rtol=0)

    def test_no_jump_bates_matches_heston(self):
        params = model_kwargs()
        kg, calls = CM99_call_price_grid_fft(**params)
        kj, jump_calls = CM99_call_price_grid_jd_fft(
            **params, lambda_j=0.0, mu_j=-0.165, sigma_j=0.135
        )
        strikes = np.array([85.0, 100.0, 115.0])
        np.testing.assert_allclose(
            interpolate_call_prices(strikes, kg, calls),
            interpolate_call_prices(strikes, kj, jump_calls), atol=1e-10, rtol=0,
        )

    def test_characteristic_functions_preserve_probability_and_forward(self):
        params = model_kwargs()
        for function, extra in [
            (H93_char_func_cm, {}),
            (Heston_jump_char_func, dict(lambda_j=1.62, mu_j=-0.165, sigma_j=0.135)),
        ]:
            self.assertAlmostEqual(function(u=0.0, **params, **extra), 1.0)
            self.assertAlmostEqual(
                function(u=-1j, **params, **extra),
                params["S0"] * np.exp(params["r"] * params["T"]),
            )

    def test_constant_variance_limit_matches_black_scholes(self):
        params = model_kwargs([2.0, 0.04, 1e-4, 0.0, 0.04])
        strikes = np.array([85.0, 100.0, 115.0])
        grid, prices = CM99_call_price_grid_fft(**params)
        reference = [bs_price(100.0, k, params["T"], 0.02, 0.2, "C") for k in strikes]
        np.testing.assert_allclose(
            interpolate_call_prices(strikes, grid, prices), reference, atol=5e-6, rtol=0
        )

    def test_out_of_grid_strikes_are_rejected(self):
        grid, prices = CM99_call_price_grid_fft(**model_kwargs())
        with self.assertRaisesRegex(ValueError, "outside"):
            interpolate_call_prices([grid[-1] * 2], grid, prices)


class CalibrationTests(unittest.TestCase):
    def test_noiseless_price_recovery(self):
        quotes = noiseless_quotes()
        params, _, history, _ = CM99_calibration_market(quotes, S0=100.0)
        priced = price_slice_with_heston_and_greeks(quotes, 100.0, params)
        # Compare prices, not non-identifiable individual Heston parameters.
        np.testing.assert_allclose(priced.Theo_Price, priced.Market_Price, atol=1e-4, rtol=0)
        self.assertLess(float(np.max(np.abs(priced.IV_Diff))), 1e-4)
        self.assertTrue(np.isfinite(history).all())
        self.assertGreaterEqual(params[-1], 0)

    def test_invalid_parameters_cannot_beat_a_valid_fit(self):
        quotes = noiseless_quotes()
        for index, value in [(0, 0.0), (1, -0.01), (2, 0.0), (3, -1.0), (4, -0.001), (4, np.nan)]:
            params = PARAMS.copy()
            params[index] = value
            with self.subTest(index=index, value=value):
                self.assertEqual(CM99_error_function_vectorized(params, quotes, 100.0), np.inf)
        # Nonnegative variance includes zero, and the Feller condition is optional.
        params = np.array([1.0, 0.01, 0.8, -0.7, 0.0])
        self.assertTrue(np.isfinite(CM99_error_function_vectorized(params, quotes, 100.0)))

    def test_nonfinite_model_prices_are_rejected(self):
        quotes = noiseless_quotes()
        with patch("src.calc.CM99_call_price_grid_fft", return_value=(
            np.array([50.0, 90.0, 110.0, 150.0]), np.full(4, np.nan)
        )):
            self.assertEqual(CM99_error_function_vectorized(PARAMS, quotes, 100.0), np.inf)

    def test_bad_market_data_fails_before_optimization(self):
        for column, value in [("Market_Price", np.nan), ("T", 0.0), ("Type", "X")]:
            quotes = noiseless_quotes()
            quotes.loc[0, column] = value
            with self.subTest(column=column), patch("src.calc.brute") as search:
                with self.assertRaises(ValueError):
                    CM99_calibration_market(quotes, 100.0)
                search.assert_not_called()

    def test_failed_optimizer_is_not_silently_accepted(self):
        result = OptimizeResult(success=False, message="Evaluation limit reached", x=PARAMS, fun=0.0)
        retry = OptimizeResult(success=False, message="Retry limit reached", x=PARAMS)
        with (
            patch("src.calc.brute", return_value=PARAMS),
            patch("src.calc.minimize", return_value=result),
            patch("src.calc.least_squares", return_value=retry),
        ):
            with self.assertRaisesRegex(RuntimeError, "Evaluation limit reached.*Retry limit reached"):
                CM99_calibration_market(noiseless_quotes(), 100.0)

    def test_failed_simplex_retries_from_its_best_iterate(self):
        quotes = noiseless_quotes()
        start = PARAMS * np.array([1.1, 0.9, 1.05, 1.0, 1.1])
        failure = OptimizeResult(success=False, message="Evaluation limit reached", x=start)
        with (
            patch("src.calc.brute", return_value=PARAMS),
            patch("src.calc.minimize", return_value=failure),
            patch("src.calc.least_squares", wraps=least_squares) as retry,
        ):
            params, _, history, _ = CM99_calibration_market(quotes, 100.0)
        np.testing.assert_array_equal(retry.call_args.args[1], start)
        priced = price_slice_with_heston_and_greeks(quotes, 100.0, params)
        np.testing.assert_allclose(priced.Theo_Price, priced.Market_Price, atol=1e-4, rtol=0)
        self.assertTrue(np.isfinite(history).all())

    def test_seed42_day14_failed_fit_recovers(self):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "seed42_calibration_day14.json").read_text()
        )
        quotes = pd.DataFrame(fixture["quotes"])
        quotes.attrs["fft_config"] = tuple(fixture["fft_config"])
        failure = OptimizeResult(
            success=False,
            message="Maximum number of function evaluations has been exceeded.",
            x=np.array(fixture["failed_iterate"]),
            fun=fixture["failed_mse"],
        )
        # Replay the real failed iterate; run the recovery solver and pricer.
        with (
            patch("src.calc.brute", return_value=np.array(fixture["grid_seed"])),
            patch("src.calc.minimize", return_value=failure),
        ):
            params, _, _, _ = CM99_calibration_market(quotes, **fixture["calibration_kwargs"])
        mse = CM99_error_function_vectorized(params, quotes, **fixture["calibration_kwargs"])
        self.assertLess(mse, 1.1e-4)
        self.assertLess(mse, fixture["failed_mse"])
        self.assertTrue(np.isfinite(params).all())
        self.assertGreaterEqual(params[-1], 0.0)

    def test_successful_simplex_does_not_retry(self):
        result = OptimizeResult(success=True, x=PARAMS, fun=0.0)
        with (
            patch("src.calc.brute", return_value=PARAMS),
            patch("src.calc.minimize", return_value=result),
            patch("src.calc.least_squares") as retry,
        ):
            CM99_calibration_market(noiseless_quotes(), 100.0)
            retry.assert_not_called()

    def test_nonfinite_retry_is_not_accepted(self):
        failure = OptimizeResult(success=False, message="Evaluation limit reached", x=PARAMS)
        retry = OptimizeResult(success=True, x=PARAMS, fun=np.array([np.nan]))
        with (
            patch("src.calc.brute", return_value=PARAMS),
            patch("src.calc.minimize", return_value=failure),
            patch("src.calc.least_squares", return_value=retry),
        ):
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                CM99_calibration_market(noiseless_quotes(), 100.0)

    def test_invalid_simplex_iterate_restarts_from_grid_seed(self):
        failure = OptimizeResult(success=False, message="Evaluation limit reached", x=np.full(5, np.nan))
        recovered = OptimizeResult(success=True, x=PARAMS, fun=np.zeros(8))
        with (
            patch("src.calc.brute", return_value=PARAMS),
            patch("src.calc.minimize", return_value=failure),
            patch("src.calc.least_squares", return_value=recovered) as retry,
        ):
            CM99_calibration_market(noiseless_quotes(), 100.0)
        np.testing.assert_array_equal(retry.call_args.args[1], PARAMS)

    def test_retry_numerical_error_reports_both_solvers(self):
        failure = OptimizeResult(success=False, message="Evaluation limit reached", x=PARAMS)
        with (
            patch("src.calc.brute", return_value=PARAMS),
            patch("src.calc.minimize", return_value=failure),
            patch("src.calc.least_squares", side_effect=ValueError("Residuals not finite")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Evaluation limit reached.*Residuals not finite"):
                CM99_calibration_market(noiseless_quotes(), 100.0)

    def test_strategy_failure_reports_the_time_index(self):
        quotes = noiseless_quotes().assign(t_index=0)
        with patch("src.vol_arb.CM99_calibration_market", side_effect=RuntimeError("Both solvers failed")):
            with self.assertRaisesRegex(RuntimeError, "time index 0.*30.0 trading days.*Both solvers failed"):
                run_vol_arb_strategy(quotes)

    def test_nonfinite_optimizer_result_is_not_accepted(self):
        result = OptimizeResult(success=True, x=PARAMS, fun=np.nan)
        with patch("src.calc.brute", return_value=PARAMS), patch("src.calc.minimize", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                CM99_calibration_market(noiseless_quotes(), 100.0)


class SharedConfigurationTests(unittest.TestCase):
    def test_mismatched_calibration_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "calibration_N must equal"):
            run_vol_arb_strategy(pd.DataFrame(), calibration_N=1024, pricing_N=4096)

    def test_market_configuration_survives_iv_inversion_and_is_checked(self):
        market = generate_market_option_prices_across_time(
            [100.0, 101.0], [0.04, 0.04], 0.04, 6.4, 0.077, 0.24, -0.7,
            0.0, 0.0, 0.0, use_last_n=2, noise_scale=0.0,
        )
        market = append_market_iv(market)
        self.assertEqual(market.attrs["fft_config"], (DEFAULT_FFT_N, DEFAULT_FFT_ALPHA, DEFAULT_FFT_ETA))
        with self.assertRaisesRegex(ValueError, "Market generation"):
            run_vol_arb_strategy(market, pricing_N=4096)
        with self.assertRaisesRegex(ValueError, "Market generation"):
            run_vol_arb_strategy(market, eta=0.5)
        with self.assertRaisesRegex(ValueError, "Market generation"):
            CM99_calibration_market(market, S0=100.0, N=4096)
        with self.assertRaisesRegex(ValueError, "Market generation"):
            price_slice_with_heston_and_greeks(market, 100.0, PARAMS, alpha=2.0)


if __name__ == "__main__":
    unittest.main()

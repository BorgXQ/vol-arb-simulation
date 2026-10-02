import unittest
from unittest.mock import patch

import numpy as np

from src.calc import european_market_quotes, put_from_call_parity
from src.utils import (
    append_market_iv,
    bs_price,
    generate_market_option_prices_across_time,
    plot_stochastic_volatility_jump_path,
)


class EuropeanQuoteTests(unittest.TestCase):
    def test_call_floor_uses_discounted_strike(self):
        # The old S-K floor would accept $12; the correct floor is ~$14.39.
        with self.assertRaisesRegex(ValueError, "discounted-strike bounds"):
            european_market_quotes([12.0], 100.0, [90.0], 0.05, 1.0)

    def test_valid_put_below_immediate_exercise_value_is_preserved(self):
        call = bs_price(100.0, 110.0, 1.0, 0.05, 0.1, "C")
        expected_put = bs_price(100.0, 110.0, 1.0, 0.05, 0.1, "P")
        calls, puts = european_market_quotes([call], 100.0, [110.0], 0.05, 1.0)
        self.assertLess(expected_put, 10.0)
        self.assertAlmostEqual(puts[0], expected_put, places=12)
        self.assertEqual(calls[0], call)

    def test_call_upper_bound_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "discounted-strike bounds"):
            european_market_quotes([100.01], 100.0, [100.0], 0.02, 0.1)

    def test_tiny_bound_errors_are_corrected_before_puts_are_derived(self):
        discount = np.exp(-0.05)
        lower = 100.0 - 90.0 * discount
        calls, puts = european_market_quotes([lower - 1e-6], 100.0, [90.0], 0.05, 1.0)
        self.assertEqual(calls[0], lower)
        self.assertEqual(puts[0], 0.0)
        calls, puts = european_market_quotes([100.0 + 1e-6], 100.0, [110.0], 0.05, 1.0)
        self.assertEqual(calls[0], 100.0)
        self.assertAlmostEqual(puts[0], 110.0 * discount)

    def test_parity_helper_does_not_conceal_invalid_call_inputs(self):
        # This deliberately invalid call must expose a negative implied put;
        # quote validation, rather than independent put clipping, rejects it.
        self.assertEqual(put_from_call_parity([1.0], 100.0, [90.0], 0.0, 1.0)[0], -9.0)

    def test_cross_strike_violations_are_rejected(self):
        for strikes, calls, reason in [
            ([100, 105, 110], [3, 4, 1], "vertical-spread"),
            ([100, 101, 102], [4, 2, 1], "vertical-spread"),
            ([100, 105, 110], [6, 5, 3], "convexity"),
            ([100, 102, 110], [6, 5.9, 4], "convexity"),
        ]:
            with self.subTest(strikes=strikes, calls=calls):
                with self.assertRaisesRegex(ValueError, reason):
                    european_market_quotes(calls, 100.0, strikes, 0.0, 1.0)

    def test_valid_quotes_at_positive_zero_and_negative_rates(self):
        strikes = np.array([80.0, 94.0, 97.0, 100.0, 104.0, 121.0])
        for rate in [0.05, 0.0, -0.03]:
            with self.subTest(rate=rate):
                raw = np.array([bs_price(100.0, k, 0.2, rate, 0.2, "C") for k in strikes])
                calls, puts = european_market_quotes(raw, 100.0, strikes, rate, 0.2)
                discounted = strikes * np.exp(-rate * 0.2)
                np.testing.assert_allclose(calls - puts, 100.0 - discounted, atol=1e-12, rtol=0)
                self.assertTrue(np.all(calls >= np.maximum(100.0 - discounted, 0.0)))
                self.assertTrue(np.all(calls <= 100.0))
                self.assertTrue(np.all(puts >= np.maximum(discounted - 100.0, 0.0)))
                self.assertTrue(np.all(puts <= discounted + 1e-12))

    def test_nonfinite_quotes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite prices"):
            european_market_quotes([np.nan], 100.0, [100.0], 0.02, 0.1)


class MarketGenerationTests(unittest.TestCase):
    def generate(self, spots, variances, **overrides):
        parameters = dict(
            v0_m=0.042, kappa_m=6.4, theta_m=0.077, xi_m=0.24, rho_m=-0.7,
            jump_intensity_m=1.62, jump_mean_m=-0.165, jump_std_m=0.135,
            use_last_n=len(spots),
        )
        parameters.update(overrides)
        return generate_market_option_prices_across_time(spots, variances, **parameters)

    def test_expiry_uses_exact_payoffs_without_fft(self):
        with patch("src.utils.CM99_call_price_grid_jd_fft") as pricer:
            quotes = self.generate([100.2], [0.04])
            pricer.assert_not_called()
        self.assertTrue((quotes["T"] == 0).all())
        expected = np.where(
            quotes.Type == "C", np.maximum(100.2 - quotes.Strike, 0),
            np.maximum(quotes.Strike - 100.2, 0),
        )
        np.testing.assert_array_equal(quotes.Market_Price, expected)
        self.assertTrue(append_market_iv(quotes).Market_IV.isna().all())

    def test_independent_noise_is_explicitly_rejected(self):
        for noise in [0.005, -0.005, np.nan, np.inf]:
            with self.subTest(noise=noise), patch("src.utils.CM99_call_price_grid_jd_fft") as pricer:
                with self.assertRaisesRegex(ValueError, "noise_scale must be 0"):
                    self.generate([100.0, 101.0], [0.04, 0.04], noise_scale=noise)
                pricer.assert_not_called()

    def test_material_pricing_error_includes_time_index(self):
        with patch("src.utils.interpolate_call_prices", side_effect=lambda strikes, *_: np.full(len(strikes), 200.0)):
            with self.assertRaisesRegex(ValueError, "time index 0.*discounted-strike bounds"):
                self.generate([100.0, 101.0], [0.04, 0.04])

    def test_generated_market_satisfies_bounds_parity_and_spread_constraints(self):
        for seed in [1, 42, 67]:
            spots, variances, _ = plot_stochastic_volatility_jump_path(seed=seed)
            for window in [20, 30, 60]:
                with self.subTest(seed=seed, window=window):
                    market = self.generate(spots, variances, use_last_n=window)
                    for _, quotes in market.groupby("t_index"):
                        calls = quotes[quotes.Type == "C"].sort_values("Strike")
                        puts = quotes[quotes.Type == "P"].sort_values("Strike")
                        spot, rate, maturity = calls[["S_t", "r", "T"]].iloc[0]
                        k = calls.Strike.to_numpy()
                        c = calls.Market_Price.to_numpy()
                        p = puts.Market_Price.to_numpy()
                        disc = np.exp(-rate * maturity)
                        np.testing.assert_allclose(c - p, spot - k * disc, atol=1e-12, rtol=0)
                        self.assertTrue(np.all(c >= np.maximum(spot - k * disc, 0)))
                        self.assertTrue(np.all(c <= spot))
                        self.assertTrue(np.all(p >= np.maximum(k * disc - spot, 0)))
                        self.assertTrue(np.all(p <= k * disc + 1e-12))
                        # Equal-spacing butterfly and vertical-spread checks.
                        tolerance = 5e-6 * spot / 100
                        self.assertTrue(np.all(np.diff(c) <= 2 * tolerance))
                        self.assertTrue(np.all(np.diff(c) >= -disc * np.diff(k) - 2 * tolerance))
                        self.assertTrue(np.all(np.diff(c, n=2) >= -4 * tolerance))


if __name__ == "__main__":
    unittest.main()

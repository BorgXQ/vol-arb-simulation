import unittest

import numpy as np
import pandas as pd

from src.utils import generate_market_option_prices_across_time


class StrikeUniverseTests(unittest.TestCase):
    def generate(self, spots, variances=None, use_last_n=None):
        if variances is None:
            variances = np.full(len(spots), 0.04)
        # Use identical random draws to isolate dependence on future states.
        random_state = np.random.get_state()
        try:
            np.random.seed(123)
            return generate_market_option_prices_across_time(
                S_path=np.asarray(spots),
                v_path=np.asarray(variances),
                v0_m=0.04,
                kappa_m=6.4,
                theta_m=0.077,
                xi_m=0.24,
                rho_m=-0.7,
                jump_intensity_m=0.7,
                jump_mean_m=-0.02,
                jump_std_m=0.04,
                N=4096,
                use_last_n=len(spots) if use_last_n is None else use_last_n,
                noise_scale=0.005,
            )
        finally:
            np.random.set_state(random_state)

    def test_future_states_do_not_change_observed_quotes_or_contracts(self):
        original = self.generate([100.0, 101.0, 102.0, 103.0])
        changed = self.generate(
            [100.0, 101.0, 60.0, 150.0], [0.04, 0.04, 0.09, 0.01]
        )
        pd.testing.assert_frame_equal(
            original[original.t_index <= 1], changed[changed.t_index <= 1]
        )
        pd.testing.assert_frame_equal(
            original[["t_index", "Type", "Strike"]],
            changed[["t_index", "Type", "Strike"]],
        )

    def test_contracts_remain_available_after_spot_leaves_initial_range(self):
        market = self.generate([100.0, 50.0, 160.0])
        initial = market.loc[market.t_index == 0, ["Type", "Strike"]].reset_index(drop=True)
        for _, quotes in market.groupby("t_index"):
            pd.testing.assert_frame_equal(
                initial, quotes[["Type", "Strike"]].reset_index(drop=True)
            )
        strikes = initial.Strike.unique()
        self.assertGreater(160.0, strikes.max())
        self.assertLess(50.0, strikes.min())

    def test_grid_uses_selected_window_inception(self):
        full = self.generate([200.0, 100.0, 101.0, 102.0], use_last_n=3)
        window = self.generate([100.0, 101.0, 102.0])
        pd.testing.assert_frame_equal(full, window)

    def test_grid_has_positive_even_strikes_and_initial_otm_coverage(self):
        for spot in [5.0, 20.0, 85.43, 100.0]:
            with self.subTest(spot=spot):
                market = self.generate([spot, spot])
                strikes = market.Strike.unique()
                self.assertTrue(np.all(strikes > 0))
                self.assertTrue(np.all(strikes % 2 == 0))
                np.testing.assert_array_equal(np.diff(strikes), 2.0)
                if spot >= 20.0:
                    self.assertGreaterEqual(np.count_nonzero(strikes < spot), 3)
                    self.assertGreaterEqual(np.count_nonzero(strikes > spot), 3)


if __name__ == "__main__":
    unittest.main()

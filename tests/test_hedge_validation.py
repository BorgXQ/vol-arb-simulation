"""Independent price-integration checks and adversarial hedge systems."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.integrate import quad_vec

from src.calc import H93_char_func_cm
from src.vol_arb import price_slice_with_heston_and_greeks, solve_option_hedge, run_vol_arb_strategy


def integrated_calls(spot, variance, days, strikes):
    alpha = 1.5
    def integrand(u):
        phi = H93_char_func_cm(u - (alpha + 1) * 1j, S0=spot, v0=variance,
                              kappa_v=6.4, theta_v=.077, xi_v=.24, rho=-.7, r=.02, T=days/252)
        return np.real(np.exp(-1j * u * np.log(strikes)) * phi /
                       (alpha**2 + alpha - u*u + 1j*(2*alpha+1)*u))
    value, _ = quad_vec(integrand, 0, np.inf, epsabs=1e-8, epsrel=1e-10)
    return np.exp(-.02 * days / 252 - alpha*np.log(strikes)) * value / np.pi


def quotes(days=30):
    return pd.DataFrame(dict(Strike=[96., 100., 104., 96., 100., 104.],
                             Type=['C']*3 + ['P']*3, T=days/252, r=.02, Market_IV=.2))


class GreekTests(unittest.TestCase):
    def test_greeks_against_independent_integration_including_variance_boundary(self):
        strikes = np.array([96., 100., 104.])
        for days in (11, 30, 60):
            for v in (0., 1e-6, .04):
                with self.subTest(days=days, variance=v):
                    priced = price_slice_with_heston_and_greeks(quotes(days), 100., [6.4, .077, .24, -.7, v])
                    p = integrated_calls(100., v, days, strikes)
                    up = integrated_calls(100.05, v, days, strikes)
                    dn = integrated_calls(99.95, v, days, strikes)
                    delta, gamma = (up-dn)/.1, (up-2*p+dn)/.05**2
                    hv = 1e-5
                    if v >= hv:
                        var = (integrated_calls(100., v+hv, days, strikes)-integrated_calls(100., v-hv, days, strikes))/(2*hv)
                    else:
                        var = (-3*p+4*integrated_calls(100., v+hv, days, strikes)-integrated_calls(100., v+2*hv, days, strikes))/(2*hv)
                    np.testing.assert_allclose(priced.Delta[:3], delta, atol=2e-5, rtol=.001)
                    np.testing.assert_allclose(priced.Gamma[:3], gamma, atol=2e-5, rtol=.001)
                    np.testing.assert_allclose(priced.VarianceSensitivity[:3], var, atol=.002, rtol=.001)
                    np.testing.assert_allclose(priced.Delta[:3].to_numpy()-priced.Delta[3:].to_numpy(), 1, atol=1e-10)
                    np.testing.assert_allclose(priced.Gamma[:3], priced.Gamma[3:], atol=1e-10)
                    np.testing.assert_allclose(priced.VarianceSensitivity[:3], priced.VarianceSensitivity[3:], atol=1e-7)

    def test_underresolved_greeks_fail(self):
        with self.assertRaisesRegex(RuntimeError, 'did not converge'):
            price_slice_with_heston_and_greeks(quotes(11), 100., [6.4, .077, .24, -.7, .04], N=256)

    def test_real_options_have_small_independently_repriced_residuals(self):
        params = [6.4, .077, .24, -.7, .04]
        priced = price_slice_with_heston_and_greeks(quotes(), 100., params)
        finer = price_slice_with_heston_and_greeks(quotes(), 100., params, N=32768, eps_S_rel=.005, eps_v_rel=.025)
        for mode in ('gamma_delta', 'gamma_delta_variance'):
            weights, stock, diagnostics = solve_option_hedge(priced, priced.Option_ID.iloc[0], 1, mode)
            w = finer.Option_ID.map(weights).to_numpy()
            self.assertLess(abs(w @ finer.Delta + stock), 2e-5)
            self.assertLess(abs(w @ finer.Gamma), 2e-5)
            if mode == 'gamma_delta_variance':
                self.assertLess(abs(w @ finer.VarianceSensitivity), .002)
            self.assertLess(diagnostics['hedge_residual'], 1e-6)


class HedgeTests(unittest.TestCase):
    def universe(self):
        return pd.DataFrame(dict(Option_ID=['target', 'a', 'b'], Delta=[.5,.3,-.2],
                                 Gamma=[.02,.01,.03], VarianceSensitivity=[4.,3.,2.]))

    def test_modes_and_residuals(self):
        df = self.universe()
        for position in (-2., 1.):
            for mode in ('gamma_delta', 'gamma_delta_variance'):
                weights, stock, _ = solve_option_hedge(df, 'target', position, mode)
                w = df.Option_ID.map(weights).to_numpy()
                self.assertAlmostEqual(w @ df.Delta + stock, 0)
                self.assertAlmostEqual(w @ df.Gamma, 0)
                if mode == 'gamma_delta_variance':
                    self.assertAlmostEqual(w @ df.VarianceSensitivity, 0)
                else:
                    self.assertGreater(abs(w @ df.VarianceSensitivity), .1)

    def test_units_do_not_change_weights(self):
        df = self.universe()
        original = solve_option_hedge(df, 'target', 1)[0]
        df['Gamma'] *= 1e6
        df['VarianceSensitivity'] *= 1e-6
        scaled = solve_option_hedge(df, 'target', 1)[0]
        np.testing.assert_allclose(list(original.values()), list(scaled.values()), atol=1e-12)

    def test_small_greek_perturbations_leave_weights_stable(self):
        df = self.universe()
        original = solve_option_hedge(df, 'target', 1)[0]
        df.loc[1, 'Gamma'] *= 1.00001
        df.loc[2, 'VarianceSensitivity'] *= .99999
        perturbed = solve_option_hedge(df, 'target', 1)[0]
        np.testing.assert_allclose(list(original.values()), list(perturbed.values()), rtol=1e-4)

    def test_compatible_rank_deficiency_is_allowed(self):
        df = self.universe()
        df['VarianceSensitivity'] = df.Gamma * 100
        _, _, diag = solve_option_hedge(df, 'target', 1)
        self.assertEqual(diag['hedge_rank'], 1)

    def test_nearly_singular_incompatible_system_fails(self):
        df = self.universe()
        df['Gamma'] = [2., 1., 1.]
        df['VarianceSensitivity'] = [3., 1., 1.00001]
        with self.assertRaisesRegex(RuntimeError, 'stable hedge'):
            solve_option_hedge(df, 'target', 1)

    def test_excessive_positions_fail(self):
        df = self.universe()
        df.loc[0, ['Gamma', 'VarianceSensitivity']] *= 100
        with self.assertRaisesRegex(RuntimeError, 'position limits'):
            solve_option_hedge(df, 'target', 1)

    def test_missing_hedgers_do_not_silently_become_delta_only(self):
        with self.assertRaisesRegex(RuntimeError, 'stable hedge'):
            solve_option_hedge(self.universe().iloc[:1], 'target', 1)

    def test_invalid_input_rejected(self):
        df = self.universe()
        with self.assertRaises(ValueError):
            solve_option_hedge(df, 'target', 1, 'unknown')
        with self.assertRaises(ValueError):
            solve_option_hedge(pd.concat([df,df.iloc[:1]]), 'target', 1)
        df.loc[1, 'Gamma'] = np.nan
        with self.assertRaises(ValueError):
            solve_option_hedge(df, 'target', 1)

    def test_strategy_records_diagnostics_and_clears_variance_on_exit(self):
        first = quotes(11).assign(t_index=0, S_t=100., Market_Price=2.)
        last = quotes(10).assign(t_index=1, S_t=100., Market_Price=2.)
        with patch('src.vol_arb.CM99_calibration_market', return_value=([6.4,.077,.24,-.7,.04],0,[],[])):
            state, _ = run_vol_arb_strategy(pd.concat([first,last]), hedge_mode='gamma_delta')
        self.assertTrue(np.isfinite(state.loc[0, 'greek_error_ratio']))
        self.assertGreater(abs(state.loc[0, 'net_variance_sensitivity']), 1e-4)
        self.assertEqual(state.loc[1, 'net_variance_sensitivity'], 0)
        self.assertEqual(state.loc[1, 'net_gamma'], 0)


if __name__ == '__main__':
    unittest.main()

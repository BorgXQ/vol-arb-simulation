import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.metrics import compute_performance_metrics
from src.utils import strip_state_df, plot_strategy_dashboard_plotly


def state(pnl):
    df = pd.DataFrame(dict(pnl_incremental=pnl, pnl_cumulative=np.cumsum(pnl),
                           S_t=100., T=(len(pnl)-np.arange(len(pnl)))/252,
                           target_option_id=None, target_position=0., w_underlying=0.))
    return df


class PerformanceTests(unittest.TestCase):
    def test_hand_calculated_metrics_exclude_inception_and_include_last_period(self):
        result = compute_performance_metrics(state([0.,3.,-1.,-1.]), 100.)
        self.assertEqual(result['period_count'], 3)
        self.assertAlmostEqual(result['mean_normalized_pnl'], .01/3)
        downside = np.sqrt((.01**2+.01**2)/3)
        self.assertAlmostEqual(result['downside_deviation'], downside)
        self.assertAlmostEqual(result['sortino'], np.sqrt(252)*(.01/3)/downside)
        self.assertAlmostEqual(result['sharpe'], np.sqrt(252)*(.01/3)/np.std([.03,-.01,-.01],ddof=1))
        np.testing.assert_allclose(result['cumulative_normalized_pnl'], [0,.03,.02,.01], atol=1e-15)
        np.testing.assert_allclose(result['drawdown'], [0,0,-.01,-.02], atol=1e-15)
        self.assertAlmostEqual(result['max_drawdown'], .02)
        self.assertAlmostEqual(result['annualized_normalized_pnl'], .84)
        self.assertAlmostEqual(result['annualized_pnl_to_drawdown'], 42.)

    def test_identical_losses_have_nonzero_downside_and_finite_sortino(self):
        result = compute_performance_metrics(state([0.,-1.,-1.]), 100.)
        self.assertAlmostEqual(result['downside_deviation'], .01)
        self.assertAlmostEqual(result['sortino'], -np.sqrt(252))
        self.assertTrue(np.isnan(result['sharpe']))
        self.assertAlmostEqual(result['max_drawdown'], .02)

    def test_first_loss_counts_against_initial_zero_baseline(self):
        result = compute_performance_metrics(state([0.,-2.,1.]),100.)
        np.testing.assert_allclose(result['drawdown'],[0,-.02,-.01])
        self.assertAlmostEqual(result['max_drawdown'],.02)

    def test_no_losses_zero_variance_and_single_period_are_defined(self):
        for pnl in ([0.], [0.,0.,0.], [0.,1.,1.], [0.]+[10.]*7):
            result = compute_performance_metrics(state(pnl),100.)
            self.assertTrue(np.isnan(result['sortino']))
            self.assertTrue(np.isnan(result['sharpe']))
            self.assertTrue(np.isnan(result['annualized_pnl_to_drawdown']))
            self.assertEqual(result['max_drawdown'],0.)
        result = compute_performance_metrics(state([0.,-1.]),100.)
        self.assertTrue(np.isnan(result['sharpe']))
        self.assertAlmostEqual(result['sortino'],-np.sqrt(252))

    def test_empty_window_has_no_invented_statistics(self):
        result = compute_performance_metrics(state([]),100.)
        self.assertEqual(result['period_count'],0)
        self.assertTrue(result['cumulative_normalized_pnl'].empty)
        for name in ('sharpe','sortino','max_drawdown','annualized_pnl_to_drawdown'):
            self.assertTrue(np.isnan(result[name]))

    def test_invalid_data_is_not_silently_converted_to_zero(self):
        for value in (np.nan,np.inf,'bad'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compute_performance_metrics(pd.DataFrame({'pnl_incremental':[0,value]}),100)
        for capital in (None,0,-1,np.nan,np.inf):
            with self.subTest(capital=capital), self.assertRaises(ValueError):
                compute_performance_metrics(state([0,1]),capital)
        with self.assertRaisesRegex(ValueError,'inception'):
            compute_performance_metrics(state([1,2]),100)

    def test_irregular_intervals_do_not_get_daily_annualization(self):
        df = state([0.,2.,-1.])
        df['T'] = np.array([4,3,1])/252
        result = compute_performance_metrics(df,100)
        self.assertFalse(result['regular_intervals'])
        for name in ('sharpe','sortino','annualized_normalized_pnl','annualized_pnl_to_drawdown'):
            self.assertTrue(np.isnan(result[name]))
        self.assertAlmostEqual(result['max_drawdown'],.01)
        self.assertAlmostEqual(result['cumulative_normalized_pnl'].iloc[-1],.01)

    def test_reduced_state_preserves_tiny_pnl_and_weights(self):
        df = state([0.,1.23456789e-8,-2.34567891e-8])
        df['w_underlying'] = .123456789123
        reduced = strip_state_df(df)
        np.testing.assert_array_equal(reduced.pnl_incremental,df.pnl_incremental)
        np.testing.assert_array_equal(reduced.w_underlying,df.w_underlying)
        self.assertAlmostEqual(compute_performance_metrics(reduced,1)['max_drawdown'],2.34567891e-8,places=18)

    def test_app_and_notebook_share_values_and_labels(self):
        from app import make_strategy_dashboard_figure
        reduced = strip_state_df(state([0.,3.,-1.,-1.]))
        app_fig, metrics = make_strategy_dashboard_figure(reduced,100.)
        with patch.object(go.Figure,'show',autospec=True) as show:
            plot_strategy_dashboard_plotly(reduced,100.)
        notebook_fig = show.call_args.args[0]
        for fig in (app_fig,notebook_fig):
            trace = fig.data[-1]
            np.testing.assert_allclose(trace.y,metrics['cumulative_normalized_pnl'])
            self.assertIn('P&L',trace.name)
        text = ' '.join(a.text for a in notebook_fig.layout.annotations)
        self.assertIn('Annualized P&L / max drawdown: 42.000',text)
        self.assertNotIn('Calmar',text)


if __name__ == '__main__':
    unittest.main()

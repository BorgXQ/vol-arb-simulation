"""Hand-calculated self-financing examples, including liquidation and reporting."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.utils import strategy_reporting_window, strip_state_df
from src.vol_arb import (
    ACCOUNT_COLUMNS, add_option_id_column,
    run_vol_arb_strategy, settle_cash_account,
)


class CashAccountTests(unittest.TestCase):
    def example(self, rates=(.05, .09, .7, .7), sign=1):
        # Unequal intervals deliberately exercise T-based financing, not row count.
        slices = [pd.DataFrame(dict(Option_ID=['A', 'B'], Market_Price=prices, r=rate))
                  for prices, rate in zip(([5., 3.], [6., 2.], [4., 3.], [2., 1.]), rates)]
        state = pd.DataFrame(dict(S_t=[100.,102.,101.,99.], T=np.array([13.,12.,10.,9.])/252,
                                  w_A=np.array([2.,1.,0.,0.])*sign,
                                  w_B=np.array([-1.,-2.,0.,0.])*sign,
                                  w_underlying=np.array([.5,-.25,0.,0.])*sign,
                                  account_status=['active','active','exited','closed']))
        for col in (*ACCOUNT_COLUMNS, 'pnl_incremental', 'pnl_cumulative'):
            state[col] = 0.
        for i in range(4):
            settle_cash_account(state, slices[i], i, slices[i-1] if i else None, i-1 if i else None)
        return state, slices

    def test_hand_calculated_entry_rebalance_exit_and_frozen_cash(self):
        state, _ = self.example()
        # Entry: buy 2*A (10), sell B (3), buy .5 stock (50): borrow 57.
        # Day 1: old assets = 12 - 2 + 51 = 61; new = 6 - 4 - 25.5 = -23.5.
        # Trades therefore receive 84.5. Price P&L is 4, independent of trades.
        i1 = -57 * np.expm1(.05/252)
        cash1 = 27.5 + i1
        # Exit: old assets = 4 - 6 - 25.25 = -27.25; pay 27.25 to close.
        i2 = cash1 * np.expm1(.09*2/252)
        final_cash = .25 + i1 + i2
        np.testing.assert_allclose(state.trade_cashflow, [-57,84.5,-27.25,0], atol=1e-12)
        np.testing.assert_allclose(state.cash_balance, [-57,cash1,final_cash,final_cash], atol=1e-12)
        np.testing.assert_allclose(state.holdings_value, [57,-23.5,0,0], atol=1e-12)
        np.testing.assert_allclose(state.trading_pnl_incremental, [0,4,-3.75,0], atol=1e-12)
        np.testing.assert_allclose(state.financing_incremental, [0,i1,i2,0], atol=1e-12)
        np.testing.assert_allclose(state.pnl_incremental, [0,4+i1,-3.75+i2,0], atol=1e-12)
        np.testing.assert_allclose(state.equity, state.cash_balance + state.holdings_value, atol=1e-12)
        np.testing.assert_allclose(state.pnl_cumulative, state.pnl_incremental.cumsum(), atol=1e-12)
        self.assertAlmostEqual(state.financing_cumulative.iloc[-1], i1+i2)

    def test_zero_rate_recovers_holdings_only_pnl(self):
        state, _ = self.example(rates=(0.,)*4)
        np.testing.assert_allclose(state.pnl_incremental, [0,4,-3.75,0], atol=1e-12)
        np.testing.assert_allclose(state.financing_incremental, 0)

    def test_short_positions_reverse_cash_and_pnl(self):
        long, _ = self.example()
        short, _ = self.example(sign=-1)
        for col in (*ACCOUNT_COLUMNS, 'pnl_incremental', 'pnl_cumulative'):
            np.testing.assert_allclose(short[col], -long[col], atol=1e-12)

    def test_missing_held_quote_fails_instead_of_dropping_pnl(self):
        state, slices = self.example()
        with self.assertRaisesRegex(ValueError, 'Missing market quotes.*A'):
            settle_cash_account(state, slices[2].iloc[1:], 2, slices[1], 1)

    def test_invalid_marks_rates_and_time_fail(self):
        state, slices = self.example()
        bad_marks = slices[1].copy()
        bad_marks.loc[0, 'Market_Price'] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite marks'):
            settle_cash_account(state, bad_marks, 1, slices[0], 0)
        bad_rates = slices[0].copy()
        bad_rates.loc[0, 'r'] = .1
        with self.assertRaisesRegex(ValueError, 'single finite funding rate'):
            settle_cash_account(state, slices[1], 1, bad_rates, 0)
        state.loc[1,'T'] = state.loc[0,'T']
        with self.assertRaisesRegex(ValueError, 'decreasing maturities'):
            settle_cash_account(state, slices[1], 1, slices[0], 0)

    def test_reporting_includes_liquidation_once(self):
        state, _ = self.example()
        result = strategy_reporting_window(state)
        self.assertEqual(result.index.tolist(), [0,1,2])
        self.assertEqual(result.account_status.iloc[-1], 'exited')
        self.assertAlmostEqual(result.pnl_cumulative.iloc[-1], state.pnl_cumulative.iloc[-1])
        self.assertEqual(len(strategy_reporting_window(state.iloc[:2])), 2)
        self.assertEqual(len(strategy_reporting_window(state.iloc[3:])), 1)


class StrategyAccountingTests(unittest.TestCase):
    def market(self):
        return pd.concat([
            pd.DataFrame(dict(t_index=i, S_t=spot, T=days/252, r=.02,
                              Type=['C','P'], Strike=[104.,96.], Market_IV=.2,
                              Market_Price=prices))
            for i, (days, spot, prices) in enumerate(zip([13,12,10,9], [100,102,101,99],
                                                        [[5,3],[6,2],[4,3],[2,1]]))
        ], ignore_index=True)

    def test_strategy_settles_trades_and_reports_exit_quotes(self):
        market = self.market()
        ids = add_option_id_column(market).Option_ID.iloc[:2].tolist()
        def priced(universe, **kwargs):
            return add_option_id_column(universe).assign(Theo_Price=5., Theo_IV=.25, IV_Diff=.05,
                                                         Delta=.5, Gamma=.02, Vega=3., Greek_Error_Ratio=0.)
        # Deterministic weights isolate accounting from expensive calibration and hedging.
        with (patch('src.vol_arb.CM99_calibration_market', return_value=([6,.07,.2,-.7,.04],0,[],[])),
              patch('src.vol_arb.price_slice_with_heston_and_greeks', side_effect=priced),
              patch('src.vol_arb.select_target_contract', return_value=(ids[0],1.,None)),
              patch('src.vol_arb.solve_option_hedge', side_effect=[
                  ({ids[0]:1.,ids[1]:-1.}, .5, {}),
                  ({ids[0]:1.,ids[1]:-2.}, -.25, {}),
              ])):
            state, gross = run_vol_arb_strategy(market)
        self.assertEqual(gross, 58.)  # 5 + 3 + 50, not a cash deposit
        i1 = -52*np.expm1(.02/252)
        cash1 = 26.5 + i1  # old holdings 6 - 2 + 51 = 55; new holdings -23.5
        i2 = cash1*np.expm1(.02*2/252)
        expected = -.75 + i1 + i2
        self.assertAlmostEqual(state.loc[2,'cash_balance'], expected)
        self.assertAlmostEqual(state.loc[2,'pnl_incremental'], -3.75+i2)
        self.assertEqual(state.account_status.tolist(), ['active','active','exited','closed'])
        self.assertEqual(state.loc[2,'target_position'], 0)
        self.assertTrue((state.loc[2, [c for c in state if c.startswith('w_')]] == 0).all())
        self.assertEqual(state.loc[2, f'mkt_price_{ids[0]}'], 4.)
        self.assertEqual(state.loc[3,'pnl_incremental'], 0.)
        reduced = strip_state_df(strategy_reporting_window(state))
        self.assertEqual(len(reduced), 3)
        self.assertEqual(reduced.mkt_price_target.iloc[-1], 4.)
        self.assertAlmostEqual(reduced.pnl_cumulative.iloc[-1], expected, places=6)
        self.assertEqual(reduced.cash_balance.iloc[-1], reduced.equity.iloc[-1])

    def test_window_already_inside_exit_threshold_opens_no_positions(self):
        with patch('src.vol_arb.CM99_calibration_market') as calibrate:
            state, gross = run_vol_arb_strategy(self.market().query('T <= 10/252'))
        calibrate.assert_not_called()
        self.assertIsNone(gross)
        self.assertTrue((state.account_status == 'closed').all())
        self.assertTrue((state.pnl_cumulative == 0).all())
        self.assertTrue((state.cash_balance == 0).all())


if __name__ == '__main__':
    unittest.main()

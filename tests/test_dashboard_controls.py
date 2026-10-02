import json
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from app import DEFAULTS, TTE_MIN_DAYS, TTE_MAX_DAYS, run_analysis_cached, settings_from_controls


class DashboardControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fit = ([6.4,.077,.24,-.7,.04],0,[],[])
        run_analysis_cached.clear()
        with patch('src.vol_arb.CM99_calibration_market', return_value=fit) as strategy_fit, patch('app.CM99_calibration_market', return_value=fit):
            cls.result = run_analysis_cached(**dict(DEFAULTS,seed=42))
            cls.inception_fit_count = strategy_fit.call_count
            cls.changed_market = run_analysis_cached(**dict(DEFAULTS,seed=42,jump_on=False,kappa=7.))

    def test_tte_is_days_plus_one_observation(self):
        self.assertEqual((TTE_MIN_DAYS,TTE_MAX_DAYS),(11,30))
        for days in (11,30):
            args = settings_from_controls(dict(DEFAULTS,tte_days=days))
            self.assertEqual(args['use_last_n'],days+1)
        self.assertEqual(self.result['run_settings']['use_last_n'],12)
        self.assertAlmostEqual(self.result['full_slice_t0']['T'].iloc[0],11/252)
        self.assertEqual(self.inception_fit_count,1)
        self.assertEqual(self.result['cash_account'].account_status.tolist(),['active','exited'])

    def test_market_controls_do_not_change_underlying_path(self):
        np.testing.assert_array_equal(self.result['S_path'],self.changed_market['S_path'])
        np.testing.assert_array_equal(self.result['v_path'],self.changed_market['v_path'])
        self.assertFalse(np.allclose(self.result['options_market_df'].Market_Price,
                                     self.changed_market['options_market_df'].Market_Price))
        self.assertTrue(self.result['run_settings']['jump_on'])
        self.assertFalse(self.changed_market['run_settings']['jump_on'])

    def test_editing_controls_preserves_displayed_run_and_reset_clears_it(self):
        app = AppTest.from_file('app.py')
        app.session_state['analysis_result'] = self.result
        app.session_state['seed'] = 42
        app.run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(app.slider(key='tte_days').value,11)
        self.assertFalse(any('Controls have changed' in item.value for item in app.info))
        before = [chart.proto.spec for chart in app.get('plotly_chart')]
        self.assertGreater(len(before),0)
        labels = [item.label for item in app.metric]
        app.slider(key='tte_days').set_value(30).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Controls have changed' in item.value for item in app.info))
        self.assertEqual(before,[chart.proto.spec for chart in app.get('plotly_chart')])
        self.assertEqual(labels,[item.label for item in app.metric])
        self.assertTrue(any('initial expiry 11 trading days' in item.value for item in app.caption))
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.slider(key='tte_days').value,11)
        self.assertEqual(len(app.get('plotly_chart')),0)

    def test_notebook_uses_actual_tte_explicitly(self):
        from pathlib import Path
        notebook = json.loads(Path('test.ipynb').read_text())
        first = ''.join(notebook['cells'][0]['source'])
        self.assertIn('initial_tte_days = 11',first)
        self.assertIn('use_last_n = initial_tte_days + 1',first)


if __name__ == '__main__':
    unittest.main()

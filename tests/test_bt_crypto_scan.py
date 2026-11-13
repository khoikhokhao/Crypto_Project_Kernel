import ast
import contextlib
import io
import re
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import nbformat

from bt_crypto import BacktestConfig, backtest
from bt_crypto_scan import scan_risk_grid
from research.alpha_helpers import build_targets, run_grid


class FastScanParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = staticmethod(build_targets)
        cls.cfg = BacktestConfig(asset='BTCUSDT', timeframe='1h', timestamp_label='open',
                                 contract_type='linear_usdt', fx_vnd_per_usdt=25000,
                                 initial_capital_vnd=10e9, fixed_margin_budget_vnd=10e9,
                                 leverage=5, fee_rate_per_side=.0005, slippage_bps=10)

    def compare(self, bars, signal, risks):
        fast = scan_risk_grid(ohlcv=bars, entry_signals=signal, risk_grid=risks,
                              config=self.cfg, workers=2, batch_size=2)
        f = pd.DataFrame({'sig_long': signal.eq(1), 'sig_short': signal.eq(-1)}, index=bars.index)
        mapping = {'net_profit_vnd':'net_profit_vnd', 'annualized_simple_return':'annualized_simple_return',
                   'mdd_vnd':'max_drawdown_vnd', 'mdd_pct_initial':'max_drawdown_pct_initial',
                   'sharpe':'sharpe', 'closed_trades':'closed_trade_count', 'open_trades':'open_trade_count',
                   'win_rate':'win_rate', 'avg_net_per_trade_vnd':'average_net_pnl_per_closed_trade_vnd',
                   'total_fees_vnd':'total_fees_vnd', 'total_slippage_vnd':'total_slippage_vnd',
                   'avg_daily_margin_turnover':'average_daily_margin_turnover_on_initial'}
        for i, risk in enumerate(risks.to_dict('records')):
            target, pos, entry, _ = self.build(bars, f, risk, self.cfg.slippage_bps)
            reference = backtest(ohlcv=bars, target_position=target, config=self.cfg)
            np.testing.assert_array_equal(pos, reference.ledger.position)
            np.testing.assert_allclose(entry, reference.ledger.entry_fill_price, equal_nan=True)
            for a, b in mapping.items():
                v, expected = fast[a].iat[i], reference.metrics[b]
                if expected is None:
                    self.assertTrue(pd.isna(v), f'{a}: {v} expected None')
                else:
                    self.assertAlmostEqual(float(v), float(expected), delta=max(0.01, abs(float(expected))*1e-9), msg=a)
            self.assertEqual(fast.missing_bars.iat[i], reference.diagnostics['missing_bar_count'])

    def test_multiple_risk_settings_gap_partial_days_terminal(self):
        rng = np.random.default_rng(27)
        full = pd.date_range('2025-01-01', periods=240, freq='1h', tz='UTC')
        idx = full.delete(list(range(48, 72)) + [90, 91])
        close = 100 * np.exp(np.cumsum(rng.normal(0, .015, len(idx))))
        opens = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, .005, len(idx)))
        bars = pd.DataFrame({'open': opens, 'high': np.maximum(opens,close)*1.02,
                             'low': np.minimum(opens,close)*.98, 'close':close, 'volume':1.}, index=idx)
        signal = pd.Series(rng.choice([-1,0,1],len(idx),p=[.1,.8,.1]),index=idx)
        risks = pd.DataFrame({'tp':[.01,.04,.1], 'sl':[.01,.02,.05], 'max_hold':[1,5,1000]})
        self.compare(bars,signal,risks)

    def test_flat_undefined_and_last_signal(self):
        idx = pd.date_range('2025-01-01',periods=48,freq='1h',tz='UTC')
        bars = pd.DataFrame({'open':100.,'high':100.,'low':100.,'close':100.,'volume':1.},index=idx)
        signal = pd.Series(0,index=idx,dtype='int8')
        signal.iloc[-1] = 1
        self.compare(bars,signal,pd.DataFrame({'tp':[.05],'sl':[.05],'max_hold':[20]}))

    def test_funding_rejected_and_invalid_risk(self):
        from dataclasses import replace
        idx = pd.date_range('2025-01-01',periods=3,freq='1h',tz='UTC')
        bars = pd.DataFrame({'open':100.,'high':100.,'low':100.,'close':100.,'volume':1.},index=idx)
        signal = pd.Series(0,index=idx)
        risk = pd.DataFrame({'tp':[.1],'sl':[.1],'max_hold':[5]})
        with self.assertRaisesRegex(ValueError, 'funding'):
            scan_risk_grid(ohlcv=bars,entry_signals=signal,risk_grid=risk,
                           config=replace(self.cfg,funding_convention='positive_rate_longs_pay'))
        with self.assertRaisesRegex(ValueError, 'integer'):
            scan_risk_grid(ohlcv=bars,entry_signals=signal,risk_grid=risk.assign(max_hold=1.5),config=self.cfg)

    def test_notebook_grid_integration(self):
        notebook = nbformat.read(Path(__file__).resolve().parents[1] / 'research/btc_alpha38_vwpd_5m.ipynb', as_version=4)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        namespace = {'np':np, 'pd':pd, 'backtest':backtest, 'display':lambda x: None,
                     'run_grid':run_grid,
                     'ROOT':Path(temp.name),
                     'TIMEFRAME':'1h', 'SLIPPAGE_BPS':self.cfg.slippage_bps, 'config':self.cfg,
                     'PARAMS':{'atr_short':2,'atr_long':4,'vwpd_threshold':.1,'tp':.02,'sl':.01,'max_hold':5}}
        for cell in notebook.cells:
            for node in ast.parse(cell.source).body:
                if isinstance(node, ast.FunctionDef) and node.name in ('calculate_vwpd_features', 'entry_conditions'):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook function>', 'exec'), namespace)
        idx = pd.date_range('2025-01-01', periods=120, freq='1h', tz='UTC')
        rng = np.random.default_rng(3)
        close = 100*np.exp(np.cumsum(rng.normal(0,.02,len(idx))))
        opens = np.r_[close[0],close[:-1]]
        namespace['df'] = pd.DataFrame({'open':opens,'high':np.maximum(opens,close)*1.01,
            'low':np.minimum(opens,close)*.99,'close':close,'volume':1.},index=idx)
        grid = next(c.source for c in notebook.cells if 'GRID SEARCH' in c.source)
        grid = re.sub(r'GRID_START_UTC = "[^"]+"', 'GRID_START_UTC = "2025-01-01"', grid)
        grid = re.sub(r'GRID_END_UTC = "[^"]+"', 'GRID_END_UTC = "2025-01-06"', grid)
        grid = grid.replace('WORKERS = 8', 'WORKERS = 2')
        start, end = grid.index('GRID = {'), grid.index('\n}',grid.index('GRID = {'))+2
        grid = grid[:start] + '''GRID = {"atr_short":[2,4], "atr_long":[4], "vwpd_threshold":[0.01,0.2],
                   "tp":[0.02,0.04], "sl":[0.01], "max_hold":[2,5]}''' + grid[end:]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(grid, '<notebook grid>', 'exec'), namespace)
        results = namespace['grid_results']
        self.assertEqual(len(results),8)
        saved = list((Path(temp.name) / 'research/grid_results').glob('*.csv'))
        self.assertEqual(len(saved), 1)
        saved_results = pd.read_csv(saved[0])
        self.assertEqual(len(saved_results), 8)
        self.assertEqual(saved_results.columns.tolist(), [
            'atr_short', 'atr_long', 'vwpd_threshold', 'tp', 'sl', 'max_hold',
            'sharpe', 'mdd_pct_initial', 'annualized_simple_return', 'closed_trades'])
        pd.testing.assert_frame_equal(saved_results, results, check_dtype=False, check_exact=False)
        self.assertTrue((results.atr_short < results.atr_long).all())
        for row in results.to_dict('records'):
            f = namespace['calculate_vwpd_features'](namespace['df'],row)
            target, _, _, _ = self.build(namespace['df'],f,row,self.cfg.slippage_bps)
            reference = backtest(ohlcv=namespace['df'],target_position=target,config=self.cfg)
            self.assertAlmostEqual(row['annualized_simple_return'], reference.metrics['annualized_simple_return'], delta=1e-9)
            self.assertAlmostEqual(row['mdd_pct_initial'], reference.metrics['max_drawdown_pct_initial'], delta=1e-9)
            self.assertEqual(row['closed_trades'], reference.metrics['closed_trade_count'])


if __name__ == '__main__':
    unittest.main()

"""Hand-calculated examples. No market downloads, secrets, or live calls."""
from dataclasses import replace
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from bt_crypto import BacktestConfig, backtest


def config(**overrides):
    base = BacktestConfig(asset='BTCUSDT', timeframe='1min', timestamp_label='open',
                          contract_type='linear_usdt', fx_vnd_per_usdt=1.,
                          fee_rate_per_side=0., slippage_bps=0., initial_capital_vnd=1000.,
                          fixed_margin_budget_vnd=1000.)
    return replace(base, **overrides)


def candles(opens, closes=None, index=None, freq='1min'):
    op = np.array(opens, dtype=float)
    cl = op.copy() if closes is None else np.array(closes, dtype=float)
    if index is None:
        index = pd.date_range('2025-01-01', periods=len(op), freq=freq, tz='UTC')
    return pd.DataFrame({'open':op, 'high':np.maximum(op, cl), 'low':np.minimum(op, cl),
                         'close':cl, 'volume':np.ones(len(op))}, index=index)


def run(df, targets, cfg=None, funding=None):
    pos = targets if isinstance(targets, pd.Series) else pd.Series(targets, index=df.index)
    return backtest(ohlcv=df, target_position=pos, config=cfg or config(), funding_events=funding)


class AccountingTests(unittest.TestCase):
    def reconcile(self, result):
        l, t, e, m = result.ledger, result.trades, result.events, result.metrics
        cap, fx = result.config.initial_capital_vnd, result.config.fx_vnd_per_usdt
        np.testing.assert_allclose(l.equity_vnd, cap+(l.realized_cash_pnl_usdt+l.unrealized_gross_pnl_usdt)*fx)
        np.testing.assert_allclose(l.bar_pnl_vnd.cumsum(), l.equity_vnd-cap, atol=1e-6)
        np.testing.assert_allclose(l.bar_pnl_vnd,
                                   l.bar_gross_pnl_vnd-l.fees_vnd+l.funding_cashflow_vnd, atol=1e-6)
        self.assertAlmostEqual(t.net_pnl_vnd.sum(), m['net_profit_vnd'], places=6)
        self.assertAlmostEqual(e.cash_change_usdt.sum(), l.realized_cash_pnl_usdt.iat[-1], places=8)
        self.assertAlmostEqual(t.fees_vnd.sum(), m['total_fees_vnd'], places=6)
        self.assertAlmostEqual(t.slippage_vnd.sum(), m['total_slippage_vnd'], places=6)
        self.assertAlmostEqual(result.daily_equity.net_pnl_vnd.sum(), m['net_profit_vnd'], places=6)

    def test_long_next_open_entry_bar_gap_and_same_direction(self):
        df = candles([90,100,120,130], [95,110,125,131])
        r = run(df, [1,1,0,1])
        np.testing.assert_allclose(r.ledger.equity_vnd, [1000,1100,1250,1300])
        np.testing.assert_allclose(r.ledger.bar_pnl_vnd, [0,100,150,50])
        self.assertEqual(r.trades.quantity.tolist(), [10])
        self.assertEqual(r.ledger.entry_fill_price.iloc[1:3].tolist(), [100,100])
        self.assertEqual(r.fills.kind.tolist(), ['entry','exit'])
        self.assertEqual(r.trades.entry_time.iat[0], df.index[1])
        self.assertTrue(r.diagnostics['last_target_requires_unavailable_fill'])
        self.reconcile(r)

    def test_short_next_open_pnl(self):
        r = run(candles([90,100,120,130], [95,110,125,131]), [-1,-1,0,0])
        np.testing.assert_allclose(r.ledger.equity_vnd, [1000,900,750,700])
        self.assertEqual(r.trades.gross_pnl_usdt.iat[0], -300)
        self.reconcile(r)

    def test_flip_two_legs_and_fresh_quantity(self):
        r = run(candles([90,100,120,110], [90,105,115,110]), [1,-1,0,0], config(fee_rate_per_side=.001))
        self.assertEqual(r.fills.kind.tolist(), ['entry','exit','entry','exit'])
        self.assertEqual(r.fills.time.iloc[1], r.fills.time.iloc[2])
        np.testing.assert_allclose(r.trades.quantity, [10,1000/120])
        self.assertAlmostEqual(r.metrics['gross_profit_vnd'], 200+1000/120*10)
        expected_fees = 1+1.2+1+(1000/120*110*.001)
        self.assertAlmostEqual(r.metrics['total_fees_vnd'], expected_fees)
        self.assertAlmostEqual(r.ledger.fees_vnd.iat[2], 2.2)
        self.assertEqual(r.metrics['closed_trade_count'], 2)
        self.reconcile(r)

    def test_fixed_budget_no_compounding_and_fractional_quantity(self):
        r = run(candles([90,100,120,80,100]), [1,0,1,0,0])
        np.testing.assert_allclose(r.trades.quantity, [10,12.5])
        self.assertEqual(r.metrics['net_profit_vnd'], 450)
        fractional = run(candles([200,300,330]), [1,0,0])
        self.assertAlmostEqual(fractional.trades.quantity.iat[0], 10/3)
        self.assertAlmostEqual(fractional.metrics['net_profit_vnd'], 100)
        self.reconcile(r)

    def test_terminal_open_entry_fee_mtm_no_exit_or_exit_fee(self):
        r = run(candles([90,100], [90,110]), [1,0], config(fee_rate_per_side=.001))
        self.assertEqual(r.metrics['open_trade_count'], 1)
        self.assertEqual(r.metrics['closed_trade_count'], 0)
        self.assertEqual(r.metrics['net_profit_vnd'], 99)
        self.assertEqual(r.metrics['total_fees_vnd'], 1)
        self.assertEqual(r.trades.exit_fee_usdt.iat[0], 0)
        self.assertTrue(pd.isna(r.trades.exit_time.iat[0]))
        self.assertEqual(len(r.fills), 1)
        self.reconcile(r)

    def test_last_target_cannot_fill_start_flat(self):
        for targets in ([0,0,1], [0,0,-1], [0,0,0]):
            r = run(candles([90,100,110]), targets)
            self.assertTrue(r.trades.empty)
            self.assertEqual(r.metrics['net_profit_vnd'], 0)
            self.assertEqual(r.metrics['closed_trade_count'], 0)
            self.assertIsNone(r.metrics['win_rate'])
            self.assertIn('win_rate', r.metric_reasons)
            self.reconcile(r)

    def test_leverage_fees_actual_notional_same_exit_price(self):
        # Margin 10bn VND, leverage 5, notional 50bn, fee .05% EACH side.
        cfg = config(initial_capital_vnd=10e9, fixed_margin_budget_vnd=10e9,
                     fx_vnd_per_usdt=25_000, leverage=5, fee_rate_per_side=.0005)
        r = run(candles([90,100,100]), [1,0,0], cfg)
        self.assertEqual(r.trades.quantity.iat[0], 20_000)
        np.testing.assert_allclose(r.fills.fee_usdt*25_000, [25e6,25e6])
        self.assertEqual(r.metrics['total_fees_vnd'], 50e6)
        self.assertEqual(r.metrics['net_profit_vnd'], -50e6)
        self.reconcile(r)

    def test_default_roundtrip_fee_on_actual_notional(self):
        cfg = BacktestConfig(asset='BTCUSDT', timeframe='1min', timestamp_label='open',
                             contract_type='linear_usdt', fx_vnd_per_usdt=25_000,
                             slippage_bps=0, leverage=5)
        self.assertEqual(cfg.fee_rate_per_side, 0.0005)
        same = run(candles([90,100,100]), [1,0,0], cfg)
        self.assertEqual(same.metrics['total_fees_vnd'], 50_000_000)
        different = run(candles([90,100,110]), [1,0,0], cfg)
        self.assertEqual(different.metrics['total_fees_vnd'], 52_500_000)

    def test_leverage_fees_actual_notional_different_exit_price(self):
        cfg = config(initial_capital_vnd=10e9, fixed_margin_budget_vnd=10e9,
                     fx_vnd_per_usdt=25_000, leverage=5, fee_rate_per_side=.0005)
        r = run(candles([90,100,110]), [1,0,0], cfg)
        np.testing.assert_allclose(r.fills.fee_usdt*25_000, [25e6,27.5e6])
        self.assertEqual(r.metrics['gross_profit_vnd'], 5e9)
        self.assertEqual(r.metrics['total_fees_vnd'], 52.5e6)
        self.assertEqual(r.metrics['net_profit_vnd'], 4_947_500_000)
        self.reconcile(r)

    def test_slippage_buy_sell_direction_and_not_double_charged(self):
        for d, entry, exit_price in [(1,101,108.9), (-1,99,111.1)]:
            with self.subTest(direction=d):
                r = run(candles([90,100,110]), [d,0,0], config(slippage_bps=100, fee_rate_per_side=.001))
                q = 1000/entry
                gross = d*q*(exit_price-entry)
                fees = 1000*.001+q*exit_price*.001
                slippage = q*(1+1.1)
                np.testing.assert_allclose(r.fills.fill_price, [entry,exit_price])
                self.assertAlmostEqual(r.metrics['gross_profit_vnd'], gross)
                self.assertAlmostEqual(r.metrics['net_profit_vnd'], gross-fees)
                self.assertAlmostEqual(r.metrics['total_slippage_vnd'], slippage)
                self.assertAlmostEqual(r.metrics['reference_gross_profit_before_slippage_vnd'], d*q*10)
                self.reconcile(r)

    def test_terminal_open_slippage_and_fees_reconcile(self):
        r = run(candles([90,100], [90,110]), [1,1], config(slippage_bps=100, fee_rate_per_side=.001))
        q = 1000/101
        self.assertAlmostEqual(r.metrics['net_profit_vnd'], q*9-1)
        self.assertAlmostEqual(r.metrics['total_slippage_vnd'], q)
        self.reconcile(r)

    def test_gap_fill_and_funding_during_gap(self):
        idx = pd.to_datetime(['2025-01-01T00:00Z','2025-01-01T00:01Z','2025-01-01T00:04Z'])
        df = candles([90,100,120], [95,105,125], index=idx)
        funding = pd.DataFrame({'rate':[.01,.01], 'mark_price':[110,120]},
                               index=pd.to_datetime(['2025-01-01T00:02Z','2025-01-01T00:04Z']))
        cfg = config(funding_convention='positive_rate_longs_pay', funding_at_open='before_fill')
        r = run(df, [1,0,0], cfg, funding)
        self.assertEqual(r.ledger.missing_bars_before.tolist(), [0,0,2])
        self.assertEqual(r.gaps.missing_bars.tolist(), [2])
        self.assertEqual(r.fills.time.iloc[-1], idx[-1])
        self.assertEqual(r.metrics['total_funding_cashflow_vnd'], -23)
        self.assertEqual(r.metrics['net_profit_vnd'], 177)
        self.assertEqual(len(r.ledger), 3)
        self.reconcile(r)

    def test_funding_positive_negative_rates_long_short(self):
        df = candles([100]*4)
        funding = pd.DataFrame({'rate':[.01,-.02], 'mark_price':[100,100]},
                               index=df.index[[1,2]]+pd.Timedelta(seconds=30))
        for d in [1,-1]:
            r = run(df, [d,d,0,0], config(funding_convention='positive_rate_longs_pay', funding_at_open='before_fill'), funding)
            np.testing.assert_allclose(r.events.loc[r.events.kind=='funding','funding_cashflow_usdt'], [-d*10,d*20])
            self.assertEqual(r.metrics['total_funding_cashflow_vnd'], d*10)
            self.reconcile(r)

    def test_funding_open_boundary_eligibility_explicit(self):
        df = candles([100]*3)
        funding = pd.DataFrame({'rate':[.01,.01], 'mark_price':[100,100]}, index=df.index[[1,2]])
        for policy, payments in [('before_fill',[0,-10]), ('after_fill',[-10,0])]:
            r = run(df, [1,0,0], config(funding_convention='positive_rate_longs_pay', funding_at_open=policy), funding)
            np.testing.assert_allclose(r.events.loc[r.events.kind=='funding','funding_cashflow_usdt'], payments)
            self.reconcile(r)

    def test_open_trade_and_closed_trade_reconciliation_with_funding(self):
        df = candles([90,100,120,110,100], [90,105,115,105,95])
        funding = pd.DataFrame({'rate':[.01,-.01], 'mark_price':[105,105]},
                               index=df.index[[1,3]]+pd.Timedelta(seconds=30))
        r = run(df, [1,-1,-1,-1,0], config(slippage_bps=10, fee_rate_per_side=.001,
                    funding_convention='positive_rate_longs_pay', funding_at_open='before_fill'), funding)
        self.assertEqual(r.metrics['closed_trade_count'], 1)
        self.assertEqual(r.metrics['open_trade_count'], 1)
        self.reconcile(r)

    def test_funding_on_flip_uses_declared_old_or_new_quantity(self):
        df = candles([90,100,120])
        funding = pd.DataFrame({'rate':[.01], 'mark_price':[120]}, index=df.index[[2]])
        for policy, expected, trade_id in [('before_fill',-12,1), ('after_fill',10,2)]:
            r = run(df,[1,-1,-1],config(funding_convention='positive_rate_longs_pay',funding_at_open=policy),funding)
            payments = r.events[r.events.kind=='funding']
            self.assertAlmostEqual(payments.funding_cashflow_usdt.iat[0],expected)
            self.assertEqual(payments.trade_id.iat[0],trade_id)
            self.reconcile(r)

    def test_after_open_funding_risk_flag_survives_close_price_recovery(self):
        df = candles([100,100],[100,200])
        for seconds in (0,30):
            funding = pd.DataFrame({'rate':[.6],'mark_price':[100]}, index=df.index[[1]]+pd.Timedelta(seconds=seconds))
            r = run(df,[1,1],config(leverage=2,funding_convention='positive_rate_longs_pay',funding_at_open='after_fill'),funding)
            self.assertEqual(r.ledger.equity_vnd.iat[-1],1800)
            self.assertTrue(r.ledger.equity_non_positive.iat[-1])  # -200 after funding, 1800 at Close
            self.assertTrue(r.ledger.margin_exceeds_equity.iat[-1])
            self.reconcile(r)

    def test_equity_nonpositive_and_margin_flag_do_not_stop_model(self):
        r = run(candles([100,100,10,100]), [1,1,0,0], config(leverage=2))
        self.assertTrue(r.ledger.equity_non_positive.iat[2])
        self.assertTrue(r.ledger.margin_exceeds_equity.iat[2])
        self.assertEqual(r.metrics['net_profit_vnd'], 0)
        self.assertEqual(len(r.fills), 2)
        self.reconcile(r)


class MetricTests(unittest.TestCase):
    def test_initial_peak_and_fixed_initial_drawdown_denominator(self):
        r = run(candles([100,100,100], [100,150,120]), [1,1,1])
        self.assertEqual(r.metrics['max_drawdown_vnd'], 300)
        self.assertEqual(r.metrics['max_drawdown_pct_initial'], .3)
        loss = run(candles([100,100], [100,90]), [1,1])
        self.assertEqual(loss.metrics['max_drawdown_vnd'], 100)

    def test_annualized_simple_five_years_50bn_on_10bn(self):
        first = pd.Timestamp('2020-01-01', tz='UTC')
        idx = pd.DatetimeIndex([first,first+pd.Timedelta(days=1),first+pd.Timedelta(days=1824)])
        r = run(candles([100,100,100],[100,100,600],index=idx), [1,1,1],
                config(timeframe='1D', initial_capital_vnd=10e9, fixed_margin_budget_vnd=10e9))
        self.assertEqual(r.metrics['net_profit_vnd'], 50e9)
        self.assertEqual(r.metrics['elapsed_years'], 5)
        self.assertEqual(r.metrics['annualized_simple_return'], 1)

    def test_daily_sharpe_365_sample_std_on_initial(self):
        df = candles([100]*72, np.repeat([101,103,102],24), freq='1h')
        r = run(df, [1]*72, config(timeframe='1h'))
        expected = np.array([.01,.02,-.01])
        np.testing.assert_allclose(r.daily_equity.return_on_initial, expected)
        self.assertAlmostEqual(r.metrics['sharpe'], expected.mean()/expected.std(ddof=1)*np.sqrt(365))
        self.assertEqual(r.diagnostics['sharpe_days_used'], 3)

    def test_full_flat_days_zero_pnl_undefined_sharpe(self):
        r = run(candles([100]*48,freq='1h'), [0]*48, config(timeframe='1h'))
        self.assertEqual(r.daily_equity.net_pnl_vnd.tolist(), [0,0])
        self.assertIsNone(r.metrics['sharpe'])
        self.assertIn('zero', r.metric_reasons['sharpe'])

    def test_entire_missing_day_not_filled_or_used_for_sharpe(self):
        idx = pd.date_range('2025-01-01',periods=72,freq='1h',tz='UTC').delete(slice(24,48))
        r = run(candles([100]*48,index=idx), [1]*48, config(timeframe='1h'))
        self.assertEqual(r.daily_equity.observed_bars.tolist(), [24,0,24])
        self.assertTrue(pd.isna(r.daily_equity.net_pnl_vnd.iat[1]))
        self.assertTrue(pd.isna(r.daily_equity.equity_vnd.iat[1]))
        self.assertEqual(r.daily_equity.usable_for_sharpe.tolist(), [True,False,False])
        self.assertIsNone(r.metrics['sharpe'])

    def test_margin_turnover_both_legs_divide_leverage(self):
        r = run(candles([90,100,110]), [1,0,0], config(leverage=5))
        # Actual notionals 5000 + 5500; / leverage 5 / initial 1000 = 2.1
        self.assertAlmostEqual(r.metrics['average_daily_margin_turnover_on_initial'], 2.1)

    def test_zero_net_trade_not_a_win_holding_duration(self):
        r = run(candles([100]*3), [1,0,0])
        self.assertEqual(r.metrics['win_rate'], 0)
        self.assertEqual(r.metrics['average_net_pnl_per_closed_trade_vnd'], 0)
        self.assertEqual(r.metrics['average_closed_holding_duration'], pd.Timedelta(minutes=1))

    def test_no_funding_reported_as_unknown_not_free(self):
        df = candles([100]*3)
        r = run(df, [1,0,0])
        self.assertFalse(r.diagnostics['funding_included'])
        self.assertIsNone(r.metrics['total_funding_cashflow_vnd'])
        self.assertIn('NOT included', r.diagnostics['cost_scope'])
        empty = pd.DataFrame({'rate':pd.Series(dtype=float),'mark_price':pd.Series(dtype=float)},
                             index=pd.DatetimeIndex([],tz='UTC'))
        supplied = run(df,[1,0,0],config(funding_convention='positive_rate_longs_pay', funding_at_open='before_fill'),empty)
        self.assertTrue(supplied.diagnostics['funding_included'])
        self.assertFalse(supplied.diagnostics['funding_coverage_verified'])
        self.assertEqual(supplied.metrics['total_funding_cashflow_vnd'], 0)


class ValidationAndCausalityTests(unittest.TestCase):
    def test_default_vnd_capital_budget_and_explicit_costs(self):
        cfg = BacktestConfig(asset='BTCUSDT',timeframe='1min',timestamp_label='open',
                             contract_type='linear_usdt',fx_vnd_per_usdt=25000,
                             fee_rate_per_side=0,slippage_bps=0)
        self.assertEqual(cfg.initial_capital_vnd,10e9)
        self.assertEqual(cfg.fixed_margin_budget_vnd,10e9)
        self.assertEqual(cfg.leverage,1)

    def test_nan_invalid_position_no_normalization_or_implicit_fill(self):
        df = candles([100]*3)
        for targets in ([0,np.nan,1],[0,2,0],[0,-2,0],[0,.5,0],['0','1','0'],[False,True,False]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                run(df,targets)

    def test_exact_index_and_utc_open_semantics(self):
        df = candles([100]*3)
        p = pd.Series([1,0,0],index=df.index+pd.Timedelta(minutes=1))
        with self.assertRaisesRegex(ValueError, 'exactly match'):
            run(df,p)
        naive = df.copy()
        naive.index = naive.index.tz_localize(None)
        with self.assertRaisesRegex(ValueError, 'UTC'):
            run(naive,[1,0,0])
        with self.assertRaisesRegex(ValueError, 'timestamp_label'):
            config(timestamp_label='close')
        wrong = df.copy()
        wrong['open_time'] = (df.index.as_unit('ns').asi8//1_000_000)+60_000
        with self.assertRaisesRegex(ValueError, 'open_time'):
            run(wrong,[1,0,0])
        correct = df.copy()
        correct['open_time'] = df.index.as_unit('ns').asi8//1_000_000
        correct['close_time'] = correct.open_time+59_999
        run(correct,[1,0,0])
        correct.loc[correct.index[0],'close_time'] += 1
        with self.assertRaisesRegex(ValueError, 'close_time'):
            run(correct,[1,0,0])

    def test_index_duplicates_unsorted_bad_prices_and_multiple_assets(self):
        df = candles([100]*3)
        cases = []
        bad = df.copy(); bad.index = pd.DatetimeIndex([df.index[0]]*3); cases.append(bad)
        cases.append(df.iloc[::-1])
        bad = df.copy(); bad.loc[bad.index[0],'open'] = np.nan; cases.append(bad)
        bad = df.copy(); bad.loc[bad.index[0],'high'] = 99; cases.append(bad)
        bad = df.copy(); bad['symbol'] = ['BTCUSDT','ETHUSDT','BTCUSDT']; cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):
                run(bad,[1,0,0])

    def test_explicit_fx_fees_slippage_linear_contract_and_funding(self):
        for kwargs in ({'fx_vnd_per_usdt':0}, {'fee_rate_per_side':-1}, {'slippage_bps':10000},
                       {'leverage':0},{'contract_type':'inverse'}, {'funding_convention':'unknown'}):
            with self.assertRaises(ValueError): config(**kwargs)
        with self.assertRaises(TypeError):
            BacktestConfig(asset='BTCUSDT',timeframe='1min',timestamp_label='open',contract_type='linear_usdt')
        df = candles([100]*3)
        funding = pd.DataFrame({'rate':[.01],'mark_price':[100]}, index=df.index[[1]])
        with self.assertRaisesRegex(ValueError, 'explicit'):
            run(df,[1,0,0],funding=funding)
        with self.assertRaisesRegex(ValueError, 'without funding'):
            run(df,[1,0,0],config(funding_convention='positive_rate_longs_pay'))

    def test_no_lookahead_prefix_invariance_and_reproducibility(self):
        df = candles([90,100,120,110,80],[95,105,115,100,85])
        target = pd.Series([1,-1,-1,0,1],index=df.index)
        original_df, original_target = df.copy(deep=True), target.copy(deep=True)
        cfg = config(fee_rate_per_side=.001,slippage_bps=10)
        full = run(df,target,cfg)
        prefix = run(df.iloc[:3],target.iloc[:3],cfg)
        assert_frame_equal(full.ledger.iloc[:3],prefix.ledger)
        changed = df.copy()
        changed.iloc[3:] *= 2
        other = run(changed, [1,-1,-1,1,-1], cfg)
        assert_frame_equal(full.ledger.iloc[:3],other.ledger.iloc[:3])
        repeated = run(df,target,cfg)
        for name in ('ledger','events','trades','daily_equity','gaps'):
            assert_frame_equal(getattr(full,name),getattr(repeated,name))
        self.assertEqual(full.metrics,repeated.metrics)
        assert_frame_equal(df,original_df)
        pd.testing.assert_series_equal(target,original_target)

    def test_current_close_target_cannot_change_current_bar_execution(self):
        df = candles([90,100,120,110])
        original = run(df,[1,1,1,1])
        changed = run(df,[1,1,-1,1])
        state = ['position','quantity','entry_fill_price','equity_vnd','bar_pnl_vnd']
        assert_frame_equal(original.ledger[state].iloc[:3],changed.ledger[state].iloc[:3])
        self.assertEqual(original.ledger.position.iat[3],1)
        self.assertEqual(changed.ledger.position.iat[3],-1)


if __name__ == '__main__':
    unittest.main()

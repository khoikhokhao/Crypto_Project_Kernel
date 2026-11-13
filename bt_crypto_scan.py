"""Fast single-asset signal scanner; detailed accounting remains in bt_crypto.

Signals are entry decisions at bar close (-1, 0, +1). While holding, TP/SL
and max_hold determine the exit target. Funding is deliberately unsupported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange, get_num_threads, set_num_threads

from bt_crypto import BacktestConfig, DAY_NS, YEAR_SECONDS, _inputs


@njit(parallel=True, cache=True)
def _scan(opens, highs, lows, closes, signals, days, day_valid, risk,
          capital, budget, fx, leverage, fee_rate, slip):
    # Independent parameter rows; time remains sequential inside each row.
    out = np.empty((risk.shape[0], 10), dtype=np.float64)
    for k in prange(risk.shape[0]):
        tp, sl, max_hold = risk[k, 0], risk[k, 1], int(risk[k, 2])
        position, target, entry_bar = 0, 0, -1
        quantity = entry = entry_fee = cash = gross_realized = 0.0
        fees = slippage = traded = trade_net_sum = 0.0
        closed_count = wins = 0
        peak, mdd, previous_equity = capital, 0.0, capital
        daily_sum, daily_mean, daily_m2 = 0.0, 0.0, 0.0
        valid_days, current_day = 0, days[0]
        for i in range(opens.size):
            day = days[i]
            if day != current_day:
                if day_valid[current_day]:
                    value = daily_sum / capital
                    valid_days += 1
                    delta = value - daily_mean
                    daily_mean += delta / valid_days
                    daily_m2 += delta * (value - daily_mean)
                daily_sum = 0.0
                current_day = day
            raw = opens[i]
            if target != position:
                if position != 0:
                    fill = raw * (1 - position * slip)
                    gross = position * quantity * (fill - entry)
                    exit_fee = quantity * fill * fee_rate
                    cash += gross - exit_fee
                    gross_realized += gross
                    fees += exit_fee
                    slippage += quantity * abs(fill - raw)
                    traded += quantity * fill
                    closed_count += 1
                    net_trade = gross - entry_fee - exit_fee
                    trade_net_sum += net_trade
                    if net_trade > 0:
                        wins += 1
                    position, quantity, entry, entry_bar = 0, 0.0, 0.0, -1
                if target != 0:
                    position = target
                    entry = raw * (1 + position * slip)
                    quantity = (budget / fx * leverage) / entry
                    entry_fee = quantity * entry * fee_rate
                    cash -= entry_fee
                    fees += entry_fee
                    slippage += quantity * abs(entry - raw)
                    traded += quantity * entry
                    entry_bar = i
            equity = capital + (cash + position * quantity * (closes[i] - entry)) * fx
            if equity > peak:
                peak = equity
            if peak - equity > mdd:
                mdd = peak - equity
            daily_sum += equity - previous_equity
            previous_equity = equity
            # Exit is decided after the bar closes; next observed open fills it.
            if position != 0:
                if position == 1:
                    stop_hit = lows[i] <= entry * (1 - sl)
                    profit_hit = highs[i] >= entry * (1 + tp)
                else:
                    stop_hit = highs[i] >= entry * (1 + sl)
                    profit_hit = lows[i] <= entry * (1 - tp)
                if stop_hit or profit_hit or i - entry_bar + 1 >= max_hold:
                    target = 0
                else:
                    target = position
            else:
                target = int(signals[i])
        if day_valid[current_day]:
            value = daily_sum / capital
            valid_days += 1
            delta = value - daily_mean
            daily_mean += delta / valid_days
            daily_m2 += delta * (value - daily_mean)
        sharpe = np.nan
        if valid_days >= 2 and daily_m2 > 0:
            sharpe = daily_mean / np.sqrt(daily_m2 / (valid_days - 1)) * np.sqrt(365.0)
        out[k, 0] = previous_equity - capital
        out[k, 1] = mdd
        out[k, 2] = sharpe
        out[k, 3] = closed_count
        out[k, 4] = 1 if position else 0
        out[k, 5] = wins / closed_count if closed_count else np.nan
        out[k, 6] = trade_net_sum * fx / closed_count if closed_count else np.nan
        out[k, 7] = fees * fx
        out[k, 8] = slippage * fx
        out[k, 9] = traded * fx / leverage
    return out


def scan_risk_grid(*, ohlcv: pd.DataFrame, entry_signals: pd.Series,
                   risk_grid: pd.DataFrame, config: BacktestConfig,
                   workers: int | None = None, batch_size: int = 256) -> pd.DataFrame:
    """Scan TP/SL/hold for one entry-signal series; no funding or full ledger.

    Entry signals use -1, 0, +1 at close and are considered only while flat.
    TP/SL use the observed bar's high/low after entry and exit at next open;
    both-hit precedence is immaterial to the next-open exit price.
    """
    if config.funding_convention is not None or config.funding_at_open is not None:
        raise ValueError('Fast scan does not support funding; use bt_crypto.backtest')
    if not isinstance(risk_grid, pd.DataFrame) or not {'tp', 'sl', 'max_hold'}.issubset(risk_grid):
        raise ValueError('risk_grid requires tp, sl, max_hold columns')
    if risk_grid.empty:
        raise ValueError('risk_grid must not be empty')
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    values = risk_grid[['tp', 'sl', 'max_hold']].to_numpy(dtype=np.float64)
    if (not np.isfinite(values).all() or (values[:, 0] <= 0).any()
            or (values[:, 1] <= 0).any() or (values[:, 1] >= 1).any()
            or (values[:, 2] < 1).any() or (values[:, 2] != np.floor(values[:, 2])).any()):
        raise ValueError('Require tp > 0, 0 < sl < 1, integer max_hold >= 1')
    idx, times, step, opens, closes, sig, _, _, _ = _inputs(ohlcv, entry_signals, config, None)
    highs = np.ascontiguousarray(ohlcv.high.to_numpy(dtype=np.float64))
    lows = np.ascontiguousarray(ohlcv.low.to_numpy(dtype=np.float64))
    opens = np.ascontiguousarray(opens)
    closes = np.ascontiguousarray(closes)
    sig = np.ascontiguousarray(sig)
    day_abs = (times + step - 1) // DAY_NS
    days = np.ascontiguousarray(day_abs - day_abs[0], dtype=np.int64)
    n_days = int(days[-1] + 1)
    counts = np.bincount(days, minlength=n_days)
    gap = np.r_[False, np.diff(times) > step]
    gap_days = np.bincount(days, weights=gap.astype(np.int8), minlength=n_days) > 0
    day_valid = (counts == DAY_NS // step) & ~gap_days
    observed_days = int((counts > 0).sum())
    years = (int(times[-1]) + step - int(times[0])) / 1e9 / YEAR_SECONDS
    original_threads = get_num_threads()
    if workers is not None:
        if not isinstance(workers, int) or workers < 1 or workers > original_threads:
            raise ValueError(f'workers must be between 1 and {original_threads}')
        set_num_threads(workers)
    try:
        chunks = []
        for start in range(0, len(values), batch_size):
            chunks.append(_scan(opens, highs, lows, closes, sig, days, day_valid,
                                np.ascontiguousarray(values[start:start+batch_size]),
                                config.initial_capital_vnd, config.fixed_margin_budget_vnd,
                                config.fx_vnd_per_usdt, config.leverage,
                                config.fee_rate_per_side, config.slippage_bps / 10_000))
        result = np.vstack(chunks)
    finally:
        if workers is not None:
            set_num_threads(original_threads)
    out = risk_grid.reset_index(drop=True).copy()
    out['net_profit_vnd'] = result[:, 0]
    out['annualized_simple_return'] = result[:, 0] / config.initial_capital_vnd / years
    out['mdd_vnd'] = result[:, 1]
    out['mdd_pct_initial'] = result[:, 1] / config.initial_capital_vnd
    out['sharpe'] = result[:, 2]
    out['sharpe_reason'] = np.where(np.isnan(result[:, 2]),
        'Need two complete gap-free UTC days with non-zero sample volatility', '')
    out['closed_trades'] = result[:, 3].astype(np.int64)
    out['open_trades'] = result[:, 4].astype(np.int64)
    out['win_rate'] = result[:, 5]
    out['win_rate_reason'] = np.where(result[:, 3] == 0, 'No closed trades', '')
    out['avg_net_per_trade_vnd'] = result[:, 6]
    out['avg_net_per_trade_reason'] = np.where(result[:, 3] == 0, 'No closed trades', '')
    out['total_fees_vnd'] = result[:, 7]
    out['total_slippage_vnd'] = result[:, 8]
    out['avg_daily_margin_turnover'] = result[:, 9] / config.initial_capital_vnd / observed_days
    out['missing_bars'] = int((np.diff(times) // step - 1).sum())
    out['funding_included'] = False
    return out

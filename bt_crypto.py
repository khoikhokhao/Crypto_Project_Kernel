"""Single-asset, fixed-budget, linear-USDT futures backtest. See docs/bt_crypto.md."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

YEAR_SECONDS = 365 * 24 * 60 * 60
DAY_NS = 86_400_000_000_000


@dataclass(frozen=True, kw_only=True)
class BacktestConfig:
    asset: str
    timeframe: str
    timestamp_label: str
    contract_type: str
    fx_vnd_per_usdt: float
    fee_rate_per_side: float = 0.0005
    slippage_bps: float
    initial_capital_vnd: float = 10_000_000_000
    fixed_margin_budget_vnd: float = 10_000_000_000
    leverage: float = 1.0
    funding_convention: str | None = None
    funding_at_open: str | None = None

    def __post_init__(self):
        if not isinstance(self.asset, str) or not self.asset.strip():
            raise ValueError('asset must identify ONE asset')
        if self.contract_type != 'linear_usdt':
            raise ValueError('Only contract_type="linear_usdt" is supported; inverse/coin-margined contracts are unsupported')
        if self.timestamp_label != 'open':
            raise ValueError('Explicit timestamp_label="open" is required')
        for key in ('fx_vnd_per_usdt', 'initial_capital_vnd', 'fixed_margin_budget_vnd', 'leverage'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ('fee_rate_per_side', 'slippage_bps'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be explicitly configured, finite and non-negative')
        if self.slippage_bps >= 10_000:
            raise ValueError('slippage_bps must be below 10000 to keep SELL fills positive')
        step = pd.Timedelta(self.timeframe)
        if step.value <= 0 or step.value > DAY_NS or DAY_NS % step.value:
            raise ValueError('timeframe must be a fixed positive duration dividing one UTC day (e.g. 1min, 15min, 4h, 1D)')
        if self.funding_convention not in (None, 'positive_rate_longs_pay'):
            raise ValueError('Supported funding_convention: positive_rate_longs_pay')
        if self.funding_at_open not in (None, 'before_fill', 'after_fill'):
            raise ValueError('funding_at_open must be before_fill or after_fill')


def _utc_index(index, name):
    if not isinstance(index, pd.DatetimeIndex) or str(index.tz) != 'UTC':
        raise ValueError(f'{name} must be a UTC DatetimeIndex; explicitly localize/convert it first')
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(f'{name} must be sorted, unique, and contain no NaT')
    return index.as_unit('ns')


def _inputs(ohlcv, target_position, config, funding_events):
    if not isinstance(config, BacktestConfig):
        raise TypeError('config must be BacktestConfig')
    if not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
        raise ValueError('ohlcv must be a non-empty DataFrame for ONE asset')
    if not ohlcv.columns.is_unique:
        raise ValueError('Duplicate OHLCV column names')
    idx = _utc_index(ohlcv.index, 'OHLCV index')
    step = pd.Timedelta(config.timeframe).value
    times = idx.asi8
    if (times % step != 0).any():
        raise ValueError('Open labels must align to the configured timeframe on the UTC epoch grid')
    if not isinstance(target_position, pd.Series):
        raise TypeError('target_position must be a Series with the exact OHLCV index')
    _utc_index(target_position.index, 'target_position index')
    if not ohlcv.index.equals(target_position.index):
        raise ValueError('target_position index must exactly match OHLCV: no reindex, shift or fill is performed')
    if target_position.isna().any():
        raise ValueError('target_position contains NaN; researcher must explicitly resolve warmup/missing signals')
    if (pd.api.types.is_bool_dtype(target_position.dtype)
            or pd.api.types.is_complex_dtype(target_position.dtype)
            or not pd.api.types.is_numeric_dtype(target_position.dtype)
            or not target_position.isin([-1, 0, 1]).all()):
        raise ValueError('target_position accepts only numeric -1, 0, +1; no normalization')
    columns = ['open', 'high', 'low', 'close', 'volume']
    if any(c not in ohlcv for c in columns):
        raise ValueError(f'OHLCV requires {columns}')
    if any(not pd.api.types.is_numeric_dtype(ohlcv[c]) for c in columns):
        raise ValueError('OHLCV columns must already be numeric')
    values = ohlcv[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('OHLCV contains NaN or non-finite values; no synthetic bars/fill is performed')
    op, hi, lo, cl, vol = values.T
    if ((values[:, :4] <= 0).any() or (vol < 0).any()
            or (hi < np.maximum.reduce([op, lo, cl])).any()
            or (lo > np.minimum.reduce([op, hi, cl])).any()):
        raise ValueError('Invalid OHLCV prices, bounds, or volume')
    if 'open_time' in ohlcv:
        expected = pd.to_datetime(ohlcv.open_time.to_numpy(), unit='ms', utc=True).as_unit('ns')
        if not idx.equals(expected):
            raise ValueError('open_time does not match index: labels must denote candle OPEN')
    if 'close_time' in ohlcv:
        expected = pd.to_datetime(ohlcv.close_time.to_numpy(), unit='ms', utc=True).as_unit('ns').asi8
        if not np.array_equal(expected, times + step - 1_000_000):
            raise ValueError('close_time disagrees with timeframe/open labels (expected inclusive millisecond close)')
    for name in ('symbol', 'asset'):
        if name in ohlcv and (ohlcv[name].nunique(dropna=False) != 1
                             or str(ohlcv[name].iat[0]) != config.asset):
            raise ValueError(f'{name} column must contain only config.asset')
    if funding_events is None:
        if config.funding_convention is not None or config.funding_at_open is not None:
            raise ValueError('Funding settings supplied without funding_events; pass an explicit empty table if intentional')
        ft = np.empty(0, dtype=np.int64)
        rates = marks = np.empty(0, dtype=np.float64)
    else:
        if config.funding_convention is None or config.funding_at_open is None:
            raise ValueError('Funding requires explicit funding_convention AND funding_at_open from the source/experiment')
        if not isinstance(funding_events, pd.DataFrame) or not funding_events.columns.is_unique:
            raise ValueError('funding_events must be a DataFrame with unique columns')
        fi = _utc_index(funding_events.index, 'funding_events index')
        if not {'rate', 'mark_price'}.issubset(funding_events):
            raise ValueError('funding_events requires rate and settlement mark_price; no forward-filled/future prices')
        rates = funding_events.rate.to_numpy(dtype=float)
        marks = funding_events.mark_price.to_numpy(dtype=float)
        ft = fi.asi8
        if not np.isfinite(rates).all() or not np.isfinite(marks).all() or (marks <= 0).any():
            raise ValueError('Funding requires finite rates and positive mark_price')
        if len(ft) and (ft[0] < times[0] or ft[-1] >= times[-1] + step):
            raise ValueError('Funding timestamps must lie inside [first open, final bar end); slice explicitly')
    return idx, times, step, op, cl, target_position.to_numpy(dtype=np.int8), ft, rates, marks


@dataclass
class BacktestResult:
    config: BacktestConfig
    metrics: dict[str, Any]
    metric_reasons: dict[str, str]
    trades: pd.DataFrame
    ledger: pd.DataFrame
    events: pd.DataFrame
    daily_equity: pd.DataFrame
    gaps: pd.DataFrame
    diagnostics: dict[str, Any]

    @property
    def fills(self):
        return self.events[self.events.kind.isin(['entry', 'exit'])].copy()

    def summary(self):
        """Notebook-friendly metric table; undefined values keep their reasons."""
        return pd.DataFrame({'value': pd.Series(self.metrics, dtype=object),
                             'reason': pd.Series(self.metric_reasons)}).fillna({'reason': ''})

    def _repr_html_(self):
        import html
        note = html.escape(self.diagnostics['cost_scope'])
        return f'<p>{html.escape(self.config.asset)} | {note}</p>' + self.summary().to_html(na_rep='undefined')

    def plot(self, *, figsize=(12, 7)):
        """Return a Matplotlib figure; never writes files. Import is optional until called."""
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]}, constrained_layout=True)
        times = pd.to_datetime(np.r_[self.ledger.index.asi8[0], self.ledger.mark_time.array.as_unit('ns').asi8], utc=True)
        equity = np.r_[self.config.initial_capital_vnd, self.ledger.equity_vnd.to_numpy()]
        axes[0].plot(times, equity / 1e9, linewidth=1)
        axes[0].axhline(self.config.initial_capital_vnd / 1e9, color='gray', linestyle='--', linewidth=.7)
        axes[0].set(ylabel='Equity (billion VND)', title=f'{self.config.asset} | {self.diagnostics["cost_scope"]}')
        dd = np.r_[0., self.ledger.drawdown_pct.to_numpy()] * 100
        axes[1].fill_between(times, -dd, 0, color='firebrick', alpha=.45)
        axes[1].set(ylabel='Drawdown (% initial)', xlabel='UTC')
        for ax in axes:
            ax.grid(alpha=.2)
        return figure


def _daily(ledger, capital, step):
    # Mark at the END of a candle; midnight end belongs to the preceding UTC day.
    days = (pd.DatetimeIndex(ledger.mark_time) - pd.Timedelta(1, 'ns')).normalize()
    calendar = pd.date_range(days[0], days[-1], freq='D', name='utc_day')
    tmp = pd.DataFrame({
        'equity_vnd': ledger.equity_vnd.to_numpy(), 'net_pnl_vnd': ledger.bar_pnl_vnd.to_numpy(),
        'margin_turnover_vnd': ledger.margin_turnover_vnd.to_numpy(),
        'observed_bars': np.ones(len(ledger), dtype=np.int64),
        'gap_affected': ledger.missing_bars_before.to_numpy() > 0,
    }, index=days)
    group = tmp.groupby(level=0)
    out = pd.DataFrame(index=calendar)
    out['equity_vnd'] = group.equity_vnd.last().reindex(calendar)
    for key in ('net_pnl_vnd', 'margin_turnover_vnd'):
        out[key] = group[key].sum(min_count=1).reindex(calendar)
    out['observed_bars'] = group.observed_bars.sum().reindex(calendar, fill_value=0)
    out['gap_affected'] = group.gap_affected.any().reindex(calendar, fill_value=False)
    out['complete_day'] = out.observed_bars == DAY_NS // step
    out['usable_for_sharpe'] = out.complete_day & ~out.gap_affected
    out['return_on_initial'] = out.net_pnl_vnd / capital
    out['margin_turnover_on_initial'] = out.margin_turnover_vnd / capital
    return out


def backtest(*, ohlcv: pd.DataFrame, target_position: pd.Series,
             config: BacktestConfig, funding_events: pd.DataFrame | None = None) -> BacktestResult:
    """Targets decided after Close[t] execute only at the next observed Open.

    Slippage changes actual fill prices. Gross PnL already includes its effect;
    the slippage ledger column is attribution, NOT an additional cash charge.
    Funding at an exact Open uses the explicitly selected before/after-fill rule.
    """
    idx, times, step, opens, closes, targets, ft, rates, marks = _inputs(
        ohlcv, target_position, config, funding_events)
    fx, capital = config.fx_vnd_per_usdt, config.initial_capital_vnd
    budget, lev = config.fixed_margin_budget_vnd, config.leverage
    slip, fee_rate = config.slippage_bps / 10_000, config.fee_rate_per_side
    n = len(times)
    float_keys = ['quantity', 'entry_fill_price', 'unrealized_gross_pnl_usdt',
                  'realized_gross_pnl_usdt', 'realized_cash_pnl_usdt', 'gross_pnl_usdt',
                  'equity_vnd', 'fees_vnd', 'slippage_vnd', 'funding_cashflow_vnd',
                  'traded_notional_vnd', 'margin_turnover_vnd', 'margin_required_vnd']
    a = {key: np.zeros(n, dtype=np.float64) for key in float_keys}
    positions = np.zeros(n, dtype=np.int8)
    nonpositive = np.zeros(n, dtype=bool)
    margin_excess = np.zeros(n, dtype=bool)
    gap_counts = np.r_[0, np.diff(times) // step - 1].astype(np.int64)
    cash = realized_gross = total_fees = total_slip = total_funding = 0.0
    direction, quantity, entry, funding_cursor = 0, 0.0, 0.0, 0
    trades, events = [], []
    active = None
    funding_nonpositive = funding_margin_excess = False

    def event(kind, time_ns, *, side=0, raw=np.nan, fill=np.nan, gross=0., fee=0.,
              slippage=0., funding=0., rate=np.nan, mark=np.nan, signal_ns=None):
        events.append({'kind': kind, 'time': pd.Timestamp(time_ns, tz='UTC'),
                       'trade_id': active['trade_id'] if active else None,
                       'signal_time': pd.Timestamp(signal_ns, tz='UTC') if signal_ns is not None else pd.NaT,
                       'direction': direction, 'side': side, 'quantity': quantity,
                       'raw_open': raw, 'fill_price': fill,
                       'realized_gross_pnl_usdt': gross, 'fee_usdt': fee,
                       'slippage_usdt': slippage, 'funding_cashflow_usdt': funding,
                       'cash_change_usdt': gross - fee + funding,
                       'funding_rate': rate, 'funding_mark_price': mark})

    def settle(j):
        nonlocal cash, total_funding, funding_nonpositive, funding_margin_excess
        payment = -direction * quantity * marks[j] * rates[j]
        cash += payment
        total_funding += payment
        if active is not None:
            active['funding_cashflow_usdt'] += payment
        settlement_equity = capital + (cash + direction * quantity * (marks[j] - entry)) * fx
        funding_nonpositive |= settlement_equity <= 0
        funding_margin_excess |= bool(direction and budget > settlement_equity)
        event('funding', int(ft[j]), funding=payment, rate=rates[j], mark=marks[j])

    for i in range(n):
        t, raw = int(times[i]), opens[i]
        previous_fees, previous_slip, previous_funding = total_fees, total_slip, total_funding
        traded = 0.
        funding_nonpositive = funding_margin_excess = False
        # Includes settlement during data gaps: old quantity persists until next observed Open.
        while funding_cursor < len(ft) and ft[funding_cursor] < t:
            settle(funding_cursor)
            funding_cursor += 1
        if config.funding_at_open == 'before_fill':
            while funding_cursor < len(ft) and ft[funding_cursor] == t:
                settle(funding_cursor)
                funding_cursor += 1
        desired = int(targets[i-1]) if i else 0
        signal_ns = int(times[i-1] + step - 1) if i else None
        if desired != direction:
            if direction:
                side = -direction
                fill = raw * (1 + side * slip)
                gross = direction * quantity * (fill - entry)
                fee = quantity * fill * fee_rate
                slippage = quantity * abs(fill - raw)
                cash += gross - fee
                realized_gross += gross
                total_fees += fee
                total_slip += slippage
                traded += quantity * fill
                event('exit', t, side=side, raw=raw, fill=fill, gross=gross, fee=fee,
                      slippage=slippage, signal_ns=signal_ns)
                active.update(status='closed', exit_time=pd.Timestamp(t, tz='UTC'),
                              exit_raw_open=raw, exit_fill_price=fill,
                              exit_fee_usdt=fee, exit_slippage_usdt=slippage,
                              gross_pnl_usdt=gross, mark_price=fill, mark_time=pd.Timestamp(t, tz='UTC'))
                active['holding_duration'] = active['exit_time'] - active['entry_time']
                active = None
                direction, quantity, entry = 0, 0., 0.
            if desired:
                direction = desired
                fill = raw * (1 + direction * slip)
                quantity = (budget / fx * lev) / fill
                entry = fill
                fee = quantity * fill * fee_rate  # quantity already contains leverage
                slippage = quantity * abs(fill - raw)
                cash -= fee
                total_fees += fee
                total_slip += slippage
                traded += quantity * fill
                active = {'trade_id': len(trades)+1, 'status': 'open', 'direction': direction,
                          'quantity': quantity, 'signal_time': pd.Timestamp(signal_ns, tz='UTC'),
                          'entry_time': pd.Timestamp(t, tz='UTC'), 'entry_raw_open': raw,
                          'entry_fill_price': fill, 'entry_fee_usdt': fee,
                          'entry_slippage_usdt': slippage, 'funding_cashflow_usdt': 0.,
                          'exit_time': pd.NaT, 'exit_raw_open': np.nan, 'exit_fill_price': np.nan,
                          'exit_fee_usdt': 0., 'exit_slippage_usdt': 0.}
                trades.append(active)
                event('entry', t, side=direction, raw=raw, fill=fill, fee=fee,
                      slippage=slippage, signal_ns=signal_ns)
        if config.funding_at_open == 'after_fill':
            while funding_cursor < len(ft) and ft[funding_cursor] == t:
                settle(funding_cursor)
                funding_cursor += 1
        open_equity = capital + (cash + direction * quantity * (raw - entry)) * fx
        while funding_cursor < len(ft) and ft[funding_cursor] < t + step:
            settle(funding_cursor)
            funding_cursor += 1
        unrealized = direction * quantity * (closes[i] - entry)
        equity = capital + (cash + unrealized) * fx
        positions[i] = direction
        a['quantity'][i] = quantity
        a['entry_fill_price'][i] = entry if direction else np.nan
        a['unrealized_gross_pnl_usdt'][i] = unrealized
        a['realized_gross_pnl_usdt'][i] = realized_gross
        a['realized_cash_pnl_usdt'][i] = cash
        a['gross_pnl_usdt'][i] = realized_gross + unrealized
        a['equity_vnd'][i] = equity
        a['fees_vnd'][i] = (total_fees - previous_fees) * fx
        a['slippage_vnd'][i] = (total_slip - previous_slip) * fx
        a['funding_cashflow_vnd'][i] = (total_funding - previous_funding) * fx
        a['traded_notional_vnd'][i] = traded * fx
        a['margin_turnover_vnd'][i] = traded * fx / lev
        a['margin_required_vnd'][i] = budget if direction else 0.
        nonpositive[i] = min(equity, open_equity) <= 0 or funding_nonpositive
        margin_excess[i] = bool(direction and budget > min(equity, open_equity)) or funding_margin_excess
    if active is not None:
        active.update(gross_pnl_usdt=direction * quantity * (closes[-1] - entry),
                      mark_price=closes[-1], mark_time=pd.Timestamp(int(times[-1] + step), tz='UTC'))
        active['holding_duration'] = active['mark_time'] - active['entry_time']

    ledger = pd.DataFrame(a, index=idx.rename('timestamp'))
    ledger.insert(0, 'mark_time', pd.to_datetime(times + step, utc=True))
    ledger['target_position'] = targets
    ledger['executed_target'] = np.r_[np.int8(0), targets[:-1]]
    ledger['position'] = positions
    ledger['missing_bars_before'] = gap_counts
    ledger['equity_non_positive'] = nonpositive
    ledger['margin_exceeds_equity'] = margin_excess
    equity = ledger.equity_vnd.to_numpy()
    ledger['bar_pnl_vnd'] = np.diff(np.r_[capital, equity])
    ledger['bar_gross_pnl_vnd'] = np.diff(np.r_[0., ledger.gross_pnl_usdt.to_numpy()]) * fx
    ledger['drawdown_vnd'] = np.maximum.accumulate(np.r_[capital, equity])[1:] - equity
    ledger['drawdown_pct'] = ledger.drawdown_vnd / capital
    trade_columns = ['trade_id', 'status', 'direction', 'quantity', 'signal_time', 'entry_time',
                     'entry_raw_open', 'entry_fill_price', 'entry_fee_usdt', 'entry_slippage_usdt',
                     'funding_cashflow_usdt', 'exit_time', 'exit_raw_open', 'exit_fill_price',
                     'exit_fee_usdt', 'exit_slippage_usdt', 'gross_pnl_usdt', 'mark_price',
                     'mark_time', 'holding_duration']
    trade_frame = pd.DataFrame(trades, columns=trade_columns)
    trade_frame['fees_usdt'] = trade_frame.entry_fee_usdt + trade_frame.exit_fee_usdt
    trade_frame['slippage_usdt'] = trade_frame.entry_slippage_usdt + trade_frame.exit_slippage_usdt
    trade_frame['net_pnl_usdt'] = trade_frame.gross_pnl_usdt - trade_frame.fees_usdt + trade_frame.funding_cashflow_usdt
    trade_frame['reference_gross_pnl_usdt'] = trade_frame.gross_pnl_usdt + trade_frame.slippage_usdt
    for key in ('gross_pnl', 'net_pnl', 'fees', 'slippage', 'funding_cashflow', 'reference_gross_pnl'):
        trade_frame[key + '_vnd'] = trade_frame[key + '_usdt'] * fx
    event_columns = ['kind', 'time', 'trade_id', 'signal_time', 'direction', 'side', 'quantity',
                     'raw_open', 'fill_price', 'realized_gross_pnl_usdt', 'fee_usdt', 'slippage_usdt',
                     'funding_cashflow_usdt', 'cash_change_usdt', 'funding_rate', 'funding_mark_price']
    event_frame = pd.DataFrame(events, columns=event_columns)
    daily = _daily(ledger, capital, step)
    closed = trade_frame[trade_frame.status == 'closed']
    reasons = {}

    def undefined(key, reason):
        reasons[key] = reason
        return None

    daily_returns = daily.loc[daily.usable_for_sharpe, 'return_on_initial']
    if len(daily_returns) < 2:
        sharpe = undefined('sharpe', 'Need at least two complete UTC days without missing-bar carry PnL')
    elif daily_returns.std(ddof=1) == 0:
        sharpe = undefined('sharpe', 'Daily return sample standard deviation is zero')
    else:
        sharpe = float(daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(365))
    elapsed_years = (int(times[-1]) + step - int(times[0])) / 1e9 / YEAR_SECONDS
    net = float(equity[-1] - capital)
    metrics = {
        'gross_profit_vnd': float(ledger.gross_pnl_usdt.iat[-1] * fx),
        'net_profit_vnd': net,
        'annualized_simple_return': net / capital / elapsed_years,
        'sharpe': sharpe,
        'max_drawdown_vnd': float(ledger.drawdown_vnd.max()),
        'max_drawdown_pct_initial': float(ledger.drawdown_pct.max()),
        'win_rate': float((closed.net_pnl_usdt > 0).mean()) if len(closed) else undefined('win_rate', 'No closed trades'),
        'closed_trade_count': len(closed), 'open_trade_count': int(len(trade_frame)-len(closed)),
        'average_net_pnl_per_closed_trade_vnd': float(closed.net_pnl_vnd.mean()) if len(closed) else undefined('average_net_pnl_per_closed_trade_vnd', 'No closed trades'),
        'average_closed_holding_duration': pd.to_timedelta(closed.holding_duration).mean() if len(closed) else undefined('average_closed_holding_duration', 'No closed trades'),
        'average_daily_margin_turnover_on_initial': float(daily.margin_turnover_on_initial.mean()),
        'total_fees_vnd': total_fees * fx, 'total_slippage_vnd': total_slip * fx,
        'total_funding_cashflow_vnd': total_funding * fx if funding_events is not None else undefined('total_funding_cashflow_vnd', 'Funding data not supplied; net profit excludes funding'),
        'reference_gross_profit_before_slippage_vnd': float(ledger.gross_pnl_usdt.iat[-1] * fx + total_slip * fx),
        'elapsed_years': elapsed_years,
    }
    gap_rows = np.flatnonzero(gap_counts)
    gaps = pd.DataFrame({'previous_open': idx[gap_rows-1], 'next_observed_open': idx[gap_rows],
                         'missing_bars': gap_counts[gap_rows]})
    cost_scope = ('Fees and fill-price slippage included; funding NOT included' if funding_events is None
                  else 'Fees, fill-price slippage and supplied funding events included; funding coverage unverified')
    diagnostics = {'cost_scope': cost_scope, 'funding_included': funding_events is not None,
                   'funding_coverage_verified': False, 'gap_count': len(gaps),
                   'missing_bar_count': int(gap_counts.sum()),
                   'equity_non_positive_bars': int(nonpositive.sum()),
                   'margin_exceeds_equity_bars': int(margin_excess.sum()),
                   'last_target': int(targets[-1]), 'last_target_requires_unavailable_fill': bool(targets[-1] != direction),
                   'sharpe_days_used': len(daily_returns),
                   'sharpe_days_excluded': int((~daily.usable_for_sharpe).sum()),
                   'daily_turnover_policy': 'Mean over observed UTC days, including partial boundary days; wholly missing days excluded',
                   'margin_diagnostic': 'Fixed entry margin budget compared with equity at observed opens/closes and supplied funding marks; no buying-power enforcement or liquidation'}
    return BacktestResult(config, metrics, reasons, trade_frame, ledger, event_frame, daily, gaps, diagnostics)

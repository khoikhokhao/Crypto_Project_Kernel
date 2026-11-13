"""Shared notebook utilities for close-signal, next-open crypto research."""
from itertools import product
from time import perf_counter
from pathlib import Path

import numpy as np
import pandas as pd

from research.load_data import load_ohlcv
from bt_crypto_scan import scan_risk_grid


def load_bars(asset, start, end, timeframe='5min'):
    """Load 1m data, validate timestamps/OHLCV, retain only complete buckets."""
    step = pd.Timedelta(timeframe)
    minute = pd.Timedelta('1min')
    if step < minute or step.value % minute.value:
        raise ValueError('timeframe phải là bội số nguyên của 1 phút')
    raw = load_ohlcv(asset, start=start, end=end, columns=[
        'timestamp', 'open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume'])
    if raw.empty:
        raise ValueError('Không có dữ liệu trong khoảng đã chọn')
    idx = pd.DatetimeIndex(raw.timestamp).as_unit('ns')
    if (str(idx.tz) != 'UTC' or not idx.is_unique or not idx.is_monotonic_increasing or idx.hasnans
            or not np.array_equal(idx.asi8, raw.open_time.to_numpy() * 1_000_000)
            or np.any(idx.asi8 % minute.value) or not (raw.close_time - raw.open_time == 59_999).all()):
        raise ValueError('Timestamp nguồn phải là đầu nến 1m UTC, tăng dần và duy nhất')
    source = raw.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']].copy()
    v = source.to_numpy(dtype=float)
    if (not np.isfinite(v).all() or (v[:, :4] <= 0).any() or (source.volume < 0).any()
            or (source.high < source[['open', 'low', 'close']].max(axis=1)).any()
            or (source.low > source[['open', 'high', 'close']].min(axis=1)).any()):
        raise ValueError('OHLCV nguồn không hợp lệ')
    grouped = source.resample(timeframe, label='left', closed='left', origin='epoch')
    count = grouped.close.count()
    bars = grouped.agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'})
    bars = bars.loc[count.eq(step.value // minute.value)].copy()
    if bars.empty:
        raise ValueError('Không có bucket hoàn chỉnh')
    print(f'{asset}: {len(source):,} nến 1m → {len(bars):,} nến {timeframe}; '
          f'gap: {bars.index.to_series().diff().gt(step).sum()}')
    print('Open đầu/cuối:', bars.index[0], bars.index[-1])
    return bars


def features_by_segment(bars, timeframe, calculate, params):
    segments = bars.index.to_series().diff().gt(pd.Timedelta(timeframe)).cumsum()
    features = pd.concat([calculate(block, params) for _, block in bars.groupby(segments, sort=False)])
    if not features.index.equals(bars.index):
        raise ValueError('Features phải giữ nguyên index OHLCV')
    return features


def build_targets(bars, features, p, slippage_bps):
    # Chỉ dùng Open hiện tại sau khi fill đã xảy ra; TP/SL xét H/L khi nến đóng.
    # Nếu cùng chạm TP và SL: ưu tiên SL. Lệnh thoát fill next observed Open.
    op = bars.open.to_numpy(dtype=float)
    high = bars.high.to_numpy(dtype=float)
    low = bars.low.to_numpy(dtype=float)
    long_sig = features.sig_long.to_numpy(dtype=bool)
    short_sig = features.sig_short.to_numpy(dtype=bool)
    target = np.zeros(len(bars), dtype=np.int8)
    actual_position = np.zeros(len(bars), dtype=np.int8)
    actual_entry = np.full(len(bars), np.nan)
    reason = np.full(len(bars), "flat", dtype=object)
    held = 0
    entry_fill = np.nan
    entry_bar = -1
    slip = slippage_bps / 10_000
    for i in range(len(bars)):
        # Target từ Close trước được thực hiện tại Open hiện tại.
        desired = int(target[i - 1]) if i else 0
        if desired != held:
            held = desired
            if held:
                entry_fill = op[i] * (1 + held * slip)
                entry_bar = i
            else:
                entry_fill = np.nan
                entry_bar = -1
        actual_position[i] = held
        actual_entry[i] = entry_fill
        if held:
            hold_bars = i - entry_bar + 1
            if held == 1:
                stop_hit = low[i] <= entry_fill * (1 - p["sl"])
                profit_hit = high[i] >= entry_fill * (1 + p["tp"])
            else:
                stop_hit = high[i] >= entry_fill * (1 + p["sl"])
                profit_hit = low[i] <= entry_fill * (1 - p["tp"])
            if stop_hit or profit_hit or hold_bars >= p["max_hold"]:
                target[i] = 0
                reason[i] = "sl" if stop_hit else ("tp" if profit_hit else "max_hold")
            else:
                target[i] = held
                reason[i] = "hold"
        elif long_sig[i]:
            target[i] = 1
            reason[i] = "entry_long"
        elif short_sig[i]:
            target[i] = -1
            reason[i] = "entry_short"
    index = bars.index
    return (pd.Series(target, index=index, name="target_position"),
            pd.Series(actual_position, index=index, name="expected_position"),
            pd.Series(actual_entry, index=index, name="expected_entry_fill"),
            pd.Series(reason, index=index, name="decision_reason"))


def run_grid(bars, config, grid, *, start, end, calculate_features, entry_conditions,
             feature_keys, signal_keys, valid_combo=None, workers=8, batch_size=256,
             save_dir=None, filename_prefix='crypto_grid'):
    """Scan fixed-budget TP/SL/hold for alpha entry conditions; no funding."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start.tz_localize('UTC') if start.tz is None else start.tz_convert('UTC')
    end = end.tz_localize('UTC') if end.tz is None else end.tz_convert('UTC')
    step = pd.Timedelta(config.timeframe)
    if start >= end or start < bars.index[0] or end > bars.index[-1] + step:
        raise ValueError('Khoảng grid phải nằm trong dữ liệu đã load')
    if start.value % step.value or end.value % step.value:
        raise ValueError('Thời gian grid phải căn đầu nến UTC')
    risk_keys = ('tp', 'sl', 'max_hold')
    keys = list(grid)
    if set(keys) != set(feature_keys) | set(signal_keys) | set(risk_keys):
        raise ValueError('GRID phải gồm đúng feature_keys, signal_keys, tp, sl, max_hold')
    if any(not isinstance(v, (list, tuple)) or not v for v in grid.values()):
        raise ValueError('Mỗi tham số GRID phải là list/tuple không rỗng')
    sample = bars.loc[(bars.index >= start) & (bars.index < end)].copy()
    if sample.empty:
        raise ValueError('Không có nến trong khoảng grid')
    groups = {}
    count = 0
    for values in product(*(grid[k] for k in keys)):
        p = dict(zip(keys, values))
        if valid_combo is not None and not valid_combo(p):
            continue
        key = tuple(p[k] for k in (*feature_keys, *signal_keys))
        groups.setdefault(key, []).append({k:p[k] for k in risk_keys})
        count += 1
    if not count:
        raise ValueError('Không có tổ hợp hợp lệ')
    print(f'Grid IS: [{start}, {end}) | {len(sample):,} nến | {count:,} cấu hình; funding chưa bao gồm')
    rows, cached_feature, features = [], None, None
    began = perf_counter()
    completed = 0
    for i, (group_key, risk_rows) in enumerate(groups.items(), 1):
        p = dict(zip((*feature_keys, *signal_keys), group_key))
        feature_key = tuple(p[k] for k in feature_keys)
        if feature_key != cached_feature:
            features = features_by_segment(sample, config.timeframe, calculate_features, p)
            cached_feature = feature_key
        long_signal, short_signal = entry_conditions(features, p)
        if not long_signal.index.equals(sample.index) or not short_signal.index.equals(sample.index):
            raise ValueError('Index signals phải khớp OHLCV')
        if long_signal.isna().any() or short_signal.isna().any() or (long_signal & short_signal).any():
            raise ValueError('Signals phải rõ ràng, không NaN và không đồng thời LONG/SHORT')
        signal = pd.Series(np.where(long_signal, 1, np.where(short_signal, -1, 0)),
                           index=sample.index, dtype='int8')
        batch = scan_risk_grid(ohlcv=sample, entry_signals=signal, risk_grid=pd.DataFrame(risk_rows),
                               config=config, workers=workers, batch_size=batch_size)
        for name, value in p.items():
            batch[name] = value
        rows.append(batch)
        completed += len(risk_rows)
        print(f'Nhóm {i}/{len(groups)} | {completed}/{count} cấu hình | {perf_counter()-began:.1f}s', flush=True)
    results = pd.concat(rows, ignore_index=True)[keys + [
        'sharpe', 'mdd_pct_initial', 'annualized_simple_return', 'closed_trades']]
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S_%f')
        path = save_dir / f'{filename_prefix}_{stamp}.csv'
        results.to_csv(path, index=False)
        print('Đã lưu:', path)
    print('MDD và annualized return là tỷ lệ trên vốn ban đầu; 0.10 = 10%.')
    return results

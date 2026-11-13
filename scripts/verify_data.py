"""Offline audit of actual Parquet files against the latest requested UTC window."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from download_ohlcv import validate

ROOT = Path(__file__).resolve().parents[1]


def main():
    report_path = ROOT / 'data/reports/quality_latest.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    start = int(pd.Timestamp(report['start_inclusive_utc']).timestamp()*1000)
    end = int(pd.Timestamp(report['end_exclusive_utc']).timestamp()*1000)
    first_month = pd.Timestamp(start, unit='ms', tz='UTC').normalize().replace(day=1)
    months = pd.date_range(first_month, pd.Timestamp(end-1, unit='ms', tz='UTC'), freq='MS')
    summaries = []
    for summary in report['summaries']:
        symbol, total = summary['symbol'], 0
        for month in months:
            left = max(start, int(month.timestamp()*1000))
            right = min(end, int((month+pd.offsets.MonthBegin(1)).timestamp()*1000))
            path = ROOT / f'data/processed/binance_um/1m/{symbol}/{month:%Y-%m}.parquet'
            frame = pd.read_parquet(path)
            validate(frame.drop(columns='timestamp'))
            frame = frame[(frame.open_time >= left) & (frame.open_time < right)].reset_index(drop=True)
            expected = np.arange(left, right, 60000, dtype='int64')
            if not np.array_equal(frame.open_time.to_numpy(), expected):
                raise ValueError(f'Missing, unordered, or extra timestamps: {path}')
            timestamps = pd.to_datetime(frame.open_time, unit='ms', utc=True)
            if not frame.timestamp.equals(timestamps.rename('timestamp')):
                raise ValueError(f'Datetime/millisecond mismatch: {path}')
            total += len(frame)
        if total != (end-start)//60000 or total != summary['rows']:
            raise ValueError(f'Total row mismatch: {symbol}')
        summaries.append({'symbol':symbol, 'rows':total, 'partitions':len(months),
                          'missing_minutes':0, 'duplicate_minutes':0, 'valid_ohlcv':True})
        print(f'{symbol}: {total:,} rows, {len(months)} partitions OK', flush=True)
    output = {'verified_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),
              'start_inclusive_utc':report['start_inclusive_utc'],
              'end_exclusive_utc':report['end_exclusive_utc'], 'summaries':summaries}
    (ROOT/'data/reports/verification_latest.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print('PASS: all stored candles match the requested minute grid.', flush=True)


if __name__ == '__main__':
    main()
